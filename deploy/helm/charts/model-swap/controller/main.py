"""DynamoModelSwap controller: the Kubernetes and HTTP half. Every 10 s, for each DynamoModelSwap in its namespace, it
observes the slots (their DGDs and pods) and what the frontend graph sees, asks logic.decide() for one step, applies
it, and writes status. All state lives in the CR's status, so a restart picks up mid-swap.

A plain loop, not kopf: kopf timers put a finalizer on the CR, and a finalizer left behind by a controller deleted in
the same sync (parking the lab) would hold the CR forever.

"The frontend acknowledges" a slot when every frontend pod of the frontend graph (1) lists the slot's prefill and decode
`generate` instances in /health (its discovery sees them) and (2) has routed a request to the slot's decode instance
(/metrics has a dynamo_frontend_worker_* series for its instance id). (2) comes from live traffic or, without
traffic, from the controller's own probe requests.
"""
from __future__ import annotations

import http.server
import json
import os
import re
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from kubernetes import client, config
from kubernetes.client.rest import ApiException

import logic

NS = os.environ["WATCH_NAMESPACE"]
INTERVAL = 10
DGD_GROUP, DGD_VERSION, DGD_PLURAL = "nvidia.com", "v1beta1", "dynamographdeployments"
HISTORY = 40
G_LABEL = "nvidia.com/dynamo-graph-deployment-name"
last_ok = time.time()


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def http_get(url: str, timeout: float = 5) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode(errors="replace") if r.status == 200 else None
    except Exception:
        return None


def probe(fe_ip: str, served: str) -> str:
    body = json.dumps({"model": served, "max_tokens": 16, "temperature": 0,
                       "messages": [{"role": "user", "content": "Count from 1 to 20, separated by spaces."}]}).encode()
    req = urllib.request.Request(f"http://{fe_ip}:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return str(r.status)
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:120]


def frontends(s: dict) -> list:
    pods = client.CoreV1Api().list_namespaced_pod(
        NS, label_selector=f"{G_LABEL}={s['frontendGraph']},"
                           "nvidia.com/dynamo-component-type=frontend").items
    return [p for p in pods if logic.pod_state(_api(p)) == "ready" and p.status.pod_ip]


def frontend_view(fe_ips: list[str]) -> list[dict]:
    """Per frontend pod: {dynamo namespace: {component: instance_id}} of generate endpoints, and the worker ids it
    has routed to."""
    views = []
    for ip in fe_ips:
        h, m = http_get(f"http://{ip}:8000/health"), http_get(f"http://{ip}:8000/metrics")
        inst: dict = {}
        try:
            for i in (json.loads(h).get("instances") or []) if h else []:
                if i.get("endpoint") == "generate":
                    inst.setdefault(i["namespace"], {})[i["component"]] = str(i["instance_id"])
        except (ValueError, KeyError):
            pass
        routed = set(re.findall(r'^dynamo_frontend_worker_last_\w+\{[^}]*worker_id="(\d+)"', m or "", re.M))
        views.append({"instances": inst, "routed": routed})
    return views


def slot_view(slot: str, views: list[dict]) -> tuple[bool, bool, str | None]:
    """(listed, acked, decode instance id) of one slot across all frontend pods."""
    if not views:
        return False, False, None
    listed = acked = True
    decode_id = None
    for v in views:
        comps = {}
        for ns, c in v["instances"].items():
            if ns.rsplit("-", 1)[0] == f"{NS}-{slot}":
                comps.update(c)
        if "prefill" not in comps or "backend" not in comps:
            listed = acked = False
            continue
        decode_id = comps["backend"]
        if decode_id not in v["routed"]:
            acked = False
    return listed, acked and listed, decode_id


def merge_diff(old: dict, new: dict) -> dict:
    """The JSON merge patch that turns old into new (a key missing from new becomes null, so it is removed)."""
    out = {}
    for k in set(old) | set(new):
        o, n = old.get(k), new.get(k)
        if o != n:
            out[k] = merge_diff(o, n) if isinstance(o, dict) and isinstance(n, dict) else n
    return out


def retired(dgd: dict | None) -> bool:
    return any(c.get("type") == "prefill" and c.get("replicas") == 0
               for c in ((dgd or {}).get("spec") or {}).get("components") or [])


def _api(obj) -> dict:
    """kubernetes-client object -> API-shaped (camelCase) dict."""
    return client.ApiClient().sanitize_for_serialization(obj)


def reconcile(cr: dict) -> None:
    name, spec = cr["metadata"]["name"], cr.get("spec") or {}
    status = {k: v for k, v in (cr.get("status") or {}).items() if k != "history"}
    s = logic.with_defaults(spec)
    co, core = client.CustomObjectsApi(), client.CoreV1Api()
    sel = f"{logic.L_SWAP}={name}"
    dgds = {d["metadata"]["name"]: d for d in
            co.list_namespaced_custom_object(DGD_GROUP, DGD_VERSION, NS, DGD_PLURAL, label_selector=sel)["items"]}
    # Pods by the graph label the Dynamo operator sets on every pod of a DGD (podTemplate labels are not relied on).
    mine = re.compile(rf"^{re.escape(s['frontendGraph'])}-{re.escape(name)}-[fb]\d+$")
    pods: dict = {}
    for p in core.list_namespaced_pod(NS, label_selector=G_LABEL).items:
        g = (p.metadata.labels or {}).get(G_LABEL)
        if g in dgds or mine.match(g or ""):
            pods.setdefault(g, []).append(_api(p))
    fe = frontends(s)
    views = frontend_view([p.status.pod_ip for p in fe])
    slots = {}
    for sl in set(dgds) | {k for k in pods if k}:
        listed, acked, decode_id = slot_view(sl, views)
        prefill_pods = [p for p in pods.get(sl, [])
                        if (p["metadata"].get("labels") or {}).get("nvidia.com/dynamo-component-type") == "prefill"]
        slots[sl] = {"state": logic.slot_state(dgds.get(sl), pods.get(sl, [])), "listed": listed, "acked": acked,
                     "decode": decode_id, "retired": retired(dgds.get(sl)),
                     "prefill": "present" if prefill_pods else "missing"}

    cache: list = []

    def preflight(pct: int):
        if not cache:
            cache.extend(_api(p) for p in core.list_pod_for_all_namespaces().items)
        return logic.gpu_preflight(cache, logic.graph_gpus(s), int(pct))

    acts, new = logic.decide(name, spec, status, slots, preflight, time.time())

    for act in acts:
        if act[0] == "create":
            body = logic.render_slot(name, NS, s, act[2], act[1])
            # Owner reference only, no ArgoCD tracking label: ArgoCD shows the slot under the CR and never prunes it.
            body["metadata"]["ownerReferences"] = [{"apiVersion": cr["apiVersion"], "kind": cr["kind"], "name": name,
                                                    "uid": cr["metadata"]["uid"], "controller": True}]
            try:
                co.create_namespaced_custom_object(DGD_GROUP, DGD_VERSION, NS, DGD_PLURAL, body)
                log(f"{name}: created {act[1]} slot {act[2]} ({s['model']})")
            except ApiException as e:
                if e.status != 409:
                    raise
        elif act[0] == "delete":
            try:
                co.delete_namespaced_custom_object(DGD_GROUP, DGD_VERSION, NS, DGD_PLURAL, act[1],
                                                   propagation_policy="Background")
                log(f"{name}: deleted slot {act[1]}")
            except ApiException as e:
                if e.status != 404:
                    raise
        elif act[0] == "retire":
            # The prefill to 0 replicas (replicas are not in Dynamo's worker hash, so no new generation). A merge patch
            # replaces the whole components list, so send it back whole with only that field changed.
            d = co.get_namespaced_custom_object(DGD_GROUP, DGD_VERSION, NS, DGD_PLURAL, act[1])
            comps = d["spec"]["components"]
            for c in comps:
                if c.get("type") == "prefill":
                    c["replicas"] = 0
            co.patch_namespaced_custom_object(DGD_GROUP, DGD_VERSION, NS, DGD_PLURAL, act[1],
                                              {"spec": {"components": comps}})
            log(f"{name}: retired slot {act[1]}: prefill scaled to 0")
        elif act[0] == "probe":
            with ThreadPoolExecutor(8) as ex:
                res = list(ex.map(lambda ip: probe(ip, s["servedModelName"]), [p.status.pod_ip for p in fe] * 4))
            log(f"{name}: probe for {act[1]}: {res}")

    new["slots"] = {k: {kk: v[kk] for kk in ("state", "listed", "acked", "retired")} for k, v in sorted(slots.items())}
    new["frontends"] = len(fe)
    history = list((cr.get("status") or {}).get("history") or [])
    if new.get("phase") != status.get("phase") or (new.get("message") != status.get("message")
                                                    and new.get("phase") in ("Serving", "Degraded", "Failed")):
        history.append({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": new.get("phase"),
                        "message": new.get("message")})
        log(f"{name}: {status.get('phase')} -> {new.get('phase')}: {new.get('message')}")
    new["history"] = history[-HISTORY:]
    patch = merge_diff(cr.get("status") or {}, new)
    if patch:
        co.patch_namespaced_custom_object_status(logic.GROUP, logic.VERSION, NS, logic.PLURAL, name,
                                                 {"status": patch})


def loop() -> None:
    global last_ok
    co = client.CustomObjectsApi()
    while True:
        t0 = time.time()
        try:
            for cr in co.list_namespaced_custom_object(logic.GROUP, logic.VERSION, NS, logic.PLURAL)["items"]:
                if cr["metadata"].get("deletionTimestamp"):
                    continue  # the slots go with it (owner references)
                try:
                    reconcile(cr)
                except Exception:
                    log(f"{cr['metadata']['name']}: reconcile failed\n{traceback.format_exc()}")
            last_ok = time.time()
        except Exception:
            log(f"list failed\n{traceback.format_exc()}")
        time.sleep(max(1.0, INTERVAL - (time.time() - t0)))


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = time.time() - last_ok < 180
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(b"ok\n" if ok else b"stale\n")

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    threading.Thread(target=http.server.ThreadingHTTPServer(("0.0.0.0", 8080), Health).serve_forever,
                     daemon=True).start()
    log(f"watching {logic.PLURAL}.{logic.GROUP} in {NS}")
    loop()
