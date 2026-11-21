"""DynamoModelSwap controller: the Kubernetes and HTTP half. Every 10 s, for each DynamoModelSwap in its namespace, it
observes the slots (their DGDs and pods) and what the frontend graph sees, asks logic.decide() for one step, applies
it, and writes status. All state lives in the CR's status, so a restart picks up mid-swap.

A plain loop, not kopf: kopf timers put a finalizer on the CR, and a finalizer left behind by a controller deleted in
the same sync (parking the lab) would hold the CR forever.

"The frontend acknowledges" a slot when every frontend pod of the frontend graph (1) lists the slot's prefill and decode
`generate` instances in /health (its discovery sees them) and (2) has routed a request to the slot's decode instance
(/metrics has a dynamo_frontend_worker_* series for its instance id). (2) comes from live traffic or, without
traffic, from the controller's own probe requests.

The download gate and the model links run as Jobs, one per node of the graph (fetch.py). An in-place reload calls the
workers' admin routes (/engine/* on their system ports), which admin-port-guard only lets through from a node's own
host network: this pod runs hostNetwork for that.
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
L_JOB, L_NODE = f"{logic.GROUP}/job", f"{logic.GROUP}/node"
PAUSE_CAP = float(os.environ.get("PAUSE_CAP_SECONDS", "60"))
FETCH_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch.py")).read()
# The namespace's compile-cache store (a ReadWriteMany claim the chart makes; empty: no seeding), mounted by the
# cache-seed and cache-publish Jobs at /store, one directory per namespace.
STORE_PVC = os.environ.get("CACHE_STORE_PVC", "")
logic.CACHE_TIMEOUT = float(os.environ.get("CACHE_TIMEOUT_SECONDS") or logic.CACHE_TIMEOUT)
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
    """"200" when the frontend answers and the answer counts (greedy "1 2 3 4 5 ..."): a canary for weights that
    load but are garbage. Else what went wrong."""
    body = json.dumps({"model": served, "max_tokens": 24, "temperature": 0,
                       "messages": [{"role": "user", "content": "Count from 1 to 20, separated by spaces."}]}).encode()
    req = urllib.request.Request(f"http://{fe_ip}:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            text = (json.loads(r.read()).get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if re.search(r"1\D+2\D+3\D+4\D+5", text):
                return str(r.status)
            return f"{r.status} but no count: {text[:40]!r}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:120]


# ---------------------------------------------------------------- node Jobs (download gate, model links)

def _main_container(comp: dict) -> dict:
    cs = ((comp.get("podTemplate") or {}).get("spec") or {}).get("containers") or [{}]
    return next((c for c in cs if c.get("name") == "main"), cs[0])


def job_body(cr: dict, s: dict, key: str, mode: str, params: dict, node: str, i: int) -> dict:
    comp = next(iter(logic.workers(s).values()))
    pod_spec = (comp.get("podTemplate") or {}).get("spec") or {}
    f = s["fetch"]
    name = cr["metadata"]["name"]
    env = [{"name": "MODE", "value": mode}, {"name": "HF_HOME", "value": logic.HF_ROOT},
           {"name": "HF_ENDPOINT", "value": f["hfEndpoint"]}, {"name": "HF_HUB_DISABLE_XET", "value": "1"},
           # the image sets NVIDIA_VISIBLE_DEVICES=all: no GPUs for a download
           {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
           {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": f["tokenSecret"], "key": "token",
                                                               "optional": True}}}]
    volumes = [{"name": "hf", "hostPath": {"path": logic.HF_ROOT, "type": "Directory"}}]
    mounts = [{"name": "hf", "mountPath": logic.HF_ROOT}]
    deadline = int(f["timeoutSeconds"])
    if mode == "fetch":
        env += [{"name": "MODEL", "value": params["model"]}, {"name": "REVISION", "value": params["revision"]},
                {"name": "COMPARE", "value": params["compare"]}]
    elif mode in ("cache-seed", "cache-publish"):
        # The node's compile cache and the namespace's store; the copy stops 30 s before the controller stops waiting.
        env = [{"name": "MODE", "value": mode}, {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
               {"name": "CACHE_DIR", "value": "/cache"}, {"name": "STORE_DIR", "value": f"/store/{NS}"},
               {"name": "CACHE_SECONDS", "value": str(max(30, int(logic.CACHE_TIMEOUT) - 30))}]
        volumes = [{"name": "cache", "hostPath": {"path": logic.cache_dir(s, node), "type": "DirectoryOrCreate"}},
                   {"name": "store", "persistentVolumeClaim": {"claimName": STORE_PVC}}]
        mounts = [{"name": "cache", "mountPath": "/cache", "readOnly": mode == "cache-publish"},
                  {"name": "store", "mountPath": "/store"}]
        deadline = int(logic.CACHE_TIMEOUT) + 60
    else:
        env += [{"name": "LINK", "value": logic.link_path(NS, name, params["kind"])},
                {"name": "TARGET", "value": params["target"]}]
    labels = {logic.L_SWAP: name, L_JOB: key, L_NODE: node}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": f"dms-{name}-{key}-{i}", "namespace": NS, "labels": labels,
                     "ownerReferences": [{"apiVersion": cr["apiVersion"], "kind": cr["kind"], "name": name,
                                          "uid": cr["metadata"]["uid"], "controller": True}]},
        "spec": {
            "backoffLimit": 0, "ttlSecondsAfterFinished": 1800, "activeDeadlineSeconds": deadline,
            "template": {"metadata": {"labels": labels}, "spec": {
                "restartPolicy": "Never",
                "nodeSelector": {"kubernetes.io/hostname": node},
                "tolerations": [{"operator": "Exists"}],
                "imagePullSecrets": pod_spec.get("imagePullSecrets") or [],
                "volumes": volumes,
                "containers": [{
                    "name": mode, "image": f.get("image") or _main_container(comp)["image"],
                    "imagePullPolicy": "IfNotPresent", "command": ["python3", "-c", FETCH_SRC], "env": env,
                    "securityContext": {"runAsUser": 0},
                    "resources": {"requests": {"cpu": "1", "memory": "1Gi"}, "limits": {"memory": "8Gi"}},
                    "volumeMounts": mounts}]}}}}


def ensure_job(cr: dict, s: dict, key: str, mode: str, params: dict) -> None:
    batch = client.BatchV1Api()
    for i, node in enumerate(logic.graph_nodes(s)):
        try:
            batch.create_namespaced_job(NS, job_body(cr, s, key, mode, params, node, i))
            log(f"{cr['metadata']['name']}: job {key} ({mode}) on {node}")
        except ApiException as e:
            if e.status != 409:
                raise


def job_views(name: str) -> dict:
    """{key: {state, result, error}} with every node's Job of a key merged: failed if one failed, succeeded once all
    did. A fetch's nodes must hold the same snapshot (revision) of the model."""
    sel = f"{logic.L_SWAP}={name},{L_JOB}"
    msgs = {}
    for p in client.CoreV1Api().list_namespaced_pod(NS, label_selector=sel).items:
        for cs in p.status.container_statuses or []:
            if cs.state and cs.state.terminated and cs.state.terminated.message:
                msgs[(p.metadata.labels or {}).get("job-name")] = cs.state.terminated.message
    by_key: dict = {}
    for j in client.BatchV1Api().list_namespaced_job(NS, label_selector=sel).items:
        lb = j.metadata.labels or {}
        st = j.status
        failed = (st.failed or 0) > 0 or any(c.type == "Failed" and c.status == "True" for c in st.conditions or [])
        state = "succeeded" if (st.succeeded or 0) > 0 else "failed" if failed else "running"
        try:
            res = json.loads(msgs.get(j.metadata.name) or "{}")
        except ValueError:
            res = {"error": (msgs.get(j.metadata.name) or "")[:300]}
        by_key.setdefault(lb.get(L_JOB), []).append((lb.get(L_NODE), state, res))
    views = {}
    for key, items in by_key.items():
        bad = [f"{n}: {r.get('error') or 'job failed'}" for n, st, r in items if st == "failed"]
        results = {n: {k: v for k, v in r.items() if k in ("copied", "bytes", "present", "busy", "failed", "errors", "note",
                                                          "timedOut", "error")} for n, _, r in items if r}
        if bad:
            views[key] = {"state": "failed", "error": "; ".join(bad)[:500], "results": results}
        elif all(st == "succeeded" for _, st, _ in items):
            rs = [r for _, _, r in items]
            snaps = {r.get("snapshot") for r in rs if "snapshot" in r}
            if len(snaps) > 1:
                views[key] = {"state": "failed", "error": f"nodes hold different revisions: {sorted(snaps)}"}
                continue
            notc = [r for r in rs if "compatible" in r and not r["compatible"]]
            views[key] = {"state": "succeeded", "results": results, "result": {
                "snapshot": snaps.pop() if snaps else None,
                "compatible": bool(rs) and all(r.get("compatible") for r in rs),
                "why": (notc[0] if notc else rs[0]).get("why", "")}}
        else:
            views[key] = {"state": "running"}
    return views


# ---------------------------------------------------------------- in-place reload (the workers' admin routes)

def admin(addr: tuple, route: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"http://{addr[0]}:{addr[1]}/engine/{route}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def slot_workers(slot: str) -> dict:
    """{role: (host ip, system port)} of a slot's Ready prefill and decode pods (hostNetwork: pod IP = host IP)."""
    out = {}
    for p in client.CoreV1Api().list_namespaced_pod(NS, label_selector=f"{G_LABEL}={slot}").items:
        role = (p.metadata.labels or {}).get("nvidia.com/dynamo-component-type")
        if role not in logic.ROLES or logic.pod_state(_api(p)) != "ready":
            continue
        c = next((c for c in p.spec.containers if c.name == "main"), p.spec.containers[0])
        port = next((e.value for e in c.env or [] if e.name == "DYN_SYSTEM_PORT"), None)
        if role in out or not port:
            raise RuntimeError(f"slot {slot}: expected one Ready {role} pod with DYN_SYSTEM_PORT")
        out[role] = (p.status.pod_ip, int(port))
    if set(out) != set(logic.ROLES):
        raise RuntimeError(f"slot {slot}: Ready workers are {sorted(out)}, not prefill + decode")
    return out


def reload_in_place(slot: str, p: dict, fe_ips: list[str], served: str) -> dict:
    """Pause prefill then decode (wait mode: running requests finish, new ones queue; at most PAUSE_CAP s each), load
    p["to"] into both, resume decode then prefill, then check both report the new version and the frontends answer.
    Any failure loads p["from"] back. Returns the outcome decide() reads from status["reload"]."""
    res = {"hash": p["hash"], "ok": False, "to": p["to"]}
    roles = ("prefill", "decode")
    try:
        w = slot_workers(slot)
    except Exception as e:
        return {**res, "why": str(e)[:300]}

    def each(route, body, timeout, order=roles):
        out = {}
        with ThreadPoolExecutor(2) as ex:
            futs = {r: ex.submit(admin, w[r], route, body, timeout) for r in order}
            for r, fu in futs.items():
                try:
                    out[r] = fu.result()
                except Exception as e:
                    out[r] = {"status": "error", "message": f"{type(e).__name__}: {e}"[:200]}
        return out

    def resume():                       # decode first, so the prefill's first new requests find it ready
        return {r: admin(w[r], "resume_generation", {}, 600) for r in ("decode", "prefill")}

    def back(why):
        """Old weights back in (the engines must be paused), resume, and report why."""
        rb = each("update_weights_from_disk", {"model_path": p["from"], "weight_version": "rollback"}, 900)
        rs = resume()
        ok = all(x.get("status") == "ok" for x in [*rb.values(), *rs.values()])
        log(f"{slot}: rolled back to {p['from']}: {'ok' if ok else rb}")
        return {**res, "why": why[:300], "rolledBack": ok}

    t0 = time.time()
    for r in roles:
        try:
            x = admin(w[r], "pause_generation", {"mode": "wait"}, PAUSE_CAP)
            if x.get("status") != "ok":
                raise RuntimeError(x.get("message"))
        except Exception as e:
            resume()                    # a pause still draining finishes first, then this resumes it
            return {**res, "why": f"pause {r} did not finish in {PAUSE_CAP:.0f} s: {e}"[:300]}
    t_paused = time.time()
    up = each("update_weights_from_disk", {"model_path": p["to"], "weight_version": p["version"]}, 900)
    t_loaded = time.time()
    if any(x.get("status") != "ok" for x in up.values()):
        return back(f"update_weights_from_disk: {up}")
    resume()
    t_resumed = time.time()
    versions = {r: admin(w[r], "get_weight_version", {}, 10).get("version") for r in roles}
    probes = [probe(ip, served) for ip in fe_ips for _ in range(2)]
    if any(v != p["version"] for v in versions.values()) or any(x != "200" for x in probes) or not probes:
        for r in roles:
            admin(w[r], "pause_generation", {"mode": "wait"}, max(PAUSE_CAP, 120))
        return back(f"check failed: versions {versions}, probes {probes}")
    return {**res, "ok": True, "pausedSeconds": round(t_resumed - t0, 1), "drainSeconds": round(t_paused - t0, 1),
            "loadSeconds": round(t_loaded - t_paused, 1), "probes": probes}


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

    acts, new = logic.decide(name, spec, status, slots, preflight, time.time(), job_views(name), bool(STORE_PVC))

    for act in acts:
        if act[0] == "job":
            ensure_job(cr, s, act[1], act[2], act[3])
        elif act[0] == "reload":
            log(f"{name}: reloading {act[1]} in place: {act[2]['from']} -> {act[2]['to']}")
            out = reload_in_place(act[1], act[2], [p.status.pod_ip for p in fe], s["servedModelName"])
            new["reload"] = out
            log(f"{name}: in-place reload of {act[1]}: {out}")
            if out["ok"]:
                co.patch_namespaced_custom_object(DGD_GROUP, DGD_VERSION, NS, DGD_PLURAL, act[1], {
                    "metadata": {"annotations": {f"{logic.GROUP}/model": logic.model_ref(s)}}})
        elif act[0] == "create":
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
    threading.Thread(target=http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("HEALTH_PORT", "8080"))), Health).serve_forever,
                     daemon=True).start()
    log(f"watching {logic.PLURAL}.{logic.GROUP} in {NS}")
    loop()
