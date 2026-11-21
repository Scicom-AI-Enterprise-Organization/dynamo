"""DynamoModelSwap: the pure half (no Kubernetes, no HTTP), so the swap order can be tested on its own.

One DynamoModelSwap holds a Dynamo graph's worker spec (a prefill and a decode component, as written in a
DynamoGraphDeployment) plus the model. Edit spec.model and the controller swaps with no downtime, one replica.

Every change starts with the download gate (Fetching): a Job on each node of the graph fetches the new model into
/data/hf if it is missing, checks every shard, and compares it with the serving model. Then one of two paths:

  In place (Reloading), when only spec.model changed, the gate found the two versions identical but for their weights
  (config, tokenizer, chat template, tensor names), spec.reload is Auto and spec.versionAlias is false: the controller
  pauses the running pair (wait mode, at most 60 s), loads the new weights into it, resumes, and checks the version and
  a probe request. Any failure loads the old weights back and falls back to the bridge. Seconds, not minutes.

  Bridge, for everything else:
    full(old, 0.70)                       serving
    + bridge(new, 0.27)                   Bridging     both serve once the frontend routes to the bridge
    - full(old)                           DrainingOld  prefill scaled to 0 first, then the slot; wait until its pods are gone
    + full(new, 0.70)                     Loading      bridge + new full serve
    - bridge(new)                         Finalizing
    full(new, 0.70)                       Serving

Each slot is its own workers-only DynamoGraphDeployment named <frontendGraph>-<swap>-<f|b><gen>, so the frontend
graph's namespace-prefix discovery picks it up. Fulls always use host port set a, bridges set b. A slot loads
--model from its set's link on each node (/data/hf/swap/<namespace>/<swap>-<set> -> a snapshot dir), which the
controller points at the slot's version before it starts or reloads it, so a restarted worker comes back on the
version its slot serves now.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

GROUP = "dynamoswap.scicom.dev"
VERSION = "v1alpha1"
PLURAL = "dynamoswaps"
KIND = "DynamoModelSwap"

L_SWAP = f"{GROUP}/swap"
L_SLOT = f"{GROUP}/slot"
L_KIND = f"{GROUP}/kind"
A_GPUS = "nvidia.com/use-gpuuuid"
R_GPUMEM = "nvidia.com/gpumem-percentage"

ROLES = ("prefill", "decode")
# Host ports (the workers run hostNetwork): index per (port set, role), as tools/gen-slot.sh.
PORT_INDEX = {("a", "prefill"): 0, ("a", "decode"): 1, ("b", "prefill"): 2, ("b", "decode"): 3}
SET_OF = {"full": "a", "bridge": "b"}
# The operator labels every worker pod nvidia.com/selector=<dgd>-<component, lower case>-<8 hex worker hash>, and a
# label value must stay <= 63 characters.
LABEL_MAX = 63
LINK_ROOT = "/data/hf/swap"
HF_ROOT = "/data/hf"

DEFAULTS = {
    "frontendGraph": "lab",
    "steady": {"gpuMemoryUtilization": 0.70, "gpumemPercentage": 70},
    "bridge": {"gpuMemoryUtilization": 0.27, "gpumemPercentage": 27},
    "readyTimeoutSeconds": 1800,
    "fullRetries": 1,
    "versionAlias": True,
    "servedModelAliases": [],
    "revision": "",
    "reload": "Auto",
    "fetch": {"hfEndpoint": "https://serverlessgpu.aifoundry.scicom.dev/hf", "tokenSecret": "aif-hub-token",
              "timeoutSeconds": 3600},
}


def with_defaults(spec: dict) -> dict:
    s = copy.deepcopy(spec or {})
    for k, v in DEFAULTS.items():
        if isinstance(v, dict):
            s[k] = {**v, **(s.get(k) or {})}
        else:
            s.setdefault(k, v)
    return s


def config_hash(s: dict, without: tuple = ()) -> str:
    """What a full slot is: a change in any of these starts a swap. Bridge, reload and fetch settings are not in it."""
    key = {k: s.get(k) for k in ("model", "revision", "servedModelName", "servedModelAliases", "versionAlias",
                                 "steady", "graph") if k not in without}
    for k in ("revision", "servedModelAliases"):     # unset: hash as before they existed, so an upgrade starts no swap
        if not key.get(k):
            key.pop(k, None)
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def shape_hash(s: dict) -> str:
    """The slot without its model: equal shapes mean only the weights would change."""
    return config_hash(s, without=("model", "revision"))


def model_ref(s: dict) -> str:
    """spec.model, plus @revision when one is pinned."""
    return f"{s['model']}@{s['revision']}" if s.get("revision") else s["model"]


def slot_name(s: dict, swap: str, kind: str, gen: int) -> str:
    return f"{s['frontendGraph']}-{swap}-{kind[0]}{gen}"


def workers(s: dict) -> dict:
    """{role: component} from spec.graph.components."""
    return {c.get("type"): c for c in ((s.get("graph") or {}).get("components") or [])}


def validate(s: dict, swap: str) -> str:
    for k in ("model", "servedModelName", "graph"):
        if not s.get(k):
            return f"spec.{k} is required"
    comps = (s["graph"].get("components") or [])
    types = sorted(c.get("type") or "" for c in comps)
    if types != ["decode", "prefill"]:
        return f"spec.graph.components must be exactly one prefill and one decode worker (got {types})"
    for c in comps:
        if not _gpus_of(c):
            return f"component {c.get('name')} has no {A_GPUS} annotation; pin its cards"
        if not _node_of(c):
            return f"component {c.get('name')} has no nodeSelector kubernetes.io/hostname; pin its node"
    total = int(s["steady"]["gpumemPercentage"]) + int(s["bridge"]["gpumemPercentage"])
    if total > 100:
        return f"steady + bridge gpumem is {total}% > 100: the bridge and a full engine must fit on the same cards"
    longest = max(len(str(c.get("name") or "")) for c in comps)
    label = len(slot_name(s, swap, "bridge", 999)) + 1 + longest + 1 + 8
    if label > LABEL_MAX:
        return (f"names too long: {slot_name(s, swap, 'bridge', 999)}-<component>-<hash> is {label} characters "
                f"(> {LABEL_MAX}); shorten the swap name or the component names")
    if s["reload"] not in ("Auto", "Bridge"):
        return f"spec.reload must be Auto or Bridge (got {s['reload']!r})"
    return ""


def _node_of(component: dict) -> str:
    spec = (component.get("podTemplate") or {}).get("spec") or {}
    return (spec.get("nodeSelector") or {}).get("kubernetes.io/hostname") or ""


def graph_nodes(s: dict) -> list[str]:
    """The nodes the slots run on: where the download gate and the model links must be."""
    return sorted({_node_of(c) for c in workers(s).values()})


def link_path(namespace: str, swap: str, kind: str) -> str:
    return f"{LINK_ROOT}/{namespace}/{swap}-{SET_OF[kind]}"


def inplace_ok(s: dict, active: dict | None, cand: dict) -> tuple[bool, str]:
    """Can the active full slot load cand's weights in place instead of bridging? (ok, why not)."""
    if s["reload"] != "Auto":
        return False, "spec.reload is Bridge"
    if s["versionAlias"]:
        return False, "spec.versionAlias is on (a slot's served names cannot change while it runs)"
    if not active or not active.get("snapshot"):
        return False, "the serving slot does not load from a model link (it predates in-place reloads)"
    if active.get("shape") != shape_hash(s):
        return False, "more than spec.model changed"
    if not cand.get("compatible"):
        return False, f"not weight-compatible: {cand.get('why')}"
    return True, ""


def _gpus_of(component: dict) -> list[str]:
    raw = (((component.get("podTemplate") or {}).get("metadata") or {}).get("annotations") or {}).get(A_GPUS) or ""
    return [g.strip() for g in raw.split(",") if g.strip()]


def graph_gpus(s: dict) -> list[str]:
    return sorted({g for c in workers(s).values() for g in _gpus_of(c)})


# ---------------------------------------------------------------- rendering a slot

OWNED_ARGS = ("--model", "--served-model-name", "--gpu-memory-utilization")


def strip_args(args: list[str]) -> list[str]:
    """Drop the flags the controller owns, with their values (--served-model-name takes several)."""
    out, skip = [], False
    for a in args:
        if skip and not str(a).startswith("--"):
            continue
        skip = False
        flag = str(a).split("=", 1)[0]
        if flag in OWNED_ARGS:
            skip = "=" not in str(a)
            continue
        out.append(a)
    return out


def _set_env(container: dict, name: str, value: str) -> None:
    env = [e for e in container.get("env") or [] if e.get("name") != name]
    env.append({"name": name, "value": value})
    container["env"] = env


def render_slot(swap: str, namespace: str, s: dict, slot: str, kind: str) -> dict:
    """The workers-only DynamoGraphDeployment for one slot. kind: full | bridge."""
    pset = SET_OF[kind]
    util = s["steady"] if kind == "full" else s["bridge"]
    names = list(dict.fromkeys([s["servedModelName"], *(s.get("servedModelAliases") or []),
                                *([s["model"]] if s.get("versionAlias") else [])]))
    graph = copy.deepcopy(s["graph"])
    graph.pop("services", None)
    for comp in graph["components"]:
        role = comp["type"]
        i = PORT_INDEX[(pset, role)]
        ports = {"system": 9090 + i, "nixl": 19090 + i, "fpm": 20380 + 10 * i, "side": 5600 + 100 * i,
                 "kv": 20080 + (0 if pset == "a" else 1)}
        pt = comp.setdefault("podTemplate", {})
        md = pt.setdefault("metadata", {})
        md["labels"] = {**(md.get("labels") or {}), L_SWAP: swap, L_SLOT: slot, L_KIND: kind}
        containers = pt.setdefault("spec", {}).setdefault("containers", [{}])
        c = next((x for x in containers if x.get("name") == "main"), containers[0])
        args = strip_args([str(a) for a in c.get("args") or []])
        for j, a in enumerate(args[:-1]):
            if a == "--kv-events-config":
                cfg = json.loads(args[j + 1])
                cfg["endpoint"] = f"tcp://*:{ports['kv']}"
                args[j + 1] = json.dumps(cfg, separators=(",", ":"))
        c["args"] = ["--model", link_path(namespace, swap, kind), "--served-model-name", *names,
                     "--gpu-memory-utilization", f"{float(util['gpuMemoryUtilization']):.2f}", *args]
        for p in c.get("ports") or []:
            if p.get("name") in ("system", "nixl"):
                p["containerPort"] = ports[p["name"]]
        _set_env(c, "DYN_SYSTEM_PORT", str(ports["system"]))
        _set_env(c, "DYN_FORWARDPASS_METRIC_PORT", str(ports["fpm"]))
        _set_env(c, "VLLM_NIXL_SIDE_CHANNEL_PORT", str(ports["side"]))
        if any(e.get("name") == "MODEL" for e in c.get("env") or []):
            _set_env(c, "MODEL", s["model"])
        limits = c.setdefault("resources", {}).setdefault("limits", {})
        limits[R_GPUMEM] = str(int(util["gpumemPercentage"]))
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": slot, "namespace": namespace,
                     "labels": {L_SWAP: swap, L_SLOT: slot, L_KIND: kind},
                     "annotations": {f"{GROUP}/model": model_ref(s)}},
        "spec": graph,
    }


# ---------------------------------------------------------------- pods and GPUs (as modelserve-operator)

def pod_state(pod: dict | None) -> str:
    """missing | pending | ready | failed | terminating."""
    if not pod:
        return "missing"
    if (pod.get("metadata") or {}).get("deletionTimestamp"):
        return "terminating"
    st = pod.get("status") or {}
    for cs in [*(st.get("initContainerStatuses") or []), *(st.get("containerStatuses") or [])]:
        waiting = ((cs.get("state") or {}).get("waiting") or {}).get("reason")
        if waiting in ("CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff", "CreateContainerConfigError"):
            return "failed"
        if (cs.get("restartCount") or 0) >= 2:
            return "failed"
    if st.get("phase") == "Failed":
        return "failed"
    conds = {c.get("type"): c.get("status") for c in st.get("conditions") or []}
    return "ready" if conds.get("Ready") == "True" else "pending"


def slot_state(dgd: dict | None, pods: list[dict]) -> str:
    """missing | pending | ready | failed | terminating for a whole slot (its DGD and its prefill + decode pods)."""
    if dgd is None or (dgd.get("metadata") or {}).get("deletionTimestamp"):
        return "terminating" if pods or dgd is not None else "missing"
    by_role: dict[str, list[str]] = {}
    for p in pods:
        role = ((p.get("metadata") or {}).get("labels") or {}).get("nvidia.com/dynamo-component-type")
        by_role.setdefault(role, []).append(pod_state(p))
    states = [st for sts in by_role.values() for st in sts]
    if "failed" in states:
        return "failed"
    live = {r: [st for st in sts if st != "terminating"] for r, sts in by_role.items()}
    if all(live.get(r) == ["ready"] for r in ROLES):
        return "ready"
    return "pending"


def _pod_gpus(pod: dict) -> set[str]:
    raw = ((pod.get("metadata") or {}).get("annotations") or {}).get(A_GPUS) or ""
    return {g.strip() for g in raw.split(",") if g.strip()}


def _pod_gpumem(pod: dict) -> int:
    total = 0
    for c in (pod.get("spec") or {}).get("containers") or []:
        v = ((c.get("resources") or {}).get("limits") or {}).get(R_GPUMEM)
        if v is not None:
            total = max(total, int(str(v)))
    return total


def gpu_preflight(pods: list[dict], gpus: list[str], pct: int) -> tuple[bool, str]:
    """Can a new engine start on `gpus` now? HAMi refuses a card past 100 % (say why instead of a Pending pod), and two
    engines must not profile free memory on a card at once, so a pod on these cards that is not Ready yet, or still
    terminating and holding memory, makes us wait."""
    want = set(gpus)
    booked = {g: 0 for g in want}
    for p in pods:
        overlap = _pod_gpus(p) & want
        if not overlap or (p.get("status") or {}).get("phase") in ("Succeeded", "Failed"):
            continue
        state = pod_state(p)
        if state in ("pending", "terminating"):
            return False, f"waiting: pod {p['metadata']['name']} on the same cards is {state}"
        for g in overlap:
            booked[g] += _pod_gpumem(p)
    over = {g: b for g, b in booked.items() if b + pct > 100}
    if over:
        g, b = next(iter(over.items()))
        return False, f"no HAMi room: card {g[:12]} has {b}% booked, +{pct}% > 100"
    return True, ""


# ---------------------------------------------------------------- the swap state machine

def _slot(s: dict, swap: str, kind: str, gen: int, now: float, snapshot: str, attempts: int = 1) -> dict:
    return {"slot": slot_name(s, swap, kind, gen), "kind": kind, "model": model_ref(s), "hash": config_hash(s),
            "shape": shape_hash(s), "snapshot": snapshot, "since": now, "attempts": attempts}


def _wait(slot: dict, slots: dict, timeout: float, now: float, acts: list) -> str:
    """acked | timeout | failed | wait. Asks main to probe a slot the frontend lists but has not routed to yet."""
    o = slots.get(slot["slot"]) or {}
    if o.get("state") == "ready" and o.get("acked"):
        return "acked"
    if o.get("state") == "failed":
        return "failed"
    if now - slot["since"] > timeout:
        return "timeout"
    if o.get("state") == "ready" and o.get("listed"):
        acts.append(("probe", slot["slot"], slot["model"]))
    return "wait"


def decide(swap: str, spec: dict, status: dict, slots: dict, preflight, now: float | None = None,
           jobs: dict | None = None):
    """One step. slots: {slot: {state, listed, acked, retired, prefill}} for every slot seen in the cluster (retired: its
    prefill is scaled to 0; prefill: "missing" once no prefill pod is left). preflight(pct) -> (ok, why). jobs: {key:
    {state: running|succeeded|failed, result, error}} for the node Jobs main ran, all nodes merged.
    Returns (actions, new_status); actions are ("create", kind, slot) / ("retire", slot) / ("delete", slot) /
    ("probe", slot, model) / ("job", key, mode, params) / ("reload", slot, params). main writes a reload's outcome to
    status["reload"] before the next step."""
    now = time.time() if now is None else now
    jobs = jobs or {}
    s = with_defaults(spec)
    # A deep copy: main diffs the result against the CR's status to patch it, so nothing here may touch the input.
    st = copy.deepcopy({k: v for k, v in (status or {}).items() if k not in ("kopf",)})
    acts: list = []
    err = validate(s, swap)
    if err:                                  # change nothing in the cluster; keep the phase so a fix resumes it
        st.update(phase=st.get("phase") or "Invalid", invalid=err, message=f"invalid spec: {err}")
        return acts, _finish(st, slots)
    st.pop("invalid", None)
    want = config_hash(s)
    timeout = float(s["readyTimeoutSeconds"])
    phase = st.get("phase") or "New"
    active, bridge, nxt, old = st.get("active"), st.get("bridge"), st.get("next"), st.get("old")
    gen = int(st.get("gen") or 0)
    nodes = graph_nodes(s)

    def state(name):
        return (slots.get(name) or {}).get("state", "missing")

    def create(kind, slot_obj, pct):
        ok, why = preflight(pct)
        if not ok:
            st["message"] = why
            return False
        acts.append(("create", kind, slot_obj["slot"]))
        return True

    def remove(name):
        """Take a slot away without cutting requests: its prefill first (scaled to 0), the whole slot only once that
        prefill pod is gone. A slot without a prefill is an incomplete worker set, so the frontend stops picking it at
        once, while requests already prefilled there finish on its decode (which then drains 60 s on delete).
        Deleting both at once cut requests caught between prefill and decode (2026-11-21, 3 x 503 \"no backend
        worker available\"). Returns (gone, what it waits for)."""
        o = slots.get(name) or {}
        now_state = o.get("state", "missing")
        if now_state == "missing":
            return True, "gone"
        if now_state == "terminating":
            return False, "its pods are exiting"
        if o.get("retired"):
            if o.get("prefill", "missing") == "missing":
                acts.append(("delete", name))
                return False, "prefill gone; deleting the slot"
            return False, "prefill scaled to 0; waiting for it to exit"
        if now_state == "failed" or not o.get("listed"):
            acts.append(("delete", name))
            return False, "never took traffic; deleting"
        acts.append(("retire", name))
        return False, "scaling its prefill to 0 first"

    def keep_only(*names):
        for name in slots:
            if name not in names:
                remove(name)

    def next_seq():
        st["seq"] = int(st.get("seq") or 0) + 1
        return st["seq"]

    def gate():
        """The download gate for the spec as it is now. (candidate, None) once the model is on every node, (None, None)
        while the Jobs run, (None, error) when they failed."""
        c = st.get("candidate")
        if c and c.get("hash") == want and c.get("nodes") == nodes:
            return c, None
        f = st.get("fetching") or {}
        if f.get("hash") != want or f.get("nodes") != nodes:
            f = {"hash": want, "model": model_ref(s), "nodes": nodes, "key": f"fetch-{next_seq()}", "since": now}
            st["fetching"] = f
        j = jobs.get(f["key"]) or {}
        if j.get("state") == "succeeded":
            r = j.get("result") or {}
            c = {"hash": want, "model": model_ref(s), "nodes": nodes, "snapshot": r.get("snapshot"),
                 "compatible": bool(r.get("compatible")), "why": r.get("why", "")}
            st.update(candidate=c, fetching=None)
            return c, None
        if j.get("state") == "failed" or now - f["since"] > float(s["fetch"]["timeoutSeconds"]):
            st["fetching"] = None
            return None, j.get("error") or f"no result in {s['fetch']['timeoutSeconds']} s"
        if not j:
            acts.append(("job", f["key"], "fetch", {"model": s["model"], "revision": s["revision"],
                                                     "compare": (active or {}).get("snapshot") or ""}))
        st["message"] = f"download gate: fetching and checking {model_ref(s)} on {', '.join(nodes)}"
        return None, None

    def linked(kind, target):
        """Is the link of kind's port set pointing at target on every node? Asks main to point it there if not."""
        pset = SET_OF[kind]
        links = st.setdefault("links", {})
        if links.get(pset) == {"target": target, "nodes": nodes}:
            return True
        p = st.get("linking") or {}
        if (p.get("set"), p.get("target"), p.get("nodes")) != (pset, target, nodes):
            p = {"set": pset, "target": target, "nodes": nodes, "key": f"link-{pset}-{next_seq()}"}
            st["linking"] = p
        j = jobs.get(p["key"]) or {}
        if j.get("state") == "succeeded":
            links[pset] = {"target": target, "nodes": nodes}
            st["linking"] = None
            return True
        if j.get("state") == "failed":
            st.update(linking=None, message=f"pointing link {pset} at {target} failed ({j.get('error')}); retrying")
            return False
        if not j:
            acts.append(("job", p["key"], "link", {"kind": kind, "target": target}))
        st["message"] = f"pointing link {pset} at {target.rsplit('/snapshots/', 1)[0].rsplit('/', 1)[-1]}"
        return False

    def start_bridge(cand):
        if not linked("bridge", cand["snapshot"]):
            return
        slot = _slot(s, swap, "bridge", gen + 1, now, cand["snapshot"])
        if create("bridge", slot, s["bridge"]["gpumemPercentage"]):
            st.update(gen=gen + 1, bridge=slot, phase="Bridging", failedHash=None,
                      message=f"bridge {slot['slot']} ({model_ref(s)}) loading beside {active['slot']}")

    if phase in ("New", "Failed", "Invalid") and not active and not nxt:
        keep_only()
        if want == st.get("failedHash"):
            return acts, _finish(st, slots)
        st.update(phase="Fetching", resume="New")
        phase = "Fetching"

    if phase in ("Serving", "Fetching", "Reloading") and active and state(active["slot"]) == "missing":
        st.update(phase="New", active=None, message=f"active slot {active['slot']} vanished; loading a new one")
        return acts, _finish(st, slots)

    if phase == "Serving":
        keep_only(active["slot"])
        if want not in (active["hash"], st.get("failedHash")):
            st.update(phase="Fetching", resume="Serving")
            phase = "Fetching"
        else:
            if want == active["hash"]:
                st["message"] = f"serving {active['model']} from {active['slot']}"
            return acts, _finish(st, slots)

    if phase == "Degraded" and nxt is None:     # serving on the bridge only: retry once the spec changes
        keep_only(bridge["slot"])
        if want != st.get("failedHash") and state(slot_name(s, swap, "full", gen)) == "missing":
            st.update(phase="Fetching", resume="Degraded")
            phase = "Fetching"
        else:
            return acts, _finish(st, slots)

    if phase == "Fetching":
        frm = st.get("resume") or "New"
        if frm == "Serving" and want == active["hash"]:      # edited back before the gate finished
            st.update(phase="Serving", fetching=None, message=f"serving {active['model']} from {active['slot']}")
            return acts, _finish(st, slots)
        keep_only(*[x["slot"] for x in (active, bridge) if x])
        cand, err = gate()
        if err:
            st.update(phase={"New": "Failed"}.get(frm, frm), failedHash=want,
                      lastFailure=f"download gate for {model_ref(s)}: {err}",
                      message=f"download gate failed for {model_ref(s)}: {err}"[:300])
            return acts, _finish(st, slots)
        if cand is None:
            return acts, _finish(st, slots)
        if frm == "Serving":
            ok, why = inplace_ok(s, active, cand)
            if ok:
                st.update(phase="Reloading", reload=None,
                          message=f"{model_ref(s)} differs from {active['model']} only in its weights; reloading in place")
            else:
                st["bridgeReason"] = why
                start_bridge(cand)
            return acts, _finish(st, slots)
        g = gen + 1 if frm == "New" else gen            # Degraded: the full of the bridge's generation
        if linked("full", cand["snapshot"]):
            slot = _slot(s, swap, "full", g, now, cand["snapshot"])
            if create("full", slot, s["steady"]["gpumemPercentage"]):
                beside = f" beside {bridge['slot']}" if bridge else ""
                st.update(gen=g, next=slot, phase="Loading", message=f"loading {slot['slot']} ({model_ref(s)}){beside}")
        return acts, _finish(st, slots)

    if phase == "Reloading":
        cand = st.get("candidate") or {}
        if cand.get("hash") != want:                 # edited again before the reload ran
            st.update(phase="Serving" if want == active["hash"] else "Fetching", resume="Serving")
            return acts, _finish(st, slots)
        r = st.get("reload") or {}
        if r.get("hash") != want:                    # not run yet: point the slot's link first, then reload
            if linked("full", cand["snapshot"]):
                acts.append(("reload", active["slot"], {"hash": want, "to": cand["snapshot"], "from": active["snapshot"],
                                                         "version": cand["model"].rsplit("/", 1)[-1]}))
                st["message"] = f"reloading {active['slot']} in place: {active['model']} -> {cand['model']}"
            return acts, _finish(st, slots)
        if r.get("ok"):
            st.update(phase="Serving", failedHash=None,
                      active={**active, "model": cand["model"], "hash": want, "snapshot": cand["snapshot"]},
                      message=f"serving {cand['model']} from {active['slot']}, reloaded in place "
                              f"(no new tokens for {r.get('pausedSeconds')} s)")
            return acts, _finish(st, slots)
        # Failed: main already put the old weights back. Point the link back too, then bridge as for any other change.
        st["lastFailure"] = f"in-place reload to {cand['model']} failed: {r.get('why')}; old weights restored"
        if linked("full", active["snapshot"]):
            st["bridgeReason"] = "in-place reload failed"
            start_bridge(cand)
        return acts, _finish(st, slots)

    if phase == "Bridging":
        if want == active["hash"]:          # edited back before the swap moved on: drop the bridge
            remove(bridge["slot"])
            st.update(phase="Finalizing", message=f"spec is back to {active['model']}; removing {bridge['slot']}")
            return acts, _finish(st, slots)
        r = _wait(bridge, slots, timeout, now, acts)
        if r == "acked":
            remove(active["slot"])
            st.update(phase="DrainingOld", old=active, active=None,
                      message=f"frontend routes to {bridge['slot']}; retiring {active['slot']}, prefill first")
        elif r in ("timeout", "failed"):
            remove(bridge["slot"])
            st.update(phase="Finalizing", failedHash=bridge["hash"],
                      lastFailure=f"bridge {bridge['slot']} ({bridge['model']}) {r}; kept {active['slot']}",
                      message=f"bridge {r}; still serving {active['model']}")
        else:
            st["message"] = f"waiting for the frontend to route to {bridge['slot']} ({state(bridge['slot'])})"
        return acts, _finish(st, slots)

    if phase == "DrainingOld":
        gone, why = remove(old["slot"])
        if not gone:
            st["message"] = f"draining {old['slot']}: {why}"
            return acts, _finish(st, slots)
        # Rendered from the spec as it is now: an edit made during the swap wins here (after its own download gate).
        cand, err = gate()
        if err:
            st.update(phase="Degraded", old=None, failedHash=want, lastFailure=f"download gate for {model_ref(s)}: {err}",
                      message=f"download gate failed; serving on bridge {bridge['slot']} only")
            return acts, _finish(st, slots)
        if cand is None:
            return acts, _finish(st, slots)
        if linked("full", cand["snapshot"]):        # the old full is gone, so set a's link is free to move
            slot = _slot(s, swap, "full", gen, now, cand["snapshot"])
            if create("full", slot, s["steady"]["gpumemPercentage"]):
                st.update(phase="Loading", next=slot, old=None,
                          message=f"{old['slot']} gone; loading {slot['slot']} beside {bridge['slot']}")
        return acts, _finish(st, slots)

    if phase == "Loading":
        if nxt.get("retiring"):             # a failed attempt, waiting for its pods to go before the retry
            if state(nxt["slot"]) != "missing":
                st["message"] = f"waiting for failed {nxt['slot']} to exit"
                return acts, _finish(st, slots)
            cand = _slot(s, swap, "full", gen, now, nxt["snapshot"], nxt["attempts"] + 1)
            cand.update(model=nxt["model"], hash=nxt["hash"], shape=nxt["shape"])
            if create("full", cand, s["steady"]["gpumemPercentage"]):
                st.update(next=cand, message=f"retrying {cand['slot']} (attempt {cand['attempts']})")
            return acts, _finish(st, slots)
        r = _wait(nxt, slots, timeout, now, acts)
        if r == "acked":
            done = {k: v for k, v in nxt.items() if k not in ("since", "attempts")}
            st.update(active=done, next=None, failedHash=None)
            if bridge:
                remove(bridge["slot"])
                st.update(phase="Finalizing", message=f"{done['slot']} serves; removing bridge {bridge['slot']}")
            else:
                st.update(phase="Serving", message=f"serving {done['model']} from {done['slot']}")
        elif r in ("timeout", "failed"):
            acts.append(("delete", nxt["slot"]))
            if nxt["attempts"] <= int(s["fullRetries"]):
                st.update(next={**nxt, "retiring": True}, message=f"{nxt['slot']} {r}; retrying")
            elif bridge:
                st.update(phase="Degraded", next=None, failedHash=nxt["hash"],
                          lastFailure=f"full {nxt['slot']} {r} {nxt['attempts']}x",
                          message=f"full engine {r}; serving on bridge {bridge['slot']} only. Fix spec to retry")
            else:
                st.update(phase="Failed", next=None, failedHash=nxt["hash"],
                          lastFailure=f"{nxt['slot']} {r} {nxt['attempts']}x", message=f"{nxt['slot']} {r}")
        else:
            st["message"] = f"waiting for the frontend to route to {nxt['slot']} ({state(nxt['slot'])})"
        return acts, _finish(st, slots)

    if phase == "Finalizing":
        if bridge:
            gone, why = remove(bridge["slot"])
            if not gone:
                st["message"] = f"removing bridge {bridge['slot']}: {why}"
                return acts, _finish(st, slots)
        st.update(phase="Serving", bridge=None, message=f"serving {active['model']} from {active['slot']}")
        return acts, _finish(st, slots)

    st.update(phase="Failed", message=f"unknown phase {phase!r}")
    return acts, _finish(st, slots)


def _finish(st: dict, slots: dict) -> dict:
    """Display fields: which slots should be taking traffic now, and their models."""
    serving = []
    for key in ("active", "old", "bridge", "next"):
        x = st.get(key)
        if x and (slots.get(x["slot"]) or {}).get("listed") and x["slot"] not in [y["slot"] for y in serving]:
            serving.append(x)
    st["serving"] = [x["slot"] for x in serving]
    st["servingModel"] = " + ".join(dict.fromkeys(x["model"].rsplit("/", 1)[-1] for x in serving)) or "-"
    return st
