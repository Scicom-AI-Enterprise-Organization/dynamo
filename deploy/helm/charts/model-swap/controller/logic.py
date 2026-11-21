"""DynamoModelSwap: the pure half (no Kubernetes, no HTTP), so the swap order can be tested on its own.

One DynamoModelSwap holds a Dynamo graph's worker spec (a prefill and a decode component, as written in a
DynamoGraphDeployment) plus the model. Edit spec.model and the controller swaps with no downtime, one replica:

    full(old, 0.70)                       serving
    + bridge(new, 0.27)                   Bridging     both serve once the frontend routes to the bridge
    - full(old)                           DrainingOld  prefill scaled to 0 first, then the slot; wait until its pods are gone
    + full(new, 0.70)                     Loading      bridge + new full serve
    - bridge(new)                         Finalizing
    full(new, 0.70)                       Serving

Each step is its own workers-only DynamoGraphDeployment ("slot") named <frontendGraph>-<swap>-<f|b><gen>, so the
frontend graph's namespace-prefix discovery picks it up. Fulls always use host port set a, bridges set b.
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
# The DGD's own pod-template-hash label value is <dgd>-vllmprefillworker-<8 hex> and must stay <= 63 characters.
MAX_SLOT_NAME = 36

DEFAULTS = {
    "frontendGraph": "lab",
    "steady": {"gpuMemoryUtilization": 0.70, "gpumemPercentage": 70},
    "bridge": {"gpuMemoryUtilization": 0.27, "gpumemPercentage": 27},
    "readyTimeoutSeconds": 1800,
    "fullRetries": 1,
    "versionAlias": True,
}


def with_defaults(spec: dict) -> dict:
    s = copy.deepcopy(spec or {})
    for k, v in DEFAULTS.items():
        if isinstance(v, dict):
            s[k] = {**v, **(s.get(k) or {})}
        else:
            s.setdefault(k, v)
    return s


def config_hash(s: dict) -> str:
    """What a full slot is: a change in any of these starts a swap. Bridge settings are not in it."""
    key = {k: s.get(k) for k in ("model", "servedModelName", "versionAlias", "steady", "graph")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


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
    total = int(s["steady"]["gpumemPercentage"]) + int(s["bridge"]["gpumemPercentage"])
    if total > 100:
        return f"steady + bridge gpumem is {total}% > 100: the bridge and a full engine must fit on the same cards"
    if len(slot_name(s, swap, "bridge", 9999)) > MAX_SLOT_NAME:
        return f"names too long: {slot_name(s, swap, 'bridge', 9999)} exceeds {MAX_SLOT_NAME} characters"
    return ""


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
    names = [s["servedModelName"]] + ([s["model"]] if s.get("versionAlias") and s["model"] != s["servedModelName"]
                                      else [])
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
        c["args"] = ["--model", s["model"], "--served-model-name", *names,
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
                     "annotations": {f"{GROUP}/model": s["model"]}},
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

def _slot(s: dict, swap: str, kind: str, gen: int, now: float, attempts: int = 1) -> dict:
    return {"slot": slot_name(s, swap, kind, gen), "kind": kind, "model": s["model"], "hash": config_hash(s),
            "since": now, "attempts": attempts}


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


def decide(swap: str, spec: dict, status: dict, slots: dict, preflight, now: float | None = None):
    """One step. slots: {slot: {state, listed, acked, retired, prefill}} for every slot seen in the cluster (retired: its
    prefill is scaled to 0; prefill: "missing" once no prefill pod is left). preflight(pct) -> (ok, why).
    Returns (actions, new_status); actions are ("create", kind, slot) / ("retire", slot) / ("delete", slot) /
    ("probe", slot, model)."""
    now = time.time() if now is None else now
    s = with_defaults(spec)
    st = {k: v for k, v in (status or {}).items() if k not in ("kopf",)}
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

    if phase in ("New", "Failed", "Invalid") and not active and not nxt:
        keep_only()
        if want != st.get("failedHash"):
            cand = _slot(s, swap, "full", gen + 1, now)
            if create("full", cand, s["steady"]["gpumemPercentage"]):
                st.update(gen=gen + 1, next=cand, phase="Loading", message=f"loading {cand['slot']} ({s['model']})")
        return acts, _finish(st, slots)

    if phase == "Serving" and state(active["slot"]) == "missing":
        st.update(phase="New", active=None, message=f"active slot {active['slot']} vanished; loading a new one")
        return acts, _finish(st, slots)

    if phase == "Serving":
        keep_only(active["slot"])
        if want not in (active["hash"], st.get("failedHash")):
            cand = _slot(s, swap, "bridge", gen + 1, now)
            if create("bridge", cand, s["bridge"]["gpumemPercentage"]):
                st.update(gen=gen + 1, bridge=cand, phase="Bridging", failedHash=None,
                          message=f"bridge {cand['slot']} ({s['model']}) loading beside {active['slot']}")
        elif want == active["hash"]:
            st["message"] = f"serving {active['model']} from {active['slot']}"
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
        # Rendered from the spec as it is now: an edit made during the swap wins here.
        cand = _slot(s, swap, "full", gen, now)
        if create("full", cand, s["steady"]["gpumemPercentage"]):
            st.update(phase="Loading", next=cand, old=None,
                      message=f"{old['slot']} gone; loading {cand['slot']} beside {bridge['slot']}")
        return acts, _finish(st, slots)

    if phase in ("Loading", "Degraded"):
        if nxt is None:                     # Degraded: retry only once the spec changes
            if want != st.get("failedHash") and state(slot_name(s, swap, "full", gen)) == "missing":
                cand = _slot(s, swap, "full", gen, now)
                if create("full", cand, s["steady"]["gpumemPercentage"]):
                    st.update(phase="Loading", next=cand, message=f"loading {cand['slot']} beside {bridge['slot']}")
            return acts, _finish(st, slots)
        if nxt.get("retiring"):             # a failed attempt, waiting for its pods to go before the retry
            if state(nxt["slot"]) != "missing":
                st["message"] = f"waiting for failed {nxt['slot']} to exit"
                return acts, _finish(st, slots)
            cand = _slot(s, swap, "full", gen, now, nxt["attempts"] + 1)
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
