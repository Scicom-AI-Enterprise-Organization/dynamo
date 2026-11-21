"""The swap order, against a simulated cluster: slots take ticks to load, to be routed to, and to exit; node Jobs take
ticks to finish. Run: uv run --with pytest pytest controller/test_logic.py"""
import json

import logic

V1, V2, V3 = "org/model-v1", "org/model-v2", "org/model-v3"
GPUS = "GPU-a,GPU-b"


def spec(model, alias=True, nodes=("node-2", "node-2"), names=("prefill", "decode"), **kw):
    def worker(t, node, name):
        return {"name": name, "type": t, "podTemplate": {
            "metadata": {"annotations": {logic.A_GPUS: GPUS}},
            "spec": {"nodeSelector": {"kubernetes.io/hostname": node},
                     "containers": [{"name": "main", "args": ["--model", "x", "--gpu-memory-utilization", "0.9",
                                                              "--served-model-name", "a", "b", "--block-size", "64"]}]}}}
    return {"model": model, "servedModelName": "served", "versionAlias": alias,
            "graph": {"components": [worker("prefill", nodes[0], names[0]), worker("decode", nodes[1], names[1])]},
            **kw}


def snap(ref):
    model, _, rev = ref.partition("@")
    return f"/data/hf/hub/models--{model.replace('/', '--')}/snapshots/{rev or 'r1'}"


class World:
    """load: ticks from create to Ready; ack: further ticks until the frontend routes to it; exit: ticks to vanish;
    job: ticks a node Job takes. compatible(old, new) says what the gate finds."""

    def __init__(self, load=3, ack=2, exit=2, job=1, fail=(), fetch_fail=(), reload_fail=(),
                 compatible=lambda old, new: True):
        self.load, self.ack, self.exit, self.job, self.fail = load, ack, exit, job, set(fail)
        self.fetch_fail, self.reload_fail, self.compatible = set(fetch_fail), set(reload_fail), compatible
        self.slots, self.jobs, self.links, self.t, self.log, self.status = {}, {}, {}, 0, [], {}

    def view(self):
        out = {}
        for n, x in self.slots.items():
            ret = x["ret"] is not None
            prefill = "missing" if ret and self.t - x["ret"] >= self.exit else "present"
            if x["del"] is not None:
                out[n] = {"state": "terminating", "listed": False, "acked": False, "retired": ret, "prefill": prefill}
            elif n in self.fail and self.t - x["born"] >= self.load:
                out[n] = {"state": "failed", "listed": False, "acked": False, "retired": ret, "prefill": prefill}
            elif ret:   # prefill unregistered at once: the frontend stops routing to the slot
                out[n] = {"state": "pending", "listed": False, "acked": False, "retired": True, "prefill": prefill}
            else:
                age = self.t - x["born"]
                out[n] = {"state": "ready" if age >= self.load else "pending", "listed": age >= self.load,
                          "acked": age >= self.load + self.ack, "retired": False, "prefill": "present"}
        return out

    def job_view(self):
        out = {}
        for k, j in self.jobs.items():
            if self.t - j["born"] < self.job:
                out[k] = {"state": "running"}
            elif j["mode"] == "fetch" and j["p"]["model"] in self.fetch_fail:
                out[k] = {"state": "failed", "error": "shard missing"}
            elif j["mode"] == "fetch":
                old, ref = j["p"]["compare"], j["p"]["model"] + (f"@{j['p']['revision']}" if j["p"]["revision"] else "")
                out[k] = {"state": "succeeded", "result": {
                    "snapshot": snap(ref), "compatible": bool(old) and self.compatible(old, ref), "why": "test"}}
            else:
                out[k] = {"state": "succeeded", "result": {}}
        return out

    def preflight(self, pct):
        if any(v["state"] in ("pending", "terminating") for v in self.view().values()):
            return False, "busy"
        used = sum(x["pct"] for x in self.slots.values())
        return (used + pct <= 100), f"{used}+{pct}"

    def step(self, sp, now=None):
        if isinstance(sp, str):
            sp = spec(sp)
        before = json.dumps(self.status, sort_keys=True)
        acts, new = logic.decide("sw", sp, self.status, self.view(), self.preflight,
                                 now if now is not None else self.t, self.job_view())
        assert json.dumps(self.status, sort_keys=True) == before, "decide() changed the status it was given"
        self.status = json.loads(json.dumps(new))      # as the CR stores it
        s = logic.with_defaults(sp)
        for a in acts:
            if a[0] == "create":
                pct = s["steady" if a[1] == "full" else "bridge"]["gpumemPercentage"]
                link = self.links.get(logic.SET_OF[a[1]])
                assert link == snap(logic.model_ref(s)) or a[1] == "full" and self.status.get("next", {}).get("retiring"), \
                    f"{a[2]} created while its link points at {link}"
                self.slots[a[2]] = {"born": self.t, "del": None, "ret": None, "pct": pct, "kind": a[1],
                                    "model": logic.model_ref(s)}
                self.log.append(("+", a[2], logic.model_ref(s)))
            elif a[0] == "retire" and a[1] in self.slots and self.slots[a[1]]["ret"] is None:
                self.slots[a[1]]["ret"] = self.t
                self.log.append(("r", a[1]))
            elif a[0] == "delete" and a[1] in self.slots and self.slots[a[1]]["del"] is None:
                self.slots[a[1]]["del"] = self.t
                self.log.append(("-", a[1]))
            elif a[0] == "job" and a[1] not in self.jobs:
                self.jobs[a[1]] = {"born": self.t, "mode": a[2], "p": a[3]}
                if a[2] == "link":
                    self.links[logic.SET_OF[a[3]["kind"]]] = a[3]["target"]
                    self.log.append(("link", logic.SET_OF[a[3]["kind"]], a[3]["target"]))
            elif a[0] == "reload":
                p = a[2]
                assert self.links["a"] == p["to"], "reloaded before the slot's link moved"
                ok = p["to"] not in {snap(m) for m in self.reload_fail}
                self.status["reload"] = {"hash": p["hash"], "ok": ok, "why": "" if ok else "boom", "pausedSeconds": 3}
                self.log.append(("reload", a[1], p["to"], ok))
                if ok:
                    self.slots[a[1]]["model"] = p["to"]
        self.t += 1
        for n in [n for n, x in self.slots.items() if x["del"] is not None and self.t - x["del"] >= self.exit]:
            del self.slots[n]
        assert sum(x["pct"] for x in self.slots.values()) <= 100, self.slots

    def run(self, sp, until, limit=200):
        for _ in range(limit):
            self.step(sp)
            if until(self):
                return
        raise AssertionError(f"stuck in {self.status.get('phase')}: {self.status.get('message')}")

    def acked(self):
        return [n for n, v in self.view().items() if v["acked"]]

    def slot_events(self):
        return [e for e in self.log if e[0] in ("+", "r", "-")]


def serving(model):
    return lambda w: w.status.get("phase") == "Serving" and w.status["active"]["model"] == model


def test_swap_order_and_always_one_slot_routed():
    w = World()
    w.run(V1, serving(V1))
    assert w.slot_events() == [("+", "lab-sw-f1", V1)]
    w.log.clear()
    while True:
        w.step(V2)
        assert w.acked(), f"nothing routed at tick {w.t}: {w.status['phase']}"
        if serving(V2)(w):
            break
    assert w.slot_events() == [("+", "lab-sw-b2", V2), ("r", "lab-sw-f1"), ("-", "lab-sw-f1"),
                               ("+", "lab-sw-f2", V2), ("r", "lab-sw-b2"), ("-", "lab-sw-b2")]
    assert list(w.slots) == ["lab-sw-f2"]
    assert w.status["bridgeReason"].startswith("spec.versionAlias is on")


def test_old_removed_only_after_bridge_routed_and_full_only_after_old_gone():
    w = World(load=4, ack=3, exit=5)
    w.run(V1, serving(V1))
    events = []
    while not serving(V2)(w):
        before = dict(w.view())
        n = len(w.log)
        w.step(V2)
        for e in w.log[n:]:
            events.append((e, before))
    for (e, before) in events:
        if e == ("r", "lab-sw-f1"):
            assert before["lab-sw-b2"]["acked"]
        if e[:2] == ("+", "lab-sw-f2"):
            assert "lab-sw-f1" not in before
        if e == ("link", "a", snap(V2)):        # set a's link moves only once the old full is gone
            assert "lab-sw-f1" not in before
        if e == ("r", "lab-sw-b2"):
            assert before["lab-sw-f2"]["acked"]
        if e[0] == "-" and e[1] in ("lab-sw-f1", "lab-sw-b2"):   # took traffic: deleted only once its prefill is gone
            assert before[e[1]]["retired"] and before[e[1]]["prefill"] == "missing", e


def test_inplace_when_only_the_weights_change():
    w = World()
    w.run(spec(V1, alias=False), serving(V1))
    w.log.clear()
    while not serving(V2)(w):
        w.step(spec(V2, alias=False))
        assert w.acked() == ["lab-sw-f1"]
    assert w.log == [("link", "a", snap(V2)), ("reload", "lab-sw-f1", snap(V2), True)]
    assert w.status["active"]["snapshot"] == snap(V2) and w.status["active"]["slot"] == "lab-sw-f1"
    w.log.clear()
    w.run(spec(V3, alias=False), serving(V3))           # and again from there
    assert [e[0] for e in w.log] == ["link", "reload"] and list(w.slots) == ["lab-sw-f1"]


def test_new_revision_of_the_same_model_reloads_in_place():
    w = World()
    w.run(spec(V1, alias=False), serving(V1))
    w.log.clear()
    pinned = f"{V1}@abc123"
    w.run(spec(V1, alias=False, revision="abc123"), serving(pinned))
    assert w.log == [("link", "a", snap(pinned)), ("reload", "lab-sw-f1", snap(pinned), True)]
    assert [j["p"]["revision"] for j in w.jobs.values() if j["mode"] == "fetch"] == ["", "abc123"]
    w.log.clear()
    w.run(spec(V1, alias=False), serving(V1))          # unpinned again: back to main's snapshot, in place
    assert [e[0] for e in w.log] == ["link", "reload"]


def test_failed_reload_restores_the_link_then_bridges():
    w = World(reload_fail={V2})
    w.run(spec(V1, alias=False), serving(V1))
    w.log.clear()
    w.run(spec(V2, alias=False), serving(V2))
    assert w.log[:4] == [("link", "a", snap(V2)), ("reload", "lab-sw-f1", snap(V2), False),
                         ("link", "a", snap(V1)), ("link", "b", snap(V2))]
    assert w.slot_events()[0] == ("+", "lab-sw-b2", V2) and list(w.slots) == ["lab-sw-f2"]
    assert "in-place reload" in w.status["lastFailure"]


def test_incompatible_model_takes_the_bridge():
    w = World(compatible=lambda old, new: False)
    w.run(spec(V1, alias=False), serving(V1))
    w.log.clear()
    w.run(spec(V2, alias=False), serving(V2))
    assert not [e for e in w.log if e[0] == "reload"] and list(w.slots) == ["lab-sw-f2"]
    assert w.status["bridgeReason"].startswith("not weight-compatible")


def test_more_than_the_model_changed_takes_the_bridge():
    w = World()
    w.run(spec(V1, alias=False), serving(V1))
    w.run(spec(V2, alias=False, steady={"gpuMemoryUtilization": 0.6, "gpumemPercentage": 60}), serving(V2))
    assert w.status["bridgeReason"] == "more than spec.model changed"


def test_download_gate_failure_keeps_serving_and_creates_nothing():
    w = World(fetch_fail={V2})
    w.run(V1, serving(V1))
    w.log.clear()
    w.run(V2, lambda w: w.status.get("failedHash"))
    assert w.status["phase"] == "Serving" and w.status["active"]["model"] == V1
    assert "download gate" in w.status["lastFailure"]
    for _ in range(10):                      # no retry loop on the same spec
        w.step(V2)
    assert not w.slot_events() and list(w.slots) == ["lab-sw-f1"]
    assert len([j for j in w.jobs.values() if j["mode"] == "fetch" and j["p"]["model"] == V2]) == 1


def test_gate_and_link_come_before_the_first_slot():
    w = World(job=3)
    w.run(V1, serving(V1))
    kinds = [e[0] for e in w.log]
    assert kinds.index("link") < kinds.index("+")
    assert [j["mode"] for j in w.jobs.values()] == ["fetch", "link"]


def test_moving_to_more_nodes_points_the_links_on_the_new_nodes_too():
    w = World()
    w.run(spec(V1), serving(V1))
    two = spec(V1, nodes=("node-1", "node-2"))      # same model, new layout: a bridge swap, links redone on both nodes
    w.run(two, lambda w: serving(V1)(w) and w.status["active"]["slot"] == "lab-sw-f2")
    links = [e for e in w.log if e[0] == "link"]
    assert links[-2:] == [("link", "b", snap(V1)), ("link", "a", snap(V1))]
    assert w.status["links"]["a"] == {"target": snap(V1), "nodes": ["node-1", "node-2"]}


def test_failed_bridge_keeps_the_old_model_and_waits_for_a_new_edit():
    w = World(fail={"lab-sw-b2"})
    w.run(V1, serving(V1))
    w.run(V2, lambda w: w.status.get("phase") == "Serving" and w.status.get("failedHash"))
    assert w.status["active"]["model"] == V1 and list(w.slots) == ["lab-sw-f1"]
    for _ in range(10):                      # no retry loop on the same spec
        w.step(V2)
    assert list(w.slots) == ["lab-sw-f1"]
    w.run(V3, serving(V3))                   # a new edit swaps again
    assert list(w.slots) == ["lab-sw-f3"]  # generations count bridges: b2 failed, b3 bridged to f3


def test_failed_full_retries_then_stays_on_the_bridge():
    w = World(fail={"lab-sw-f2"})
    w.run(V1, serving(V1))
    w.run(V2, lambda w: w.status.get("phase") == "Degraded")
    assert [e for e in w.log if e[1] == "lab-sw-f2" and e[0] == "+"] == [("+", "lab-sw-f2", V2)] * 2
    for _ in range(5):                       # the failed f2 exits; the bridge keeps serving
        w.step(V2)
    assert list(w.slots) == ["lab-sw-b2"] and w.acked() == ["lab-sw-b2"] and w.status["phase"] == "Degraded"


def test_slot_timeout_counts_as_failure():
    w = World(load=10_000, job=0)
    w.status = {"phase": "Loading", "gen": 1, "links": {}, "next": {**logic._slot(
        logic.with_defaults(spec(V1)), "sw", "full", 1, 0, snap(V1))}}
    w.slots["lab-sw-f1"] = {"born": 0, "del": None, "ret": None, "pct": 70, "kind": "full", "model": V1}
    w.step(V1, now=1801)
    assert ("-", "lab-sw-f1") in w.log


def test_render_owns_model_name_and_memory_flags():
    s = logic.with_defaults(spec(V2))
    args = logic.render_slot("sw", "ns", s, "lab-sw-b2", "bridge")["spec"]["components"][1][
        "podTemplate"]["spec"]["containers"][0]["args"]
    assert args == ["--model", "/data/hf/swap/ns/sw-b", "--served-model-name", "served", V2,
                    "--gpu-memory-utilization", "0.27", "--block-size", "64"]


def test_steady_plus_bridge_over_100_is_refused():
    sp = {**spec(V1), "steady": {"gpuMemoryUtilization": 0.7, "gpumemPercentage": 75}}
    acts, st = logic.decide("sw", sp, {}, {}, lambda p: (True, ""), 0)
    assert acts == [] and "100" in st["message"]


def test_prod_names_fit_only_with_short_component_names():
    prod = "gemma4-31b-dpo-nvfp4fp8attn-p4d8"
    long = {**spec(V1, names=("VllmPrefillWorker", "VllmDecodeWorker")), "frontendGraph": prod}
    short = {**spec(V1), "frontendGraph": prod}
    assert "names too long" in logic.validate(logic.with_defaults(long), "gemma")
    assert logic.validate(logic.with_defaults(short), "gemma") == ""


def test_static_aliases_are_served_and_are_part_of_the_slot():
    sp = {**spec(V1, alias=False), "servedModelAliases": ["org/old-name"]}
    s = logic.with_defaults(sp)
    args = logic.render_slot("sw", "ns", s, "lab-sw-f1", "full")["spec"]["components"][0][
        "podTemplate"]["spec"]["containers"][0]["args"]
    assert args[2:6] == ["--served-model-name", "served", "org/old-name", "--gpu-memory-utilization"]
    assert logic.shape_hash(s) != logic.shape_hash(logic.with_defaults(spec(V1, alias=False)))
    assert logic.config_hash(logic.with_defaults({**spec(V1), "servedModelAliases": []})) == \
        logic.config_hash(logic.with_defaults(spec(V1)))
