"""The swap order, against a simulated cluster: slots take ticks to load, to be routed to, and to exit.
Run: uv run --with pytest pytest controller/test_logic.py"""
import logic

V1, V2, V3 = "org/model-v1", "org/model-v2", "org/model-v3"
GPUS = "GPU-a,GPU-b"


def spec(model):
    worker = lambda t: {"name": t, "type": t, "podTemplate": {
        "metadata": {"annotations": {logic.A_GPUS: GPUS}},
        "spec": {"containers": [{"name": "main", "args": ["--model", "x", "--gpu-memory-utilization", "0.9",
                                                          "--served-model-name", "a", "b", "--block-size", "64"]}]}}}
    return {"model": model, "servedModelName": "served", "graph": {"components": [worker("prefill"), worker("decode")]}}


class World:
    """load: ticks from create to Ready; ack: further ticks until the frontend routes to it; exit: ticks to vanish."""

    def __init__(self, load=3, ack=2, exit=2, fail=()):
        self.load, self.ack, self.exit, self.fail = load, ack, exit, set(fail)
        self.slots, self.t, self.log, self.status = {}, 0, [], {}

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

    def preflight(self, pct):
        if any(v["state"] in ("pending", "terminating") for v in self.view().values()):
            return False, "busy"
        used = sum(x["pct"] for x in self.slots.values())
        return (used + pct <= 100), f"{used}+{pct}"

    def step(self, model, now=None):
        acts, self.status = logic.decide("sw", spec(model), self.status, self.view(), self.preflight,
                                         now if now is not None else self.t)
        s = logic.with_defaults(spec(model))
        for a in acts:
            if a[0] == "create":
                pct = s["steady" if a[1] == "full" else "bridge"]["gpumemPercentage"]
                self.slots[a[2]] = {"born": self.t, "del": None, "ret": None, "pct": pct, "kind": a[1], "model": model}
                self.log.append(("+", a[2], model))
            elif a[0] == "retire" and a[1] in self.slots and self.slots[a[1]]["ret"] is None:
                self.slots[a[1]]["ret"] = self.t
                self.log.append(("r", a[1]))
            elif a[0] == "delete" and a[1] in self.slots and self.slots[a[1]]["del"] is None:
                self.slots[a[1]]["del"] = self.t
                self.log.append(("-", a[1]))
        self.t += 1
        for n in [n for n, x in self.slots.items() if x["del"] is not None and self.t - x["del"] >= self.exit]:
            del self.slots[n]
        assert sum(x["pct"] for x in self.slots.values()) <= 100, self.slots

    def run(self, model, until, limit=200):
        for _ in range(limit):
            self.step(model)
            if until(self):
                return
        raise AssertionError(f"stuck in {self.status.get('phase')}: {self.status.get('message')}")

    def acked(self):
        return [n for n, v in self.view().items() if v["acked"]]


def serving(model):
    return lambda w: w.status.get("phase") == "Serving" and w.status["active"]["model"] == model


def test_swap_order_and_always_one_slot_routed():
    w = World()
    w.run(V1, serving(V1))
    assert w.log == [("+", "lab-sw-f1", V1)]
    w.log.clear()
    while True:
        w.step(V2)
        assert w.acked(), f"nothing routed at tick {w.t}: {w.status['phase']}"
        if serving(V2)(w):
            break
    assert w.log == [("+", "lab-sw-b2", V2), ("r", "lab-sw-f1"), ("-", "lab-sw-f1"), ("+", "lab-sw-f2", V2),
                     ("r", "lab-sw-b2"), ("-", "lab-sw-b2")]
    assert list(w.slots) == ["lab-sw-f2"]


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
        if e == ("r", "lab-sw-b2"):
            assert before["lab-sw-f2"]["acked"]
        if e[0] == "-" and e[1] in ("lab-sw-f1", "lab-sw-b2"):   # took traffic: deleted only once its prefill is gone
            assert before[e[1]]["retired"] and before[e[1]]["prefill"] == "missing", e


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
    w = World(load=10_000)
    w.status = {}
    w.step(V1, now=0)
    w.step(V1, now=1801)
    assert ("-", "lab-sw-f1") in w.log


def test_render_owns_model_name_and_memory_flags():
    s = logic.with_defaults(spec(V2))
    args = logic.render_slot("sw", "ns", s, "lab-sw-b2", "bridge")["spec"]["components"][1][
        "podTemplate"]["spec"]["containers"][0]["args"]
    assert args == ["--model", V2, "--served-model-name", "served", V2, "--gpu-memory-utilization", "0.27",
                    "--block-size", "64"]


def test_steady_plus_bridge_over_100_is_refused():
    sp = {**spec(V1), "steady": {"gpuMemoryUtilization": 0.7, "gpumemPercentage": 75}}
    acts, st = logic.decide("sw", sp, {}, {}, lambda p: (True, ""), 0)
    assert acts == [] and "100" in st["message"]
