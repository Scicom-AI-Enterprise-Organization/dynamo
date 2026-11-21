"""The node Job's compile-cache sync. Run: uv run --with pytest pytest controller/test_fetch.py"""
import os
import shutil
import time

import fetch


def put(root, rel, text="x"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)
    return p


def read(root, rel):
    with open(os.path.join(root, rel)) as f:
        return f.read()


def listing(root):
    return sorted(os.path.relpath(os.path.join(r, f), root) for r, _, fs in os.walk(root) for f in fs)


def test_seed_adds_what_the_node_lacks_and_never_overwrites_or_mixes(tmp_path):
    store, node = str(tmp_path / "store"), str(tmp_path / "node")
    for rel in ("vllm/torch_compile_cache/abc/rank_0_0/backbone/a", "vllm/torch_compile_cache/abc/rank_1_0/backbone/a",
                "vllm/torch_compile_cache/torch_aot_compile/k1/rank_0_0/model",
                "vllm/flashinfer_autotune_cache/0.7.0/103a/h1/autotune_configs.json",
                "triton/cache/T1/kernel.so", "vllm/startup_plan/startup_plan_x.json"):
        put(store, rel, "store")
    put(store, "vllm/modelinfos/m.json", "store")
    put(node, "vllm/modelinfos/m.json", "node")                                  # present: kept as it is
    put(node, "vllm/torch_compile_cache/abc/rank_1_0/backbone/a", "node")        # an entry the node has (in part)
    r = fetch.sync(store, node, time.time() + 60)
    assert read(node, "vllm/modelinfos/m.json") == "node"
    assert not os.path.exists(os.path.join(node, "vllm/torch_compile_cache/abc/rank_0_0"))   # entries stay whole
    assert not os.path.exists(os.path.join(node, "vllm/startup_plan"))           # keyed without the memory budget
    assert read(node, "vllm/torch_compile_cache/torch_aot_compile/k1/rank_0_0/model") == "store"
    assert read(node, "vllm/flashinfer_autotune_cache/0.7.0/103a/h1/autotune_configs.json") == "store"
    assert read(node, "triton/cache/T1/kernel.so") == "store"
    assert (r["copied"], r["present"], r["failed"]) == (3, 2, 0)
    assert not [p for p in listing(node) if fetch.STAGE in p]
    assert fetch.sync(store, node, time.time() + 60)["copied"] == 0             # a second run copies nothing


def test_a_copy_that_breaks_halfway_leaves_no_partial_entry(tmp_path, monkeypatch):
    store, node = str(tmp_path / "store"), str(tmp_path / "node")
    put(store, "vllm/torch_compile_cache/abc/rank_0_0/a")
    put(store, "vllm/torch_compile_cache/abc/rank_1_0/a")
    put(store, "vllm/modelinfos/m.json")
    calls = {"n": 0}
    real = shutil.copy2

    def flaky(src, dst, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(src, dst, **kw)

    monkeypatch.setattr(shutil, "copy2", flaky)
    r = fetch.sync(store, node, time.time() + 60)
    assert r["failed"] == 1 and r["errors"][0].startswith("vllm/torch_compile_cache/abc: ")
    assert listing(node) == ["vllm/modelinfos/m.json"]                           # nothing of abc, no staging copy
    monkeypatch.setattr(shutil, "copy2", real)
    assert fetch.sync(store, node, time.time() + 60)["copied"] == 1              # the next sync completes it


def test_publish_leaves_entries_an_engine_may_still_be_writing(tmp_path):
    node, store = str(tmp_path / "node"), str(tmp_path / "store")
    old = put(node, "vllm/torch_compile_cache/old/rank_0_0/a")
    put(node, "vllm/torch_compile_cache/new/rank_0_0/a")
    hour_ago = time.time() - 3600
    for p in (old, os.path.dirname(old), os.path.dirname(os.path.dirname(old))):
        os.utime(p, (hour_ago, hour_ago))
    r = fetch.sync(node, store, time.time() + 60, settle=fetch.SETTLE)
    assert (r["copied"], r["busy"]) == (1, 1)
    assert listing(store) == ["vllm/torch_compile_cache/old/rank_0_0/a"]


def test_missing_store_or_past_deadline_is_reported_not_raised(tmp_path):
    node = str(tmp_path / "node")
    assert "does not exist yet" in fetch.sync(str(tmp_path / "nope"), node, time.time() + 60)["note"]
    store = str(tmp_path / "store")
    put(store, "vllm/modelinfos/m.json")
    r = fetch.sync(store, node, time.time() - 1)
    assert r["timedOut"] and r["copied"] == 0
