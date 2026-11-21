"""DynamoModelSwap's node-side step, run by the controller as a Job pinned to each node of the graph (image: the graph's
own runtime image, /data/hf mounted). Prints one JSON line to /dev/termination-log, which the controller reads.

  MODE=fetch  The download gate. Fetch $MODEL (at $REVISION, default main) into $HF_HOME the way the workers'
              fetch-model init container does (a no-op when it is all there; offline when there is no hub token),
              check that every shard in the safetensors index is on disk, and compare with $COMPARE (the serving
              slot's snapshot dir): everything but the weights and docs must be identical for an in-place reload.
              -> {snapshot, compatible, why, shards, bytes}
  MODE=link   Point the slot link $LINK at $TARGET (a snapshot dir), atomically. Slots run --model $LINK, so a
              restarted worker loads whatever version its slot serves now. -> {link, target}
  MODE=cache-seed     Copy into this node's compile cache $CACHE_DIR every entry of the namespace's cache store
                      $STORE_DIR that the node lacks, so an engine starts warm. Best effort: whatever is missing
                      afterwards the engine compiles itself. -> {copied, bytes, present, ...}
  MODE=cache-publish  The other way: entries of this node's cache that the store lacks, once its slot serves.
"""
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
SKIP = ("README.md", ".gitattributes", "LICENSE", "LICENSE.md", "NOTICE", "USE_POLICY.md")
# Keys of config.json that name the checkpoint, not its shape.
CONFIG_VOLATILE = ("_name_or_path", "transformers_version")


def out(result: dict, code: int = 0) -> None:
    line = json.dumps(result, separators=(",", ":"))
    print(line, flush=True)
    try:
        with open("/dev/termination-log", "w") as f:
            f.write(line[:4000])
    except OSError:
        pass
    sys.exit(code)


def snapshot_of(model: str, revision: str | None, offline: bool) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision or None, local_files_only=offline)


def fetch(model: str, revision: str) -> str:
    """Download with the hub CLI when there is a token (no-op if complete), then resolve the local snapshot."""
    if os.environ.get("HF_TOKEN"):
        cmd = ["hf", "download", model, "--max-workers", "4"] + (["--revision", revision] if revision else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            out({"error": f"hf download failed: {(r.stderr or r.stdout).strip()[-600:]}"}, 1)
    try:
        return snapshot_of(model, revision, offline=True)
    except Exception as e:  # noqa: BLE001 - anything here means "not on this node"
        out({"error": f"{model} is not in {os.environ.get('HF_HOME')} on this node and no hub token to fetch it: "
                      f"{type(e).__name__}: {e}"[:600]}, 1)


def shard_tensors(path: str) -> set[str]:
    """Tensor names in a safetensors file, after checking its header parses and its data fits the file (a truncated
    or half-written shard fails here, not in the engine)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if n <= 0 or 8 + n > size:
            raise ValueError(f"header length {n} does not fit a {size}-byte file")
        header = json.loads(f.read(n))
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    if 8 + n + end > size:
        raise ValueError(f"tensor data ends at byte {8 + n + end} of a {size}-byte file (truncated)")
    return {k for k in header if k != "__metadata__"}


def check_shards(snap: str) -> tuple[int, int]:
    idx = os.path.join(snap, "model.safetensors.index.json")
    wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
    shards = sorted(set(wmap.values())) or sorted(f for f in os.listdir(snap) if f.endswith(".safetensors"))
    if not shards:
        out({"error": f"no safetensors shards in {snap}"}, 1)
    total = 0
    for sh in shards:
        p = os.path.join(snap, sh)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            out({"error": f"shard {sh} missing or empty in {snap}"}, 1)
        try:
            names = shard_tensors(p)
        except (ValueError, OSError, KeyError, TypeError) as e:
            out({"error": f"shard {sh} is not a readable safetensors file: {e}"[:400]}, 1)
        missing = sorted(t for t, f in wmap.items() if f == sh and t not in names)
        if missing:
            out({"error": f"shard {sh} lacks {len(missing)} tensors the index puts there, e.g. {missing[0]}"}, 1)
        total += os.path.getsize(p)
    return len(shards), total


def fingerprint(snap: str) -> dict:
    """{file: sha256} of everything that is not weights or docs; config.json without its volatile keys; the index as
    its tensor names only (shard names and sizes may differ between versions)."""
    fp = {}
    for root, _, files in os.walk(snap):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), snap)
            if f.endswith(WEIGHTS) or f in SKIP or f.endswith(".md"):
                continue
            path = os.path.join(root, f)
            if f == "config.json":
                cfg = json.load(open(path))
                for k in CONFIG_VOLATILE:
                    cfg.pop(k, None)
                data = json.dumps(cfg, sort_keys=True).encode()
            elif f == "model.safetensors.index.json":
                data = json.dumps(sorted(json.load(open(path))["weight_map"])).encode()
            else:
                data = open(path, "rb").read()
            fp[rel] = hashlib.sha256(data).hexdigest()
    return fp


def compare(new: str, old: str) -> tuple[bool, str]:
    a, b = fingerprint(old), fingerprint(new)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return (not diff), ("same config, tokenizer, chat template and tensor names" if not diff
                        else "differs: " + ", ".join(diff[:8]))


def link(path: str, target: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    os.symlink(target, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- compile-cache seeding

# The compile cache the slots mount (vllm/, flashinfer/, triton/). Under these directories every child directory is one
# entry, copied whole or not at all: a torch.compile or AOT artifact, a FlashInfer autotune result, a Triton kernel, a
# FlashInfer JIT op. Everything else is copied file by file. Each name keys its content (a hash of the config, the
# shapes and the environment), so an entry present on both sides is the same and is never overwritten.
ENTRY_PARENTS = ("vllm/torch_compile_cache", "vllm/torch_compile_cache/torch_aot_compile",
                 "vllm/flashinfer_autotune_cache/*/*", "triton/cache", "flashinfer/*/*/cached_ops")
# Never copied. vLLM's startup plan (VLLM_ENABLE_STARTUP_PLAN) is keyed by a hash that ignores
# --gpu-memory-utilization and the ports, so a low bridge's KV budget (0.23) would be applied to the next 0.70 slot.
# Compile and autotune entries do not depend on the memory budget, so the bridge and the full slot may share them.
SKIP_DIRS = ("vllm/startup_plan",)
STAGE = ".dms-stage-"
SETTLE = 120          # publish: leave entries written in the last 2 minutes (an engine may still be writing them)


def _match(rel: str, pattern: str) -> bool:
    a, b = rel.split("/"), pattern.split("/")
    return len(a) == len(b) and all(fnmatch.fnmatchcase(x, y) for x, y in zip(a, b))


def _entry_parent(rel: str) -> bool:
    return any(_match(rel, p) for p in ENTRY_PARENTS)


def _skipped(rel: str, name: str) -> bool:
    return (name.startswith(STAGE) or ".tmp" in name or name.endswith((".lock", ".part"))
            or any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS))


def cache_entries(root: str) -> list[tuple[str, bool]]:
    """(path relative to root, is a whole-directory entry) of everything a sync copies."""
    out = []

    def walk(rel):
        for name in sorted(os.listdir(os.path.join(root, rel) if rel else root)):
            r = f"{rel}/{name}" if rel else name
            p = os.path.join(root, r)
            if _skipped(r, name) or os.path.islink(p):
                continue
            if os.path.isdir(p):
                if _entry_parent(rel) and not _entry_parent(r):
                    out.append((r, True))
                else:
                    walk(r)
            elif os.path.isfile(p):
                out.append((r, False))

    if os.path.isdir(root):
        walk("")
    return out


def _newest(path: str) -> float:
    t = os.path.getmtime(path)
    for r, ds, fs in os.walk(path):
        for x in ds + fs:
            try:
                t = max(t, os.lstat(os.path.join(r, x)).st_mtime)
            except OSError:
                pass
    return t


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def _clear_stale_stages(root: str, age: float = 3600) -> None:
    """Staging copies left by a killed sync (they are never renamed into place)."""
    for r, ds, fs in os.walk(root):
        for x in ds + fs:
            if x.startswith(STAGE):
                p = os.path.join(r, x)
                try:
                    if time.time() - os.lstat(p).st_mtime > age:
                        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.unlink(p)
                except OSError:
                    pass
        ds[:] = [d for d in ds if not d.startswith(STAGE)]


def copy_entry(src: str, dst: str) -> bool:
    """Copy one entry into place atomically and without overwriting: into a staging name beside dst first, then a rename
    (a directory) or a hard link (a file), which fail if dst appeared meanwhile. False when dst was already there."""
    if os.path.lexists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    stage = os.path.join(os.path.dirname(dst), f"{STAGE}{os.getpid()}-{os.path.basename(dst)}")
    try:
        if os.path.isdir(src):
            shutil.copytree(src, stage, symlinks=True, copy_function=shutil.copy2)
            try:
                os.rename(stage, dst)      # fails on a non-empty dst: someone else put the entry there first
            except OSError:
                return False
        else:
            shutil.copy2(src, stage)
            try:
                os.link(stage, dst)
            except FileExistsError:
                return False
        return True
    finally:
        if os.path.lexists(stage):
            shutil.rmtree(stage, ignore_errors=True) if os.path.isdir(stage) and not os.path.islink(stage) \
                else os.unlink(stage)


def sync(src: str, dst: str, deadline: float, settle: float = 0) -> dict:
    """Copy every entry of src that dst lacks. Never overwrites, never deletes, stops at the deadline. A failed entry is
    counted and skipped: the engine compiles whatever is missing."""
    res = {"copied": 0, "bytes": 0, "present": 0, "busy": 0, "failed": 0, "errors": []}
    if not os.path.isdir(src):
        return {**res, "note": f"{src} does not exist yet: nothing to copy"}
    os.makedirs(dst, exist_ok=True)
    _clear_stale_stages(dst)
    for rel, _ in cache_entries(src):
        if time.time() > deadline:
            res["timedOut"] = True
            break
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        if os.path.lexists(d):
            res["present"] += 1
            continue
        try:
            if settle and time.time() - _newest(s) < settle:
                res["busy"] += 1
                continue
            if copy_entry(s, d):
                res["copied"] += 1
                res["bytes"] += _size(d)
            else:
                res["present"] += 1
        except OSError as e:
            res["failed"] += 1
            if len(res["errors"]) < 3:
                res["errors"].append(f"{rel}: {type(e).__name__}: {e}"[:200])
    return res


if __name__ == "__main__":
    mode = os.environ.get("MODE", "fetch")
    if mode in ("cache-seed", "cache-publish"):
        cache, store = os.environ["CACHE_DIR"], os.environ["STORE_DIR"]
        deadline = time.time() + float(os.environ.get("CACHE_SECONDS", "240"))
        os.nice(10)
        try:
            r = sync(store, cache, deadline) if mode == "cache-seed" else sync(cache, store, deadline, SETTLE)
        except Exception as e:  # noqa: BLE001 - best effort: report it, the swap goes on
            out({"mode": mode, "error": f"{type(e).__name__}: {e}"[:400]}, 1)
        out({"mode": mode, **r})
    if mode == "link":
        target = os.environ["TARGET"]
        if not os.path.isfile(os.path.join(target, "config.json")):
            out({"error": f"{target} is not a model snapshot"}, 1)
        link(os.environ["LINK"], target)
        out({"link": os.environ["LINK"], "target": os.path.realpath(os.environ["LINK"])})
    model, revision = os.environ["MODEL"], os.environ.get("REVISION") or ""
    snap = fetch(model, revision)
    n, size = check_shards(snap)
    res = {"model": model, "snapshot": snap, "shards": n, "bytes": size}
    old = os.environ.get("COMPARE") or ""
    if old:
        if os.path.isfile(os.path.join(old, "config.json")):
            res["compatible"], res["why"] = compare(snap, old)
        else:
            res["compatible"], res["why"] = False, f"serving snapshot {old} is not on this node"
    out(res)
