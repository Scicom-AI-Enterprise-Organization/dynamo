"""DynamoModelSwap's node-side step, run by the controller as a Job pinned to each node of the graph (image: the graph's
own runtime image, /data/hf mounted). Prints one JSON line to /dev/termination-log, which the controller reads.

  MODE=fetch  The download gate. Fetch $MODEL (at $REVISION, default main) into $HF_HOME the way the workers'
              fetch-model init container does (a no-op when it is all there; offline when there is no hub token),
              check that every shard in the safetensors index is on disk, and compare with $COMPARE (the serving
              slot's snapshot dir): everything but the weights and docs must be identical for an in-place reload.
              -> {snapshot, compatible, why, shards, bytes}
  MODE=link   Point the slot link $LINK at $TARGET (a snapshot dir), atomically. Slots run --model $LINK, so a
              restarted worker loads whatever version its slot serves now. -> {link, target}
"""
import hashlib
import json
import os
import subprocess
import sys

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


if __name__ == "__main__":
    mode = os.environ.get("MODE", "fetch")
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
