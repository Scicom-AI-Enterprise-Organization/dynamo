# snapshot_nvidia_vma

A hook-only CRIU plugin that claims NVIDIA character-device VMAs during dump.

## Why

CRIU refuses non-regular mappings unless a loaded plugin claims them through
`HANDLE_DEVICE_VMA`. vLLM and NCCL keep small shared mappings of
`/dev/nvidiaN`; the CRIU CUDA plugin does not register that hook, so a dump
fails with:

```text
handle_device_vma plugin failed: No such file or directory
Can't handle non-regular mapping
```

The agent already checkpoints CUDA state through `cuda-checkpoint-helper`.
Loading the full CUDA plugin would run a second `PAUSE_DEVICES` /
`CHECKPOINT_DEVICES` / `RESUME_DEVICES_LATE` against that same process. This
plugin registers only `HANDLE_DEVICE_VMA`, returns 0 for the NVIDIA and UVM
device majors, and lets the driver recreate the mapping at restore.

## Build

The agent image builds it into the chart's dedicated plugin directory:

```text
/usr/local/lib/snapshot/criu-plugins/snapshot_nvidia_vma.so
```

alongside `snapshot_inet_remap.so`. The chart's default `config.criu.libDir`
points there, so CRIU loads both and never loads the full CUDA plugin from
`/usr/local/lib/criu`.

## Test

```bash
make test
```

The unit test covers the major-matching rule: 195 (`nvidiactl`/`nvidiaN`),
499 (`nvidia-uvm`), and unrelated majors. The end-to-end check is a PodSnapshot
of a CUDA process that holds a `/dev/nvidia*` mapping; see
`docs/labs/2026-11-27-snapshot-pd-node3.md`.
