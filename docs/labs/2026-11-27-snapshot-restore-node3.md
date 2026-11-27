---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Snapshot restore on B300 node-3 (2026-11-27)
subtitle: Single-GPU and eight-GPU restore with CRIU's legacy mount engine, and the 615 user-mode driver multi-GPU needs.
---

## Summary

Single-GPU and single-node eight-GPU restores now work on the B300 lab node with
CRIU's legacy mount engine (`config.criu.mntnsCompatMode: true`). With the option
off, the same node reproduces the mount-v2 failure that blocks restore for
NVIDIA-runtime bind mounts:

```text
mnt-v2: Failed to copy sharing from source /dev/nvidia1 to 7885
```

Multi-GPU additionally needs the cuinterpose launcher and a 615 user-mode
driver (`cuda-compat-13-4`); the 580 kernel driver is unchanged.

## Environment

| Item | Value |
|---|---|
| Node | B300 lab node (node-3), 8x B300, sm_103 |
| Kernel | 6.6.102-5.500.al8, Alibaba Cloud Linux 3.2104 U12.1 |
| Kernel driver | 580.105.08 (open modules), unchanged |
| CUDA | 13.0 runtime image, cu130 |
| Container runtime | containerd v2.3.2-k3s2 (k3s v1.36.2+k3s1) |
| GPU allocator | HAMi 2.9.0 |
| Snapshot | chart 0.3.0, agent/operator/PageBroker v0.3.0 (the lab's stack) |
| Workload image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0` |
| Storage | 100 GiB RWX PVC on `nfs-models` |
| PageBroker staging | 192 GiB, agent memory limit 192 GiB |

The workload pods run with the chart's `profiles/block-iouring.json` seccomp
profile. Capture and restore happen on the same node and the same driver build.

## Reproduction

Create a GPU pod, capture it with a `PodSnapshot`, then create a restore pod with
`nvidia.com/restore-from`, an `emptyDir` named `snapshot-control` mounted at
`/snapshot-control` with `subPath: main`, and `SNAPSHOT_CONTROL_DIR=/snapshot-control`.
With `mntnsCompatMode` false the restore fails at CRIU's mount-v2 engine; with it
true the same checkpoint restores.

## Control: mount-v2 fails

A single-GPU `python3` workload holding one `torch` tensor was captured with
`config.criu.mntnsCompatMode: false`. The checkpoint verified, and the restore
failed with:

```text
nvidia.com/Restored=False  RestoreFailed: ... while restoring with CRIU:
CRIU restore failed: mnt-v2: Failed to copy sharing from source /dev/nvidia1 to 7885
```

The same signature was reported upstream in
[ai-dynamo/snapshot#144](https://github.com/ai-dynamo/snapshot/issues/144) and
[checkpoint-restore/criu#2601](https://github.com/checkpoint-restore/criu/issues/2601);
CRIU's mount-v2 cannot copy the slave sharing group of the NVIDIA device bind
mounts, and `--mntns-compat-mode` selects the older mount engine that can.

## Single GPU: restore succeeds

With `config.criu.mntnsCompatMode: true` set before capture, the same tiny
workload restored:

| Phase | Time |
|---|---|
| CRIU restore | 748 ms |
| CUDA restore | 373 ms |
| Total restore | 1.33 s |
| Capture (CRIU dump 874 ms + CUDA checkpoint 3.97 s) | 6.11 s |

Evidence: `nvidia.com/Restored=True`, the restore proxy is PID 1, the restored
`python3` process is the checkpointed command, and the GPU shows 772 MiB in use
by the restored process.

The option is a restore-only CRIU field, but the agent persisted its dump-time
value into `manifest.yaml`. A checkpoint captured with the option off therefore
restored with mount-v2 even after the chart value was flipped. This branch makes
the agent pass its live value to `nsrestore`, so the setting applies to existing
checkpoints too.

## Eight GPUs on one node: restore succeeds

A workload allocating a tensor on all eight GPUs was captured and restored with
the same option, plus the multi-GPU contract:

- source pod annotation `nvidia.com/snapshot-multi-gpu: enabled`;
- the cuinterpose libraries and launcher copied into `/tmp/snapshot-cuda` by an
  init container, mounted read-only in the workload container;
- the workload started through
  `/tmp/snapshot-cuda/cuinterpose-launch --library /tmp/snapshot-cuda/libcuinterpose.so -- ...`;
- a 615 user-mode driver: `cuda-compat-13-4` (615.71.10) extracted to a host
  directory and put first in `LD_LIBRARY_PATH` for the workload, the agent, and
  PageBroker. The kernel driver stays at 580.105.08; nothing was installed or
  rebooted.

| Phase | Time |
|---|---|
| CRIU restore | 475 ms |
| CUDA restore (cuinterpose) | 4.79 s |
| Total restore | 6.13 s |

Evidence: `nvidia.com/Restored=True`; the restored `python3` process owns
allocations on all eight GPUs (1,386 MiB per GPU inside the pod); the agent log
shows `mounting cuinterpose libraries into placeholder namespace` and
`CRIU post-restore: process restored`.

## What the vendor docs say, and where they differ

The Dynamo v1.6.1 operations page lists driver 580 or newer, and 590 or newer
"if testing multi-GPU snapshots". The snapshot project's prerequisites require a
615 user-mode driver for multi-GPU, and this lab confirms that boundary: the
580 kernel driver is enough for single-GPU restore, while multi-GPU capture
needs the cuinterpose launcher and a 615 `libcuda.so`. The 590 figure in the
Dynamo page does not correspond to any check in the code (`minDriverMajor` is
580) or to a tested configuration here.

Neither the Dynamo page nor the snapshot compatibility page mentions the
NVIDIA-runtime bind-mount failure or `mntnsCompatMode`; both are absent from the
prerequisites even though restore fails without the option on this stack.

## Scope

- Node-4 was not touched; all work ran on node-3.
- Cross-node prefill/decode restore was not attempted. The snapshot project
  lists multi-node as unsupported, and NIXL/RDMA state is not restorable.
- The lab used the standalone snapshot chart 0.3.0 and its operator. The
  Dynamo chart in this branch is the operator-driven variant; its
  `mntnsCompatMode` default and restore-time override are the same code path,
  covered by the unit test in `deploy/snapshot/internal/criu/restore_test.go`.
