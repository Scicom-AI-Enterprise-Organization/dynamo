---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Prefill/decode TP4 pair on one B300 node (2026-11-27)
subtitle: The pair serves with NIXL over RoCE; with the VMA plugin a PodSnapshot of the live workers captures, but the prefill snapshot is one-shot.
---

## What ran

A lab `DynamoGraphDeployment` in `snapfix-lab`, on node-3 only, from the
retired prod TP4 graph:

| Component | GPUs | Ports | Settings |
|---|---|---|---|
| Frontend | none | 8000 | round-robin |
| Prefill | node-3 GPUs 0-3 | 9090 / 19090 / 20380 / 5600 / 20080 | TP4, NIXL, util 0.06, max-len 8192 |
| Decode | node-3 GPUs 4-7 | 9091 / 19091 / 20381 / 5700 / 20081 | TP4, NIXL, util 0.06, max-len 8192 |

Model: the prod v20a Gemma 4 31B NVFP4/FP8, read from local `/data/hf`, served as
`Scicom-intl/gemma-4-31B-it-dpo-repetition-v3.3-handover-v20a-2026-10-27-NVFP4-FP8-attn`.
Workers were `hostNetwork`, `runtimeClassName: nvidia`, seccomp `block-iouring`,
`rdma/rdma_shared_device_a: 1`, and the snapshot-control volume.

Before standing it up, node-3 had all eight cards at 4 MiB with zero compute
processes and no prod pod; prod was on node-1 (prefill) and node-2 (decode).
Node-4 was not touched.

## What worked

The pair served end to end through the frontend:

| Measurement | Value |
|---|---|
| TTFT, 6,360-character prompt, 5 requests | 141, 142, 141, 142, 168 ms |
| NIXL transfers (first baseline) | 20 successful, 0 failed, 29.3 GB/s, 13.75 MB per transfer |
| Frontend readiness | prefill 1, decode 1 |

## Capture attempts

### 1.4.0 agent, placeholder image, `cuda-checkpoint --launch-job`

Built `localhost/snapfix/dynamo-vllm-placeholder:v1.6.0-cu130` from the prod
runtime image plus the 1.4.0 agent's `nsrestore`, `criu`, `cuda-checkpoint`,
and CRIU libraries. A synthetic 4-GPU pod under
`cuda-checkpoint --launch-job` captured in **4.94 s** (CRIU 2.94 s, CUDA 1.96 s),
so the branch's own path works for a simple multi-GPU process.

The vLLM TP4 pair failed in `cuCheckpointProcessLock`:

```text
cuda-checkpoint-helper [--action checkpoint --pid 1419594] failed after 2m9.7s:
CUDA_ERROR_OPERATING_SYSTEM: OS call failed or operation not supported on this OS
```

(prefill: 3m43.8s; decode: 2m9.7s, on `VLLM::Worker_TP0`). Both workers were
using `/cuda-compat/libcuda.so.615.71.10` at the time.

### 2. 0.3.0 agent, cuinterpose, 615 user-mode

PageBroker died during the GPU checkpoint:

```text
cuCheckpointOperationComplete: CUDA_ERROR_OPERATING_SYSTEM
PageBroker fatal cleanup: GPU operation cannot release resources
```

PageBroker exited 1, the agent read EOF from it, and both snapshots failed.
PageBroker was confirmed on `/cuda-compat/libcuda.so.615.71.10`.

### 3. 0.3.0 agent, cuinterpose, 580 driver-managed

With PageBroker, the agent, and the workers all on
`libcuda.so.580.105.08`, the prefill failed at the CUDA checkpoint in **4.4 s**
with the same `CUDA_ERROR_OPERATING_SYSTEM`; the decode failed in CRIU's dump
with the `binfmt_misc: Operation not permitted` first error (the known red
herring; the fatal line is not surfaced).

## Why it fails

The Dynamo snapshot lifecycle is a **one-shot startup gate**. `prepare_snapshot_engine`
(`dynamo/vllm/main.py:237`, `dynamo/vllm/snapshot.py`) loads the engine, pauses
it, writes `ready-for-snapshot`, and waits for `snapshot-complete` or
`restore-complete`. Once the sentinel arrives the engine resumes and returns to
normal serving; a later `PodSnapshot` cannot pause it again. The lab workers had
already resumed (they served requests), so `cuda-checkpoint` tried to lock a
live vLLM engine and failed.

The supported flow starts the pod for the checkpoint: a `SnapshotJob` or a
Dynamo `DynamoCheckpoint` Job, whose pod runs under the multi-GPU launcher and
is still paused when the agent captures it. A `PodSnapshot` against an already
serving DGD worker is not that flow.

Because no checkpoint was produced, no restore was attempted. The restored
pair's TTFT and the NIXL/RDMA state after restore were not measured; upstream's
position that NIXL/RDMA state is not restorable remains untested here.

## Next step

Run the pair as two checkpoint jobs (SnapshotJob or DynamoCheckpoint) with the
same pod template, ports, and model, so each engine is captured while paused at
startup; then restore both into the pair's pods and repeat the TTFT and NIXL
checks.

## 2026-11-27 follow-up: NVIDIA VMA plugin, capture, and the wall-live

The 0.3.0 cuinterpose path removes shared mappings but not the small shared
mappings of `/dev/nvidiaN` that vLLM keeps. CRIU's `proc_parse.c` calls
`run_plugins(HANDLE_DEVICE_VMA)` for those; neither the CUDA plugin nor the
inet-remap plugin registers that hook, so the dump failed with
`handle_device_vma plugin failed: No such file or directory`.

The fix is a hook-only plugin, `deploy/snapshot/plugins/nvidia-vma`, installed
as `/usr/local/lib/snapshot/criu-plugins/snapshot_nvidia_vma.so`. It claims the
NVIDIA and UVM majors (195 and 499, resolved at init with those as fallback) and
registers no pause/checkpoint/resume hooks, so it cannot race the agent's
`cuda-checkpoint-helper`. The full CUDA plugin stays out of the chart's plugin
directory.

Configuration used for the successful captures (node-3 only):

| Item | Value |
|---|---|
| Kernel driver | 580.105.08 (unchanged) |
| CUDA user-mode | 615.71.10 via `/cuda-compat` (`LD_LIBRARY_PATH`) |
| Snapshot agent | 0.3.0, `config.criu.libDir: /criu-plugins` |
| Plugins | `snapshot_inet_remap.so` + `snapshot_nvidia_vma.so`; `cuda_plugin.so` disabled |
| Worker launch | `cuinterpose-launch --library /tmp/snapshot-cuda/libcuinterpose.so -- python3 -m dynamo.vllm` |
| Restore | `config.criu.mntnsCompatMode: true` |
| Same-node sockets | `VLLM_HOST_IP=127.0.0.1`, `VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo`, `TP_SOCKET_IFNAME=lo`, `NCCL_SOCKET_IFNAME=lo` |
| Engine timeout | `VLLM_ENGINE_ITERATION_TIMEOUT_S=600` |

Capture results with the plugin (both `Ready=True`, "Checkpoint captured and
verified"):

| Worker | Total | CRIU dump | CUDA checkpoint |
|---|---|---|---|
| Prefill TP4 | 2m41s | 55.1s | 24.0s |
| Decode TP4 | 2m8s | 50.0s | 17.2s |

The restore also completed:

| Worker | Total | CRIU restore | CUDA restore |
|---|---|---|---|
| Prefill TP4 | 1m21s | 70.7s | 6.3s |
| Decode TP4 | 42.3s | 34.6s | 4.2s |

Both restored processes resumed and re-captured CUDA graphs, but the decode
TP4 group then failed with a Gloo TCP read timeout
(`gloo/transport/tcp/pair.cc:537`, `10.0.11.1:3481`). The cause was not the
snapshot code: node-3's `wall-live` pusher installs
`-A OUTPUT -s 10.0.11.1/32 -d 10.0.11.1/32 -p tcp -j DROP` (758 packets hit;
the duplicate `INPUT` copies had 0 hits). Host-network vLLM talks to its own
host IP for the TP/NIXL bootstrap, so the wall-live drops it. Pinning the
same-node interfaces to `lo` removes that path; the pair then reaches the
snapshot gate and captures cleanly.

The post-restore serving check is not yet green: after the source pods were
force-removed, node-3's kubelet stopped posting status because its local
`127.0.0.1:6444` API proxy timed out. The restore pods are Pending on the
NotReady node, so no request has been served after restore. The node needs a
k3s-agent restart on node-3 (no reboot; prod node-1/node-2 unaffected) before
the final restore-serving run. Cross-node restore was not attempted: node-4 is
out of bounds, and a restore on a new node must re-resolve
`VLLM_NIXL_SIDE_CHANNEL_HOST` to that node's host IP instead of the captured
loopback value.

## Final lab configuration (2026-11-27, node-3)

Environment: B300 node-3 (`tm-ucc-c3ssfjix-0000`), kernel driver 580.105.08,
CUDA 13 user-mode via `/data/snapfix/cuda-compat` (615.71.10), k3s v1.36.2+k3s1,
HAMi, snapshot agent 0.3.0, chart `libDir=/criu-plugins`.

Repro fixed here: `criu/mount-v2.c:897: Failed to copy sharing from
-1:/dev/nvidia0 to 101: Invalid argument`, then
`cr-restore.c:2419: Restoring FAILED`.

Capture uses the hook-only `snapshot_nvidia_vma` CRIU plugin (majors 195/499)
plus inet-remap; the full CRIU CUDA plugin is disabled. The pair is pinned to
loopback for same-node capture/restore: `VLLM_HOST_IP=127.0.0.1`,
`VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo`,
`TP_SOCKET_IFNAME=lo`, `NCCL_SOCKET_IFNAME=lo`,
`VLLM_ENGINE_ITERATION_TIMEOUT_S=600`.

Capture: prefill 2m41s (CRIU 55.1s, CUDA 24.0s), decode 2m8s (CRIU 50.0s,
CUDA 17.2s), both `Ready=True`. Restore: prefill 1m21s (CRIU 70.7s, CUDA 6.3s),
decode 42.3s (CRIU 34.6s, CUDA 4.2s).

## Restore result

The restore pods schedule on node-3 once the k3s agent is Ready and the RDMA
device plugin has re-registered. In this run `rdma-shared-dp-ds-k5bd2`
restarted at 06:34:53Z, logged `exposing "63" devices`, and the HAMi scheduler
caught up; the pods scheduled at 06:40:15/17Z.

Decode restored (`Restore complete`, 06:41:44Z) but stayed 0/1 with
`No available shared memory broadcast block found in 60 seconds`. Prefill
failed with `Restore already used by an earlier container; restores are not
repeatable` (06:41:45Z) and went into CrashLoopBackOff. The
`snapfix-pd-prefill` PodSnapshot is consumed; a fresh capture under a new
snapshot name is required before the serving/TTFT/NIXL proof can complete.
