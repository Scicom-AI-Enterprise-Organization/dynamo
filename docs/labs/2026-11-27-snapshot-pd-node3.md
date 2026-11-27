---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Prefill/decode TP4 pair on one B300 node (2026-11-27)
subtitle: The pair serves with NIXL over RoCE, but a PodSnapshot of the live workers cannot be captured; the snapshot lifecycle is a one-shot startup gate.
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
