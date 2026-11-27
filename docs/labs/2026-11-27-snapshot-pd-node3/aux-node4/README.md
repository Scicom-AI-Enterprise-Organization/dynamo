<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Aux stack on node-4: exported before teardown (2026-11-27)

Taken down on 2026-11-27 so all eight L20D cards on `tm-ucc-hpfsset5-0001`
(node-4) could be used for the cross-node snapshot test: capture the aggregated
TP8 engine on node-3, restore it on node-4.

The four namespaces `aux-bench`, `embed-rerank`, `speech` and `mps-control` were
deleted. Their live manifests are in this directory, one JSON `List` per
namespace, with `status`, `managedFields`, `resourceVersion`, `uid`,
`creationTimestamp` and the last-applied annotations stripped.

## What was there

| Namespace | Contents |
|---|---|
| `aux-bench` | Deployment `bench` (four replicas running `sleep infinity`, CPU-only, no GPU requests) and the `bench-script` ConfigMap |
| `embed-rerank` | `qwen3-embedding-8b` and `jina-reranker-v35` vLLM deployments, one GPU each |
| `speech` | `stt-vllm-cc18ab2d` (+ inactive `stt-vllm-tp2`/`stt-vllm-tp4`), `tts-vllm-ce85e930`, `stt-api`, `tts-api`, `eot`, `enhancer`, `silero-vad`, plus model-fetch/build Jobs |
| `mps-control` | DaemonSet `mps-control`, pinned to node-4 only; provides the CUDA MPS server and control socket |

All model data lives on hostPath `/data/hf` and `/data/hf/voice-aux` and was
**not** deleted. There were no PVCs in these namespaces.

## Secrets (not exported)

Secret values were deliberately left out of the manifests. Recreate these
before redeploying:

| Namespace | Secret | Key |
|---|---|---|
| `embed-rerank` | `regcred` | `.dockerconfigjson` |
| `speech` | `regcred` | `.dockerconfigjson` |
| `speech` | `hf-token` | `token` |
| `speech` | `aif-hub-token` | `token` |
| `speech` | `silero-vad-auth` | `SILERO_VAD_API_KEY` |

## Redeploy

```bash
kubectl create namespace aux-bench
kubectl create namespace embed-rerank
kubectl create namespace speech
kubectl create namespace mps-control
# recreate the secrets listed above in their namespaces
kubectl apply -f aux-bench.json
kubectl apply -f embed-rerank.json
kubectl apply -f speech.json
kubectl apply -f mps-control.json
```

The deployments pin `kubernetes.io/hostname: tm-ucc-hpfsset5-0001` and expect
`/data/hf`, `/data/hf/voice-aux`, `/tmp/nvidia-mps` and the CUDA 13 host
binaries to exist. `mps-control` must be applied before the speech pods if
those pods expect the MPS control socket.

## Proxy

The AI Foundry aux routes were moved back to H20 before this teardown, so no
route pointed at these services when the namespaces were deleted. The removed
proxy upstreams were `b300-node4-embedding`, `b300-node4-reranker`,
`b300-node4-stt`, `b300-node4-tts`, `b300-node4-eot`,
`b300-node4-voice-isolator` and `b300-node4-stt-api`.
