# model-swap: change a Dynamo graph's model with no downtime, on one replica

A Scicom add-on, not in upstream `ai-dynamo/dynamo`. It adds the `DynamoModelSwap` custom resource (short name `dms`)
and a small controller that runs a swap from **one edit of `spec.model`** (or `spec.revision`). Callers keep sending
the same `--served-model-name` throughout, and no second replica is needed.

Every swap starts with a **download gate**. When only the weights change, the controller then **reloads them into the
running engines in seconds**. Anything else goes through a **0.27 bridge** copy beside the old engine, which takes
about 25 minutes.

Why a separate controller: the Dynamo operator (upstream v1.6.0 and this fork) handles any worker change as a new
worker-hash generation per component. It has no smaller bridge step and no paired prefill + decode swap, and a second
full-size generation does not fit beside the first on the same GPUs.

## How a swap runs

### Download gate (every swap)

A Job on each node of the graph, on the graph's own runtime image with the node's `/data/hf` mounted
(`controller/fetch.py`):

1. It fetches `spec.model` at `spec.revision` from the hub with `hf download`. It uses the Secret in `fetch.tokenSecret`
   (default `aif-hub-token`), and works offline when that Secret is absent.
2. It parses every shard's safetensors header and checks that the data fits the file and holds the tensors the index
   lists, so a truncated or half-written shard fails here and not in an engine.
3. It compares everything but the weights and docs with the serving snapshot: `config.json` (without
   `_name_or_path`), the tokenizer, the chat template, and the tensor names.

All nodes must hold the same revision. Until the gate passes, nothing changes in the cluster. If it fails, the swap
stops there and the old model keeps serving.

### In place (only the weights changed)

The controller uses this path only when all of these hold:
- the gate found the versions identical but for their weights;
- nothing else in `spec` changed;
- `spec.reload` is `Auto`;
- `versionAlias` is `false`.

It then reloads the running pair through the workers' `/engine/` routes:
1. Pause the prefill, then the decode, in `wait` mode, so running requests finish and new ones queue. Each pause may
   take at most `pauseCapSeconds` (60).
2. Load the new weights into both with `update_weights_from_disk`.
3. Resume the decode, then the prefill.
4. Check: both workers report the new version, and a greedy "count from 1 to 20" through every frontend pod answers
   1..5 in order.

Any failure loads the old weights back, resumes, and falls back to the bridge. Slots load `--model` from a link per
port set on each node (`/data/hf/swap/<namespace>/<swap>-<a|b>`), which the controller points at the slot's version
before it starts or reloads it, so a restarted worker comes back on the version its slot serves now.

### Bridge (anything else)

Each step is its own workers-only DynamoGraphDeployment (a "slot"), owned by the DynamoModelSwap and named
`<frontendGraph>-<swap>-<f|b><n>`. A long-lived frontend-only graph called `<frontendGraph>` in the same namespace
finds every slot through Dynamo's namespace-prefix discovery. The operator gives that frontend
`DYN_NAMESPACE_PREFIX=<namespace>-<frontendGraph>` (a literal `starts_with`, `lib/runtime/src/namespace.rs`). The
frontend routes only to worker sets that have both a prefill and a decode, so a request's prefill and decode always
come from the same slot (`lib/llm/src/discovery/model.rs`).

| Phase | What exists | Moves on when |
|---|---|---|
| Serving | full(old, 0.70) | `spec` changes |
| Bridging | + bridge(new, 0.27) on the same cards | the frontend has routed a request to the bridge |
| DrainingOld | − full(old): its prefill goes to 0 replicas first, then the slot is deleted | its pods are gone (GPU memory and host ports free) |
| Loading | + full(new, 0.70) | the frontend has routed a request to it |
| Finalizing | − bridge, removed the same way | its pods are gone |

- **The frontend acknowledges a slot** when every pod of the frontend graph lists the slot's prefill and decode in
  `/health`, and its `/metrics` has a `dynamo_frontend_worker_*` series for the slot's decode instance. With no live
  traffic the controller sends its own short probe requests.
- **Removal is prefill first.** Without a prefill the slot is an incomplete worker set, so the frontend stops picking
  it at once, while requests it already prefilled finish on its decode. Deleting prefill and decode together failed 3
  requests in the first lab run (503 `no backend worker available`): the frontend does not move a request to another
  slot between prefill and decode.
- **Failure handling.** A bridge that fails, or is not routed to within `readyTimeoutSeconds` (default 1800), is
  removed and the old model keeps serving. A full engine that fails `fullRetries + 1` times leaves the swap `Degraded`
  on the bridge. To retry, edit the spec again.
- **GPU preflight.** Before it creates an engine, the controller checks HAMi room on the pinned cards, and that no pod
  on those cards (in any namespace) is still loading or exiting. Two engines profiling free memory at once miscount
  each other.
- **Names.** Every slot gets `--served-model-name <servedModelName> <servedModelAliases...> [<model>]`.
  - The primary name never changes.
  - `servedModelAliases` are fixed extra names, such as the name callers sent before the swap existed.
  - With `versionAlias: true` (the default), the HF id in `spec.model` is an alias too, so a caller can pin one version
    during a swap. A running engine cannot change its names, though, so in-place reloads need `versionAlias: false`.
- **What the controller owns.** It sets `--model`, `--served-model-name`, `--gpu-memory-utilization`, the host ports
  (fulls use set a, 9090/9091; bridges set b, 9092/9093; plus the NIXL, side-channel, forward-pass-metric and
  kv-events ports) and `nvidia.com/gpumem-percentage` on each slot, and strips them from `spec.graph`. Everything
  else in `spec.graph` is copied to every slot as written.
- **State.** All of it lives in the CR's `status`, so a restarted controller picks up mid-swap. It is a plain loop
  every 10 s, with no finalizers, so deleting the controller never holds a CR.

```bash
kubectl get dms -w                                       # Phase, Serving, Message
kubectl get dms gemma -o jsonpath='{.status.history}'    # each step with its time
kubectl logs deploy/model-swap -f                        # the release's full name
```

## Requirements

- The Dynamo operator, any release with `nvidia.com/v1beta1` DynamoGraphDeployments, and its platform (discovery).
- HAMi. Each worker pins its cards with the `nvidia.com/use-gpuuuid` annotation, and the steady and bridge
  `gpumemPercentage` must add up to 100 or less (default 70 + 27).
- Workers with `hostNetwork: true`, as in the example, and one DynamoModelSwap per node: the host port sets are fixed.
- In each worker, `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` longer than the longest stream (60 s in the lab) and below
  `terminationGracePeriodSeconds`. The fork's default of 5 s cut streams.
- Each worker's nodeSelector names its node (`kubernetes.io/hostname`); the gate and the links run there. The nodes
  keep weights in `/data/hf` (hostPath), and the engines read it with `HF_HUB_OFFLINE=1`.
- Network from the controller pod to the frontend pods on port 8000, and to the workers' system ports for in-place
  reloads. The chart runs the controller on the host network (`hostNetwork: true`), which a node firewall that keeps
  pods off `/engine/` (B300's `admin-port-guard`) lets through.

## Install

```bash
helm install model-swap deploy/helm/charts/model-swap -n <namespace>
kubectl -n <namespace> apply -f deploy/helm/charts/model-swap/examples/frontend-graph.yaml
kubectl -n <namespace> apply -f deploy/helm/charts/model-swap/examples/dynamomodelswap-gemma.yaml
```

The examples are the B300 reload lab's (node-2, Gemma-4-31B NVFP4, prefill TP4 on cards 0-3 and decode TP4 on cards
4-7). Change the GPU UUIDs, node name, image, pull secret, RDMA resource and UCX devices for another cluster.

| Value | Default | |
|---|---|---|
| `watchNamespace` | release namespace | where the DynamoModelSwaps, their slots and the frontend graph live |
| `image.repository`, `image.tag` | `python`, `3.12-slim` | the controller runs on a stock Python image |
| `deps.install`, `deps.packages`, `deps.indexUrl` | `true`, `[kubernetes==31.0.0]`, PyPI | installed into an emptyDir at start; set `install: false` on an image that has them |
| `fullnameOverride` | `<release>-model-swap` | e.g. `dynamoswap-controller` to keep the reload lab's object names |
| `nodeSelector`, `tolerations`, `resources`, `imagePullSecrets` | | the controller pod |
| `hostNetwork`, `healthPort` | `true`, `18080` | host network for the `/engine/` calls; the liveness port is then a host port |
| `pauseCapSeconds` | `60` | longest a pause may wait for running requests before an in-place reload gives up and bridges |

The CRD is in `crds/`. The Helm CLI installs it on the first install only and never upgrades or deletes it, so apply
a changed CRD yourself; ArgoCD applies it on every sync. The controller code (`controller/main.py`,
`controller/logic.py`, and `controller/fetch.py`, the gate Job's script) ships in a ConfigMap, and its hash rolls the
controller pod whenever the code changes. The CRD carries `helm.sh/resource-policy: keep` and Argo CD's
`Delete=false,Prune=false`: deleting it would delete every DynamoModelSwap and, through owner references, its slots.

## Lab results (B300 node-2, 2026-11-21)

Load during both runs: about 10 short requests/s in-cluster, 2 long streams (1,500-3,000 tokens), and 4 streams through
the serverlessgpu proxy `llm-swap-test`. Times run from the push of the one edit to `Serving`.

| Run | Edit | Push to Serving | Requests | Failed because of the swap |
|---|---|---|---|---|
| 1 | v15 → v20a, slots deleted whole | 22 min 53 s | 13,560 | 3, at the old slot's delete |
| 2 | v20a → v15, prefill first | 25 min 44 s | 15,381 | **0** |

Two models served together for only seconds. Requests slowed to 1.4-3.1 s (from 0.62 s) during the new full engine's
last 5-6 minutes of loading (CUDA graphs, autotune) on the same cards; the 0.27 bridge served the load at normal speed.
Sizing on 288 GB B300s, Gemma-4-31B NVFP4 TP4: an engine at util `u` holds about `u × 267.69 GiB + 3 GiB`, and KV
capacity is about 2.55M tokens per 1.0 of util (0.27: 0.58M, 0.70: 1.68M). 0.70 + 0.30 does not fit; 0.70 + 0.27 does.

## Limits

- One prefill and one decode component per DynamoModelSwap, one replica each.
- The bridge carries about a third of the full engine's KV cache while it serves alone (a few minutes per swap).
- A bridge swap takes about 25 minutes, mostly two engine loads. An in-place reload takes seconds, plus the gate.
- Editing the DynamoGraphDeployment itself does not swap: the CR is the edit surface. Making a DGD edit do this
  needs a change to the operator's rollout or a webhook.

## Using it from a GitOps repo (e.g. b300-bangkok)

The repo keeps only its own objects: an ArgoCD Application that installs this chart from the fork, the frontend graph,
the DynamoModelSwap, and any Service in front of the frontend.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: model-swap
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/Scicom-AI-Enterprise-Organization/dynamo.git
    targetRevision: feat/dynamo-model-swap      # a tag once released
    path: deploy/helm/charts/model-swap
    helm:
      values: |
        fullnameOverride: dynamoswap-controller
  destination:
    server: https://kubernetes.default.svc
    namespace: reload-lab
  syncPolicy:
    automated: {selfHeal: true, prune: true}
```

- The fork is about 670 MB. On ArgoCD 3.3 or later, a repository Secret with `depth: "1"` for this URL makes the repo
  server fetch only the target commit.
- Moving an install that already has the CRD and controller from plain manifests (the B300 reload lab) is its own step.
  Delete the old controller objects from the old Application first, but keep the CRD and the CRs: deleting the CRD
  deletes every DynamoModelSwap and, through owner references, its slots. Give the CRD
  `argocd.argoproj.io/sync-options: Prune=false` before the move.

## Tests

```bash
cd deploy/helm/charts/model-swap/controller
# The swap order against a simulated cluster. -c /dev/null skips the repo's pytest config (it needs pytest-benchmark).
uv run --with pytest pytest -q -c /dev/null --rootdir . test_logic.py
```
