# Scicom changelog: the Dynamo fork

This repository is Scicom's fork of [`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo). It exists for one reason: `google/gemma-4-31B-it`, served with Dynamo + vLLM, returned customer-visible text it should not have (the model's chain-of-thought, `<|channel>` markers), and upstream Dynamo did not fix it. Everything Scicom changed lives on its own branches; this file is the index.

- **`main` is not where Scicom code lives.** It is a mirror of upstream `main` taken on 2026-07-26 (`769ee07`, upstream PR #11973) plus this file. It is not kept in sync with upstream.
- **Each Scicom branch starts from an upstream release or commit** and carries a small stack of Scicom commits. Its `deploy/scicom/DEPLOYMENT_CHANGELOG.md` has the full detail: evidence, rebuild steps, and why each change exists.
- **The other ~2,800 branches in this fork** (people's branches, `pull-request/N`) are copies of upstream refs from when the fork was created. They are not Scicom work.

## At a glance

Newest first. "Live" means serving production or benchmark traffic on 2026-09-30.

| Branch / tag | Tip | Based on | Status | Runs on |
|---|---|---|---|---|
| [`feat/prom-metrics-for-bucketing`](#featprom-metrics-for-bucketing) | `85aff69` | `1.5.0-scicom-gemma4-toolturn-reasoning` | live (frontend) | TM B300 Bangkok k8s |
| [`1.5.0-scicom-gemma4-toolturn-reasoning`](#150-scicom-gemma4-toolturn-reasoning) | `3e45671` | upstream `v1.5.0` | live (workers); base for new work | TM B300 Bangkok k8s |
| [`v1.3.0-gemma4-toolturn-reasoning.1`](#v130-gemma4-toolturn-reasoning1) | `0288f8f` | `v1.3.0-gemma4-channel-leak.1` | live, frozen | tm-h20 Slurm fleet |
| [`v1.3.0-gemma4-channel-leak.1`](#v130-gemma4-channel-leak1-branch-tag-and-release) (branch) | `1f1a2f5` | upstream `39c7bcb` | superseded | tm-h20, 2026-07-28 to 07-31 |
| `v1.3.0-gemma4-channel-leak.1` (tag + GitHub release) | `ce28575` | same | release artifact | prebuilt `ai_dynamo_runtime` wheel |

All Scicom commits are by `ariffnazhan-scicom` (Ariff Nazhan).

## How we work in this fork

**Branch names.**
- **A new line of work:** `<upstream-version>-scicom-<desc>`, e.g. `1.5.0-scicom-gemma4-toolturn-reasoning`.
- **A feature on top of it:** `feat/<desc>`, branched off that line.
- **Older branches** use `v1.3.0-gemma4-<desc>.<n>` and keep those names.

**Never rewrite a branch that something runs from.** New work gets a new branch. Carrying fixes to a newer upstream is a `git cherry-pick -x` onto a branch cut from the upstream tag, so every ported commit names its original.

**Rust changes need a rebuilt binary.**
- The gemma4 fixes are in `lib/llm/src/preprocessor.rs`, which compiles into `dynamo/_core.abi3.so`. Refreshing only the Python source or installing a stock wheel **silently reverts them**. Nothing errors; the leak just comes back.
- For bare metal, rebuild with `deploy/scicom/build-bindings.sh`.
- For Kubernetes, build the whole image; see [Deployments](#deployments).

**Every deployed frontend stamps `x-time-deployment`.**
- It is set from the `TIME_DEPLOYMENT` env var, and is present on every branch from `1f1a2f5` on.
- If a response has no header, a stock binary answered.
- Check it before trusting any test result.

## The gemma4 defects this fork fixes

All three are in Dynamo's frontend (the preprocessor and response parsing), not in vLLM. All three fail silently: no error, only wrong output.

| # | Symptom | Cause | Upstream status (upstream main, checked 2026-09-30) |
|---|---|---|---|
| 1 | Thinking **off**, turn ends in a tool response: reply comes back as literal `<\|channel>thought\n<channel\|>You're all clear! …` (200/200) | The gemma4 reasoning parser is switched off when `enable_thinking` is not true, but special tokens are still kept, so nothing consumes the delimiters the model writes itself | **Not fixed, and the opposite is intended.** Upstream #13061 (1.4.x) made gemma4 reasoning opt-in and added a test asserting these markers stay in `content`. The fork reverses that. |
| 2 | `guided_json` / `guided_choice` / `response_format` ignored on the vLLM chat-processor path; the caller gets fenced JSON and retries 4x | `vllm_processor.py` built `sampling_options` from a whitelist that omitted structured outputs | **Fixed upstream** by #12684 (in 1.4.0). Dropped from the 1.5.0 branch. |
| 3 | Thinking **on**, turn after a tool response: the whole chain-of-thought is returned as `content` (0/200 correct) | Gemma 4's template leaves `<\|channel>thought\n` open at the end of the prompt, so the completion starts inside the channel. `prompt_injected_reasoning_start` did not recognise the gemma4 opener, and `Gemma4ReasoningParser` ignores `set_in_reasoning` | **Not fixed, not reported.** Still true in `dynamo-parsers` 8.1.0 and on upstream main. The fork primes the parser with the opener. |

## Branches in detail

### `feat/prom-metrics-for-bucketing`

**Frontend KPIs per input-sequence-length (ISL) bucket.** Off `1.5.0-scicom-gemma4-toolturn-reasoning`, 2026-09-30.

Upstream's frontend histograms (TTFT, ITL, request duration, OSL, cached tokens) are labelled by model only, and ISL is a separate histogram. Because they cannot be joined, "TTFT of 4k-token prompts vs 1k" is impossible to answer. This branch adds, opt-in with `DYN_METRICS_ISL_BUCKETS=512,1024,2048,4096,8192,16384`:

- `dynamo_frontend_time_to_first_token_by_isl_seconds`
- `dynamo_frontend_inter_token_latency_by_isl_seconds`
- `dynamo_frontend_request_duration_by_isl_seconds`
- `dynamo_frontend_output_sequence_tokens_by_isl`
- `dynamo_frontend_cached_tokens_by_isl`

They are labelled `model` and `isl_bucket` (`0-512`, `513-1024`, … `16385+`), and use the same histogram buckets as their unbucketed twins. When the variable is unset, nothing changes.

| Commit | What |
|---|---|
| `c21fa44` | The five families. Each request's bucket is fixed at its first token, where the ISL is known. `DYN_METRICS_ISL_BUCKETS` is declared in `lib/runtime` `environment_names.rs`. Unit tests and docs (metrics catalog, env vars) included. |
| `85aff69` | Creates every bucket series at zero when the model registers (`update_metrics_from_mdc`). A series born with its final value is invisible to `rate()` / `increase()`, so a burst finishing between two scrapes would never reach a dashboard. |

**Deployed** on the B300 frontend since 2026-09-30 (image `…:1.5.0-scicom-prom-metrics-for-bucketing-85aff69`). It feeds the Grafana dashboards `dynamo-metrics` and `dynamo-isl` on that cluster.

**Verified** with 128 dynamo-llm unit tests, 93 stream-parsing tests and 2 env-registry tests in the image build. Live, TTFT p95 went from 97 ms for 0-512 tokens to 958 ms for 8193-16384 tokens, with ITL flat.

**Limitation:** requests that fail before their first token have no bucket. They are counted only by `dynamo_frontend_requests_total`.

### `1.5.0-scicom-gemma4-toolturn-reasoning`

**The gemma4 fixes rebased onto upstream `v1.5.0`.** Created 2026-09-30 (vLLM 0.28, `dynamo-parsers` 8.1.0). Built so the B300 cluster, running Dynamo operator 1.5.0, can serve gemma with the fixes; NVFP4 also needs vLLM 0.28.

| Commit | From | What changed in the port |
|---|---|---|
| `1d92aee` | `1f1a2f5` | `x-time-deployment` header. Only a conflict with new code at the same spot. |
| `5b68611` | `96f170d` | Defect 1. The gemma4 arm of `is_reasoning_disabled_by_request` returns `false`. The Python gate moved into `_reasoning_parser_enabled` in `prepost.py`. **Upstream's #13061 tests are flipped:** `gemma4_without_enable_thinking_keeps_parser_markers_as_content` became `…_still_parses_channel_markers`, plus 4 `is_reasoning_disabled_by_request` cases and the default-thinking-mode test. |
| `ec90068` | `6545d13` | Defect 3a: gemma4 arm in `prompt_injected_reasoning_start`. |
| `362a598` | `c8893ec` | Defect 3b: parser priming, moved into 1.5.0's per-choice parser factory (upstream #11563), so every choice of an `n > 1` request is primed. |
| `64398fe` | `5876eaa` | End-to-end test of defect 3 (with a control that must still leak). |
| `7ff366d`, `43ec6d3`, `3e45671` | `ce28575`, `0288f8f` | `deploy/scicom/` docs and the rebase record. |

**Dropped:** `cefaa07` (defect 2, fixed upstream) and `c4c0c77` (the vLLM 0.23.0 pin).

**Deployed** as the B300 worker image `…:1.5.0-scicom-gemma4-toolturn-reasoning-3e45671` since 2026-09-30. Without this branch's frontend changes, the stock 1.5.0 frontend would have the leaks.

**Verified:**
- 77 gemma/reasoning unit tests and all 93 `postprocessor_parsing_stream` tests (they gate the image build).
- On `google/gemma-4-31B-it`, the fork's tool-turn studies passed 200/200.
- A 196-request compatibility matrix scored 195/196 (the miss is the `<bos>` issue below).

### `v1.3.0-gemma4-toolturn-reasoning.1`

**Defect 3** on top of `v1.3.0-gemma4-channel-leak.1`, 2026-07-30 to 07-31.

| Commit | What |
|---|---|
| `6545d13` | Recognise the gemma4 opener (`<\|channel>thought`) in `prompt_injected_reasoning_start`. |
| `c8893ec` | Prime `Gemma4ReasoningParser` by replaying the opener, because `set_in_reasoning` is a no-op for it. Either fix alone changes nothing. |
| `5876eaa` | End-to-end regression test with a leaking control. |
| `0288f8f` | `deploy/scicom/DEPLOYMENT_CHANGELOG.md`. |

**Evidence** against plain vLLM as the control, n=50 per cell. Reasoning-leakage, reasoning-template, tools-ablation and channel-leakage went from **0/200 correct before to 200/200 after**. Output matched vLLM within 1-3 characters. Reproduction lives in `ucc_ai_research` `stress-test/thinking-on-toolturn/`.

**Deployed:**
- **tm-h20** (Scicom's H20 Slurm fleet; the live gemma-4 endpoint). The source is at `c8893ec` in `/mnt/data/dynamo/src/dynamo`, as an editable install in `venv-vllm023` (vLLM 0.23.0) with a rebuilt `_core.abi3.so`.
- **B300 k8s**, 2026-09-30 05:24 to 07:04 UTC (image `…:1.3.0-toolturn-reasoning.1-0288f8f`).
  - On Dynamo operator 1.5.0 it needed `--scheduler-cls vllm.v1.core.sched.async_scheduler.AsyncScheduler`.
  - The reason: the operator always sets `DYN_FORWARDPASS_METRIC_PORT`. That makes this branch swap in its `InstrumentedScheduler`, which was written for vLLM 0.24, and the vLLM 0.23 engine died on the first request.

### `v1.3.0-gemma4-channel-leak.1` (branch, tag and release)

**Defects 1 and 2.** Upstream base `39c7bcb` (a main commit between v1.3.0 and v1.4.0), 2026-07-27 to 07-28.

| Commit | What |
|---|---|
| `96f170d` | Defect 1: keep the gemma4 reasoning parser enabled when thinking is off. Mirrored in `prepost.py`. |
| `c4c0c77` | Pin the tm-h20 stack to vLLM 0.23.0 and lock its dependencies (`deploy/scicom/venv-vllm023.lock.txt`). |
| `cefaa07` | Defect 2: forward structured outputs from the vLLM chat processor. |
| `ce28575` | `deploy/scicom/build-bindings.sh`: rebuild `_core.abi3.so` with no apt (rustup 1.96.1, prebuilt protoc, PyPI libclang). **Tag `v1.3.0-gemma4-channel-leak.1` points here.** |
| `1f1a2f5` | `x-time-deployment` header from `TIME_DEPLOYMENT` (after the tag). |

**GitHub release [`v1.3.0-gemma4-channel-leak.1`](https://github.com/Scicom-AI-Enterprise-Organization/dynamo/releases/tag/v1.3.0-gemma4-channel-leak.1)** ships a prebuilt `ai_dynamo_runtime-1.3.0-cp310-abi3-manylinux_2_39_x86_64.whl` (sha256 `a6beb933…3cf9b`) for tm-h20.
- Verified on tm-h20 job 300: 0/10 channel leaks for `enable_thinking` false, true and omitted, and `guided_json` 10/10 valid.
- Superseded on tm-h20 by `v1.3.0-gemma4-toolturn-reasoning.1`.

## Deployments

| Where | Branch / image | Built by | Updated by |
|---|---|---|---|
| **tm-h20** (Slurm, bare metal, Singapore) | `v1.3.0-gemma4-toolturn-reasoning.1` @ `c8893ec` | `deploy/scicom/build-bindings.sh`: editable install, `_core.abi3.so` swapped by rename (never overwrite a mapped `.so`) | Bump `TIME_DEPLOYMENT` in the sbatch, restart the job. Record changes in `ucc_slurm-ui-job` `jobs/tm-h20/README.md`. |
| **TM B300 Bangkok** (k3s, Dynamo operator 1.5.0), workers | `1.5.0-scicom-gemma4-toolturn-reasoning` @ `3e45671` | `AIES-Infra/b300-bangkok` app `dynamo-scicom-image`: a BuildKit Job on node-1 builds manylinux wheels over `vllm-runtime:1.5.0` and imports `localhost/scicom/dynamo-vllm-runtime:<tag>` into containerd | GitOps: new Job per commit (Rust tests gate the build), then the image tag in `argocd/apps/gemma4-31b-scicom`. `runtimeVersionOverride: "1.5.0"` is required, because the tag's semver sorts below 1.5.0. |
| **TM B300 Bangkok**, frontend | `feat/prom-metrics-for-bucketing` @ `85aff69` | same | same; `DYN_METRICS_ISL_BUCKETS` set on the frontend |

## Known issues (both branch lines, not fixed)

- **`/v1/completions` does not add gemma's `<bos>`.** A raw prompt degenerates (`" France is France is …"`); the same prompt with `<bos>` answers correctly. Chat completions are unaffected. This is an upstream Dynamo bug.
- **`guided_json` together with `enable_thinking=true` on gemma4.** The grammar forces JSON from the first token, so no reasoning happens, and about 10% of replies loop into invalid JSON (27/30 valid on 1.5.0, 10/12 on 1.3.0). Structured output with thinking off is 100%.

## Outside this repository

`ariffnazhan-scicom/dynamo` (a personal fork) has `v1.3.0-gemma4-mm.1`: a gemma-4 vision prefill/decode 2P4D launch example on base `39c7bcb` (2026-07-14). It carries no fix and was never merged here.
