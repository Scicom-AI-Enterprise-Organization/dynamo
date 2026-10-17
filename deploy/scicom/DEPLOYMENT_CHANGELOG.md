# Scicom deployment changelog — tm-h20

What this fork carries on top of upstream `ai-dynamo/dynamo@39c7bcb`, newest
first. Every entry is a change that reached the tm-h20 deployment; see
`README.md` in this directory for the rebuild procedure and the reasoning behind
the two original defects.

**All three defects regress silently** — no error, no log line, just wrong output.
The `x-time-deployment` header is the only outside signal of which build answered,
so keep `TIME_DEPLOYMENT` in the sbatch in step with the `.so` you actually built.

---

## Branch `1.6.0-scicom-gemma4-toolturn-reasoning`

The same fixes on upstream `main` at `f95c85b41` (2026-10-16, the commit behind the
`ai-dynamo` 1.6.0.dev20261016 nightly; there is no plain v1.6.0 tag yet), which pins
vLLM 0.31.0. The reason for the move is the NIXL patch below: it lets a gemma-4 prefill
and decode run with different TP sizes, and it applies to vLLM 0.31.0 but not to 0.28.0.
Each commit is a `cherry-pick -x` of its 1.5.0 commit.

| 1.5.0 commit | 1.6.0 commit | On 1.6.0 |
|---|---|---|
| `1d92aee09` `x-time-deployment` | `3ce890ebd4` | Unchanged. |
| `5b686115e` defect 1 | `3afdddf3d3` | Ported. Upstream still keeps gemma4 reasoning opt-in. The gate change applied as is; upstream's new `all_unified_families_honor_normalized_request_controls_for_all_selector_shapes` test and its gemma4 selector-shape test now assert that the parser stays on whatever `enable_thinking` says. |
| `ec9006825` defect 3 | `dcc809d7c9` | Upstream now arms the gemma4 parser itself when the prompt ends in `<\|channel>thought` or `<\|channel>`. Upstream's arm is kept, with the `gemma-4` alias added; our test cases are kept. |
| `362a598b7` defect 3 priming | `af092ed2fc` | Unchanged. |
| `64398fe2e` tests | `92dcecc85e` | Kept next to upstream's new tests at the end of `postprocessor_parsing_stream.rs`. |
| `7ff366d43`, `43ec6d3d0`, `3e4567125` build and docs | `8eff15d1ab`, `2b6a0c8e74`, `929e907e1f` | Unchanged. |
| (new) | `0780d7527d` | vLLM 0.31.0 NIXL patch for prefill/decode TP mismatch, see below. |

### Prefill and decode with different TP (gemma-4)

`patches/vllm-0.31.0-nixl-per-region-kv-heads.patch`, applied with
`apply-vllm-patches.sh <venv python>`; details in `patches/README.md`. Stock vLLM rejects a
prefill TP4 to decode TP8 handshake on gemma-4's first global layer (`SPLIT region 5`), because
the NIXL connector sizes every KV region with the model-wide 16 KV heads and the 10 global layers
have 4. Both engines need `--disable-hybrid-kv-cache-manager`, which costs KV capacity: the
sliding layers keep the whole context.

### Validated on tm-h20 (2026-10-17)

Built on 1021 with `build-bindings.sh`, installed into a test venv (vLLM 0.31.0 cu129, torch 2.13
cu129), test frontend on 1021 with its own etcd (:2399) and NATS (:4242), prefill TP4 on 1021,
decode TP8 on 1019. Results are filled in after the run.

---

## Branch `1.5.0-scicom-gemma4-toolturn-reasoning`

The same fixes rebased onto upstream tag `v1.5.0` (vLLM 0.28.0, `dynamo-parsers` 8.1.0),
for the k8s deployment on the TM B300 cluster, where the Dynamo operator is 1.5.0 and
NVFP4 needs vLLM 0.28. Each commit is a `cherry-pick -x` of its 1.3.0 original.

| 1.3.0 commit | On 1.5.0 |
|---|---|
| `96f170d87` defect 1 | Ported. Upstream #13061 (1.4.x) made gemma4 reasoning opt-in and added tests asserting the markers stay in `content` without `enable_thinking=true`; this branch reverses that and updates those tests. The Python gate moved into `_reasoning_parser_enabled` in `prepost.py`. |
| `6545d133b`, `c8893ec25` defect 3 | Ported. The priming moved into the per-choice parser factory (upstream #11563), so every choice of an `n > 1` request is primed. `Gemma4ReasoningParser` in `dynamo-parsers` 8.1.0 still does not override `set_in_reasoning`, and its streaming path still sends text before a stray `<channel|>` to `content`. |
| `5876eaae9` tests | Ported unchanged. |
| `1f1a2f5ad` `x-time-deployment` | Ported unchanged. |
| `cefaa07d5` defect 2 | **Dropped**: fixed upstream by #12684 (in 1.4.0), which forwards `guided_decoding` from `prepost.py`. |
| `c4c0c777b` vLLM 0.23.0 pin | **Dropped**: 1.5.0 runs on vLLM 0.28. |

With this branch the vLLM worker no longer needs `--scheduler-cls`: the 1.3.0 branch's
`InstrumentedScheduler` was written for vLLM 0.24 and crashed vLLM 0.23 whenever
`DYN_FORWARDPASS_METRIC_PORT` was set (the operator always sets it); 1.5.0's matches vLLM 0.28.

Image for Kubernetes: `AIES-Infra/b300-bangkok` `argocd/apps/dynamo-scicom-image` builds the
wheels in manylinux_2_28 and installs them over `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>`.

---

## Branch `v1.3.0-gemma4-toolturn-reasoning.1`

Based on `v1.3.0-gemma4-channel-leak.1`.

### Defect 3 — reasoning leaks into `content` on tool-response turns (thinking ON)

**Commits:** `6545d133b`, `c8893ec25`, `5876eaae9` (2026-07-30 → 07-31)
**Files:** `lib/llm/src/preprocessor.rs`, `lib/llm/tests/postprocessor_parsing_stream.rs`

With `chat_template_kwargs.enable_thinking=true`, on any turn whose previous
message is a tool response, the model's entire chain-of-thought was returned as
customer-visible `content`.

Gemma 4's template appends an **unclosed** thought channel in that case:

```jinja
{%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}
    {{- '<|channel>thought\n' -}}
```

so the completion begins *inside* the channel and emits no opening marker of its
own — it reasons, closes with `<channel|>`, then answers.

Two independent things had to be fixed; **either one alone is a no-op**:

1. `prompt_injected_reasoning_start` only recognised `<think>` / `<mm:think>`, so
   gemma4 always returned `false` and the parser was never armed.
2. `ReasoningParser::set_in_reasoning` has a **default empty body** and
   `Gemma4ReasoningParser` (dynamo-parsers 3.1.0) never overrides it — it
   implements only `detect_and_parse_reasoning`,
   `parse_reasoning_streaming_incremental` and `finish_reasoning_stream`. Arming
   the flag therefore changed nothing. The parser is now **primed** by replaying
   `<|channel>thought\n` into it, which makes it match `START_TOKEN` at offset 0,
   enter the span and consume the `thought\n` role label.

Fixing only (1) produces a build that tests **identical to unpatched**. That was
confirmed with a second frontend on port 8001 running
`DYN_LOG=info,dynamo_llm::preprocessor=trace`, which logged
`prompt_injected_reasoning=true` beside a response whose `reasoning_content` was
still empty.

**Evidence** — plain vLLM as control (same model, box and payloads; vLLM
`--reasoning-parser gemma4` vs Dynamo `--dyn-reasoning-parser gemma4`), n=50/cell:

| tool-turn study | vLLM | Dynamo before | Dynamo after |
|---|---|---|---|
| reasoning-leakage | 50/50 · 140 ch | **0/50** · 438 | **50/50** · 140 |
| reasoning-template | 50/50 · 152 ch | **0/50** · 2002 | **50/50** · 153 |
| tools-ablation | 50/50 · 139 ch | **0/50** · 488 | **50/50** · 137 |
| channel-leakage | 50/50 · 142 ch | **0/50** · 453 | **50/50** · 139 |
| **total** | **200/200** | **0/200** | **200/200** |

Two non-tool studies (`empty-response`, `long-context`) are 50/50 on every build,
so nothing regressed. Post-fix output matches vLLM within 1–3 characters.

**Regression test:** `gemma4_prompt_injected_reasoning_routes_leading_text_to_reasoning`
in `lib/llm/tests/postprocessor_parsing_stream.rs`. It asserts the fixed path *and*
a control with `prompt_injected_reasoning=false` that must still leak — without the
control it would pass just as happily if the priming stopped running.

**Scope:** the priming is gated on the gemma4 parser name; no other model family
is affected. No crate vendoring, no `[patch.crates-io]`.

---

## Branch `v1.3.0-gemma4-channel-leak.1`

### Defect 2 — structured outputs silently ignored

**Commit:** `cefaa07d5` (2026-07-28)
**Files:** `components/src/dynamo/frontend/vllm_processor.py`, `prepost.py`

`guided_json` / `guided_choice` / `response_format` were accepted with 200 OK and
had no effect, so a caller asking for raw JSON got a ```` ```json ```` fence and
`json.loads()` threw — driving a 4× retry storm in the agent.

Path-dependent: the Rust preprocessor reads `CommonExt`'s top-level
`guided_json` / `guided_choice` / `guided_regex` / `guided_grammar`; the vLLM
Python processor built `sampling_options` from a whitelist that omitted them
entirely. Fixed in `vllm_processor.py`, though the Rust path is the one deployed.

**Send `guided_json` at the top level** (`extra_body` in the OpenAI SDK).
`response_format` is registered as unsupported in `openai.rs` and is **untested**.

### Defect 1 — gemma4 `<|channel>` delimiters leak into `content` (thinking OFF)

**Commit:** `96f170d87` (2026-07-27)
**File:** `lib/llm/src/preprocessor.rs`

With `enable_thinking=false` on a turn ending in a tool response, the reply came
back as literal `<|channel>thought\n<channel|>You're all clear! …` — **200/200
reproducible.**

`is_reasoning_disabled_by_request` disabled the gemma4 parser whenever
`enable_thinking` was false, on the premise that the model then emits no
`<|channel>` markers. That premise does not hold: on a tool-response turn the
template emits nothing and the model produces the delimiters itself, while
`parser_requires_special_tokens()` has already forced `skip_special_tokens=false`,
so nothing strips them. The gemma4 arm is now exempt from the gate.

Not a vLLM version issue — identical on 0.23.0 and 0.24.0; under Dynamo vLLM runs
`detokenize=false` and never sees the text.

### Infrastructure

| Commit | Date | What |
|---|---|---|
| `1f1a2f5ad` | 2026-07-28 | `x-time-deployment` response header, stamped from the `TIME_DEPLOYMENT` env var. Absent header = stock binary. |
| `ce28575bf` | 2026-07-28 | `deploy/scicom/README.md` + `build-bindings.sh` (rustup 1.96.1 + protoc + PyPI libclang, no apt). |
| `c4c0c777b` | 2026-07-27 | Pin the tm-h20 stack to vLLM 0.23.0, lock transitives (`venv-vllm023.lock.txt`). |

---

## Deploying a rebuild

`lib/llm/src/preprocessor.rs` compiles into `_core.abi3.so`, consumed through an
**editable** install (`ai_dynamo_runtime.pth` → `lib/bindings/python/src`). A
source refresh or a fresh node silently reverts every fix above.

```bash
bash /root/build-bindings.sh                       # ~2m40s -> /root/dyn-wheel/*.whl
# archive the current .so, then install the one from the wheel into
#   /mnt/data/dynamo/src/dynamo/lib/bindings/python/src/dynamo/_core.abi3.so
# bump TIME_DEPLOYMENT in the sbatch, then restart the job
```

The running frontend holds the old binary in memory — **the fix only goes live on
job restart.** Archived builds live in `/mnt/data/dynamo/so-archive/`.

Submit as the owning user; root has no Slurm association for the account:

```bash
su - ariff_asri -c "cd /tmp && sbatch /tmp/<script>.sh"
```

## Verifying a deployment

Assert the invariant, not the symptom:

> **thinking requested ⇒ `reasoning_content` populated**

A `<|channel>` marker grep is **blind to defect 3** — it emits no markers in
`content` — and a deliberation-phrase regex scored 5/100 on a 100/100 failure.
Note plain vLLM returns this field as **`reasoning`**, not `reasoning_content`; a
checker that looks only for the latter reads a healthy vLLM response as broken.

Also check `err` before celebrating any zero: a network drop mid-run yields
`leak 0/N` because there were no responses at all.

Reproduction scripts and raw results:
`ucc_ai_research` → `stress-test/thinking-on-toolturn/`.
