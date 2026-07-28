# Scicom tm-h20 deployment notes

What this branch changes versus upstream `ai-dynamo/dynamo@39c7bcb`, why, and how
to reproduce it. Everything here was measured on tm-h20 (2 × 8× H20-3e, Slurm,
`google/gemma-4-31B-it` FP8, 1P1D TP8 disagg with KV over RoCE).

## The two defects

### 1. gemma4 `<|channel>` delimiters leak into `content`

With `chat_template_kwargs={"enable_thinking": false}` on a turn that ends in a
tool response, the customer-visible reply came back as:

```
<|channel>thought
<channel|>You're all clear! You currently have no outstanding balance.
```

**100% reproducible** — 200/200 on the production endpoint.

`preprocessor.rs::is_reasoning_disabled_by_request` disabled the gemma4 reasoning
parser whenever `enable_thinking` was false, on the premise (stated in its own
comment) that the model then emits no `<|channel>` markers. That premise does not
hold for the canonical Gemma 4 template: `add_generation_prompt` only prefills a
closed thought channel when the previous message is neither a `tool_response` nor
a `tool_call`. On a tool-response turn the template emits nothing and the model
produces the delimiters itself — while `parser_requires_special_tokens()` has
already forced `skip_special_tokens=false`, so nothing strips them.

Upstream vLLM never gates its reasoning parser on `enable_thinking`;
`Gemma4ReasoningParser.extract_reasoning` returns `(None, model_output)` when the
markers are absent, so leaving it enabled is free. This branch matches that.

**Not a vLLM version issue.** Identical 200/200 on vLLM 0.23.0 and 0.24.0 — under
Dynamo, vLLM runs with `detokenize=false` and never sees the text at all.

### 2. Structured outputs silently ignored

`guided_json` / `guided_choice` / `response_format` were accepted with 200 OK and
had no effect, so a caller asking for raw JSON got a ```` ```json ```` fence and
`json.loads()` threw — driving a 4× retry storm in the agent.

This one was **path-dependent**, which is what made it confusing:

| | Rust preprocessor | `--dyn-chat-processor vllm` |
|---|---|---|
| `guided_choice=["ELEPHANT","GIRAFFE"]` on "What is 2+2?" | `ELEPHANT` | `2 + 2 = 4` |
| `guided_json` | valid JSON, unfenced | not JSON |

The Rust path reads `CommonExt`'s top-level `guided_json` / `guided_choice` /
`guided_regex` / `guided_grammar` and builds `GuidedDecodingOptions`. The vLLM
Python processor builds `sampling_options` itself from an explicit whitelist that
omitted structured outputs entirely — fixed here in `vllm_processor.py`, though
the Rust path is the one actually deployed.

**Send `guided_json` at the top level** (`extra_body` in the OpenAI SDK).
`response_format` is registered as an unsupported field in `openai.rs` and is
**untested** on this path — do not assume it works.

## Verified on job 300 (public endpoint, n=10/cell, temp=0.7 top_p=0.95)

| check | result |
|---|---|
| channel leak, `enable_thinking=false` / `true` / omitted | **0/10** each |
| `guided_json` — valid JSON + correct schema | **10/10** |
| `guided_json` — fenced despite prompt demanding a fence | **0/10** |
| baseline, no `guided_json`, same prompt | **still fenced** ← control |
| `guided_choice` conflicting control | `ELEPHANT` |
| tool calling | intact |

The baseline row is the important one: without the constraint the same prompt
still fences, so this is enforcement, not the model being agreeable that day.

## The fragile part

`lib/llm/src/preprocessor.rs` compiles into `_core.abi3.so`. The stock prebuilt
`.so` still carries the leak, so **a source refresh or a fresh node reverts it
silently** — nothing errors, the delimiters just come back.

Either install the wheel from the release, or run `build-bindings.sh`. On
tm-h20 the originals are kept at `/mnt/data/dynamo/_core.abi3.so.orig-39c7bcb`.

## Rebuilding

`build-bindings.sh` needs **no apt** — deliberately, because the DSW pods have a
wedged dpkg state (`libavdevice60` wants `libgl1`; `libgl1` cannot unpack because
the hand-injected NVIDIA driver userspace at
`/usr/lib/x86_64-linux-gnu/libGL.so.1.7.0` sits on a different mount device, so
dpkg cannot make its backup hardlink). Dependencies come from elsewhere:

| need | source |
|---|---|
| rustc **1.96.1** (per `rust-toolchain.toml`) | `rustup` into `$CARGO_HOME` |
| protoc | prebuilt binary from the protobuf GitHub release |
| libclang (for `nixl-sys` bindgen) | PyPI `libclang` package |
| `stdbool.h` | gcc-13's own include dir via `BINDGEN_EXTRA_CLANG_ARGS` |

`cargo check` ~1m30s, `maturin build --release` ~5m38s on 164 cores.

Install is a `mv`, not a `cp`: a running job has the old `.so` mmap'd and
overwriting it in place can SIGBUS those processes.

## Runtime pinning

`deploy/scicom/venv-vllm023.lock.txt` is the exact 240-package transitive set of
the validated venv. Note the vLLM downgrade fixed **neither** defect — it is kept
only because it is what was validated end to end. Whether 0.24.0 also enforces
guided decoding on the Rust path is untested; if you re-test it, the extra venv
can probably be dropped.
