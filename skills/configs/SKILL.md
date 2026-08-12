---
name: configs
description: How the prime-rl config system works — TOML files, CLI overrides, composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Configs

prime-rl uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — a Pydantic-based TOML + CLI config system (no tyro). Every entrypoint accepts TOML files via `@` and CLI overrides.

## Loading and composition

```bash
uv run rl @ examples/basic/reverse-text/rl.toml                                  # single TOML
uv run rl @ examples/basic/reverse-text/rl.toml --max-steps 50                   # CLI override
uv run rl @ base.toml @ overlay.toml                                       # left-to-right merge
uv run rl --model @ model.toml --data @ data.toml                          # nested section files
uv run rl @ base.toml --trainer @ trainer.toml --trainer.lr 1e-3           # mixed
```

Resolution order: CLI > config files (left-to-right) > class defaults. Merging is deep — unset fields in an overlay are preserved from the base.

Naming: CLI uses kebab-case (`--model.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect & validate

```bash
uv run rl --help                                  # all fields and defaults
uv run rl @ rl.toml --dry-run --output-dir /tmp/x # write resolved TOML to /tmp/x/configs
```

## Validators

Incompatible combinations (e.g. CP requires flash attention) must raise in a `model_validator` at resolve time, not at runtime. When renaming a field, remove the old spelling: no `validation_alias`, no auto-translating `mode="before"` validator. The old key then fails as an unknown key, which is the signal. An alias that stays forever is worse than a break — it never gets retired, and a key whose *meaning* changed silently misconfigures the run.

## Special syntax

**No inline tables** — checked-in configs use `[section]` headers, never `key = { ... }`. Expand `env.taskset = { id = "..." }` to a full-path header (`[orchestrator.train.source.env.taskset]` — subtable headers after a `[[...]]` entry attach to that entry).

**Booleans** — CLI `--flag` / `--no-flag`; TOML must be explicit (`enforce_eager = true`).

**None** — TOML has no null, use the string `"None"` (`max_model_len = "None"`); CLI: `--model.max-model-len None`.

**Lists** — TOML uses array of tables; later config files replace lists wholesale, so overlays must include the full desired list:

```toml
[[orchestrator.train.source]]
name = "reverse-text"

[orchestrator.train.source.env.taskset]
id = "reverse-text-v1"

[orchestrator.train.source.env.agent.harness]
id = "null"

[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[[orchestrator.eval.source]]
name = "reverse-text-eval"

[orchestrator.eval.source.env.taskset]
id = "reverse-text-v1"
split = "test"

[orchestrator.eval.source.env.agent.harness]
id = "null"

[orchestrator.eval.source.env.agent.runtime]
type = "subprocess"
```

CLI: `--orchestrator.train.source.0.env.taskset.id reverse-text-v1` or `--orchestrator.eval.source.0.env.taskset.id reverse-text-v1`.

**Dicts** — TOML uses a section; CLI takes a JSON string: `--vllm-extra '{"key1": "value1"}'`. This works for plain `dict` fields only — nested pydantic-model fields (e.g. `algo`) reject JSON strings; use dotted keys (`--orchestrator.algo.type max_rl`) or a TOML overlay file.

**Discriminated unions** — set the `type` field to pick the variant (`[orchestrator.algo] type = "max_rl"`). Omit `type` to keep the default variant.

**Algorithms** — `[orchestrator.algo] type = "grpo" | "max_rl" | "rae" | "hierarchical_grpo" | "opd" | "opsd" | "sft" | "echo"` — the type names the algorithm (credit assignment + loss routing, fused), and each type's class defaults are its vetted setting; any other key you set is your own assembly (e.g. `[orchestrator.algo.roles.user] alpha = 0.1` for echo — setting any echo role replaces the whole role table). `hierarchical_grpo` is only valid with a proposer-solver env: it compares solvers with attempts on the same proposed problem and proposers with other proposals in the group. There is no preset layer, and no config hook that points at user code — a new algorithm is a named class in the repo (subclass `Algorithm`, register it). Per-source override: `[orchestrator.train.source.algo] type = "opd"` (the source assembles its own algorithm). prime-rl only hosts the trainable policy; frozen models are inline external endpoints on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for opd (the frozen model scored against), `[orchestrator.algo.sampling.source]` for sft (the model it samples from), each with `name` + `base_url`. There is no shared `teacher` slot. opsd declares no model — it self-distills against the live policy. See `docs/algorithms.md`.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## RL trainer token exports

For rollout debugging, enable trainer-side token export with `trainer.enable_token_export = true` (or `--enable-token-export` when running the trainer entrypoint directly). It writes one JSONL record per exported sequence. Single-run/fallback exports go under `output_dir/token_exports/step_<step>/rank_<rank>.jsonl`; multi-run trainer exports with packer metadata go under the owning run directory, `output_dir/<run_id>/token_exports/step_<run_step>/rank_<rank>.jsonl`. Each record stores aligned per-token arrays for token ids, loss mask, component weight streams (rl/ce/ref_kl), advantages, entropy, mismatch KL, inference/trainer logprobs, importance ratios, probability deltas, and masking diagnostics. It does not decode token text in the trainer.

```toml
enable_token_export = true
```

Leave it unset for normal training. When enabled, it exports every sequence from each exporting rank.

## Experimental XOR weight transfer

The NIXL broadcaster has an opt-in GPU-resident nvCOMP LZ4 path for exact
bitwise XOR updates in BF16, FP16, or FP32:

```toml
[model]
name = "Qwen/Qwen3-0.6B-Base"

[trainer.model]
optimization_dtype = "bfloat16"

[inference.model]
dtype = "bfloat16"

[weight_broadcast]
type = "nixl"
delta_mode = "xor"
delta_adam_bucket_mb = 512
delta_pipeline_depth = 8
delta_cuda_graphs = true
```

Use `configs/debug/weight-sync/qwen3-8b-fsdp4-tp2-nixl-xor-smoke.toml` for a
dense NIXL smoke run and
`configs/debug/weight-sync/mini-glm-moe-fsdp4-ep4-tp2-nixl-xor-smoke.toml` for
the first MoE/EP smoke run.
Use `configs/debug/weight-sync/qwen3-8b-fsdp4-tp2-nixl-fp8-xor-smoke.toml`
for the producer-quantized FP8 kernel path on Hopper.
Use `configs/debug/weight-sync/qwen3-30b-a3b-fsdp4-ep4-tp4-nixl-fp8-xor-smoke.toml`
for Qwen3-MoE FP8 resident transfer with inference expert parallelism.
Use `configs/debug/weight-sync/qwen3-30b-a3b-fsdp4-ep4-tp4-nixl-xor-benchmark.toml`
for its eight-GPU BF16/source counterpart. For paired performance runs, keep
`delta_adam_bucket_mb = 512` and `delta_pipeline_depth = 8`, run the same config
with `delta_mode = "xor"` and `delta_mode = "none"`, and summarize the standard
`<scenario>/{full,xor}` output tree with
`uv run scripts/summarize_weight_sync_benchmarks.py <output-root>`. Trainer step
logs include `Policy Update`, which spans the optimizer update, source-delta
generation, FP8 resident conversion, and completed transfer. `Weight Sync`
times the broadcast call itself; for FP8 that includes resident conversion and
delta construction, while source-representation deltas have already been built
by the optimizer. The broadcaster's
`synchronized in` metric measures only the protocol portion after payload
construction. Exclude policy v1 from paired averages because it captures the
receiver CUDA graphs.
Run the dense smoke for at least four steps both as configured and with
`--weight-broadcast.delta-mode none`. The full-transfer baseline exercises the
same rendezvous with less producer work between updates, making it the more
sensitive check for consecutive-update handshake races.

Source-representation XOR requires a text-only model, AdamW, single-run training,
`dp_replicate=1`, `cp=1`, and no trainer or inference quantization. Trainer
`optimization_dtype` and inference `model.dtype` must be the same explicit
value: `bfloat16`, `float16`, or `float32`. Model loaders must route
weights through same-dtype, bit-preserving operations; the NIXL worker validates
the traced load graph before the first update. NIXL supports rank-aware TP/EP
pulls and multi-node deployment. NCCL remains available for normal full-weight
transfer but does not implement XOR updates. Fake-data runs skip weight
transfer and use standard AdamW even when delta mode is configured.

For FP8 rollout inference, select the FP8 resident-kernel transfer representation:

```toml
[inference.model]
name = "Qwen/Qwen3-8B-FP8"

[weight_broadcast]
type = "nixl"
delta_mode = "xor"
delta_representation = "fp8_kernel"
delta_fp8_scale_format = "float32"
```

This path uses AdamW with foreach operations scheduled in bounded
`delta_adam_bucket_mb` buckets, avoiding model-shard-sized optimizer temporaries
without recording redundant source-weight deltas. The initial update is loaded
from producer-built FP8 checkpoint tensors. For later updates, each trainer rank owns a subset of
layers and retains their uncompressed, TP-local vLLM resident tensors. It builds
the next resident version with the pinned vLLM packing routines and compresses
`bytes(R_old) XOR bytes(R_new)`, including packed Marlin `int32` weights and
scale tensors. Frames are tagged for one inference rank, which XORs them directly
into its resident parameters. The receiver does not retain a checkpoint shadow
or rerun layer post-processing. Resident FP8 XOR supports dense Qwen3 and
Qwen3-MoE. Qwen3-MoE requires inference expert parallelism, partitions experts
evenly across the inference world, and retains vLLM's fused `w13`/`w2` tensors
and block scales on their owning EP ranks. Other architectures require an
explicit converter for their vLLM fusion, TP/EP ownership, and selected kernel
layout. Native FP8 Hopper kernels and vLLM's Marlin FP8 weight-only fallback on
Ampere are supported. Trainer and inference GPUs must select
the same vLLM FP8 kernel backend; use homogeneous GPU architectures and matching
vLLM kernel-related environment settings. Inference DP is currently unsupported
for this representation. Keep `delta_fp8_scale_format = "float32"`; vLLM's
selected kernel adapter performs any deterministic resident scale conversion.
Set `delta_mode = "none"` with the same `fp8_kernel` representation for a
full-snapshot baseline that transfers the exact same weights and scales.
The inference checkpoint must contain the matching FP8 scheme and scale format;
online `inference.quantization` conversion is not accepted because its loader
starts from BF16 rather than producer-generated FP8 checkpoint tensors. Keep
`delta_representation = "source"` for unquantized inference.
The startup update is a normal full checkpoint transfer;
later consecutive versions are XOR updates in the selected representation. Once
all receivers and the orchestrator acknowledge the startup FP8 XOR policy, the
trainer deregisters and releases its checkpoint-format source tensors and full-sync
staging arenas. It retains only the distributed TP-local resident kernel
snapshot. The trainer logs the checkpoint and staging sizes plus allocated and
device-free CUDA memory on each rank. A later full recovery is rejected until those resources and the transfer
plan are explicitly rebuilt; `delta_mode = "none"` retains them for every update.
FP8-kernel transfer uses bucketed AdamW and one reusable full-transfer staging
buffer in both XOR and full baselines so changing the transfer protocol does not
change optimizer scheduling or startup memory. `delta_pipeline_depth` still
controls steady-state delta encoding.
For
source-representation XOR, AdamW snapshots and
updates bounded local parameter buckets, invokes one foreach-capable AdamW update per
bucket, and batch-compresses that bucket's parameter XOR tensors with nvCOMP LZ4
without leaving CUDA memory. The compressed tensor streams are packed into one
CUDA `uint8` payload per trainer rank. NIXL copies frames into reusable
registered arenas on their owning trainer ranks; rank zero publishes metadata
only, and each inference worker pulls only frames required by its traced TP/EP
routes. Receivers pull and decode the next bounded transfer group on a side
stream while applying the current group. Stable direct XOR route segments are
captured into receiver-side CUDA graphs during the first delta update and
replayed on later updates; transformed routes remain eager. The aggregate
receiver metric reports `graphs=<captures>/<replays>` and
`routes=<graphed>/<eager>`. The first delta is intentionally serialized while
graphs are captured, so exclude it from timing averages. Set
`--weight-broadcast.delta-cuda-graphs false` for a same-code eager XOR control
run. ModelExpress handles initial peer
discovery, policy metadata, and policy-level pause/update/resume coordination.
After discovery, per-group READY and ACK messages use generation-tagged NIXL
notifications instead of ModelExpress status polling. The generation includes
the policy step, group index, staging slot, and inference rank so delayed
notifications cannot authorize reuse of the wrong producer buffer.
If compression is not beneficial on any trainer rank, all ranks collectively
fall back to the full-transfer protocol for that policy version; the decision
must never be made independently because trainer ranks share the same
distributed transfer sequence.

For a single-node deployment with a loopback `weight_broadcast.host`, the `rl`
launcher starts an in-memory ModelExpress-compatible metadata server and owns
its lifecycle. Its log is `logs/model_express.log`. Multi-node deployments and
non-loopback hosts require an externally deployed ModelExpress metadata server.
Use `127.0.0.1`, rather than an IPv6 wildcard or `localhost`, for single-node
smoke configs because GPU containers may have IPv6 disabled. Use a dedicated
high coordinator port such as `18001`; port `8001` is the ModelExpress default
and may already be occupied by a service supplied by the runtime image.

Before model-level testing, exercise bidirectional NIXL notifications directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/integration/test_nixl_notifications.py -q -s
```

Before a long NIXL smoke run, verify that one installed CUDA-specific binding
actually exposes the UCX plugin:

```bash
uv sync --all-packages --extra disagg --extra flash-attn
uv run python -c 'from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent; NixlAgent("nixl-ucx-probe"); print("NIXL UCX ready")'
```

Importing a `nixl_cu12` or `nixl_cu13` Python module is not sufficient: wheels
can be present without a usable UCX plugin. The adapter probes the binding that
matches PyTorch's CUDA major first, then the other CUDA binding and the generic
NIXL module, and selects the first one that actually advertises UCX. The
`nixl-cu12==0.10.1` x86 wheel must come from PyPI; the smaller wheel formerly
hosted on the prime-rl v0.5.0 release is missing a usable packaged UCX backend.
If the probe fails after switching wheel sources, force replacement of the
same-version installed wheel with `uv sync --refresh-package nixl-cu12`.
NIXL 0.10.1's notification binding accepts text even though its Python wrapper
annotates messages as bytes, returns remote agent names as bytes, and mishandles
an explicitly selected backend. Keep notification encoding and agent-name
normalization inside `NixlAgent`; callers should exchange raw protocol bytes.
NIXL defaults `UCX_TLS` to `all`; keep that value in single-node smoke configs
so CUDA IPC/shared-memory or TCP transports remain available on hosts without
RDMA devices. An error listing unavailable `rc_x`, `rc`, `dc_x`, and `dc`
followed by `no active messages transport` means a restrictive inherited
`UCX_TLS` excluded those fallback transports.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs
