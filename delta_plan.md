# NCCL XOR-Delta Weight Synchronization Plan

## Implementation status

The current experimental bring-up implements a versioned GPU nvCOMP LZ4
`bf16_xor` NCCL path for `Qwen/Qwen3-0.6B-Base`, AdamW, BF16 trainer state,
unquantized vLLM, and inference TP=1. A normal startup broadcast establishes
the full baseline. Later consecutive steps use delta-aware AdamW to snapshot
and update bounded 256 MiB local-parameter buckets, emit each parameter's exact
source-layout XOR, and batch-compress the XOR tensors directly on GPU. The
compressed CUDA streams are packed into one NCCL payload. The receiver keeps
that payload on GPU and batch-decompresses bounded Qwen layer groups directly
into CUDA source tensors before routing and in-place XOR application.
The receiver decompresses the frames, routes their tensors layerwise through
vLLM's loader, and applies them in place.
The debug smoke config additionally enables an end-to-end SHA-256 audit for a
small set of direct-layout Qwen parameters: the trainer hashes their exact
post-Adam bytes and the receiver verifies the live vLLM bytes after the full
compress/NCCL/decompress/route/XOR path.

This is an experimental bounded-snapshot implementation whose GPU codec,
compressed-payload retention, and layer decode scratch must be benchmarked
before this mode can make a performance claim.

## 1. Objective

Add an opt-in NCCL weight synchronization mode that sends compressed XOR deltas between consecutive policy versions instead of always broadcasting full absolute tensors.

The intended steady-state operation is:

```text
training weight version N
        |
        | optimizer update
        v
training weight version N+1
        |
        | generate exact XOR delta against version N
        v
block encoding and compression
        |
        | NCCL broadcast of large byte frames
        v
vLLM verifies that its baseline is version N
        |
        | decompress and apply XOR
        v
vLLM resident weight version N+1
```

The performance hypothesis is:

```text
delta generation + compression + compressed transfer + decompression/application
<
full tensor preparation + full transfer + normal vLLM reload
```

The project must measure every term in that inequality. Compression ratio alone is not an adequate success criterion.

## 2. Scope and non-goals

### In scope

- Prime-RL's NCCL weight broadcast path.
- Exact, lossless XOR deltas over tensor byte representations.
- Block-level zero detection and independently decodable blocks.
- Versioned full-sync and delta-sync messages.
- A bit-preserving BF16 source-delta path for explicitly audited model configurations.
- A direct vLLM-resident update path for explicitly supported model and inference formats.
- Bounded temporary GPU memory.
- Large buffered NCCL messages rather than per-block collectives.
- Full-sync fallback when a delta is invalid or uneconomical.
- Correctness and performance instrumentation.
- A later AdamW-specific path that emits deltas during parameter updates.

### Explicitly out of scope

- Any modification to NIXL.
- Making NIXL understand the delta protocol.
- A supposedly optimizer-generic wrapper that changes an arbitrary optimizer's `param_groups` and invokes `step()` multiple times.
- Lossy deltas.
- Applying deltas while vLLM is serving requests.
- LoRA support in the first implementation.
- Claiming generic direct-kernel support for models that do not implement vLLM-kernel export.
- Replacing the existing NCCL checkpoint broadcast. Delta synchronization remains opt-in until it has extensive correctness and performance coverage.

### First-version support contract

Keep the first end-to-end implementation intentionally narrow:

- Optimizer: Prime-RL's default PyTorch AdamW semantics, implemented through an AdamW-specific delta-recording wrapper/subclass.
- Optimizer CPU offload: unsupported.
- Training/wire weights: BF16 only.
- Inference `model.dtype`: explicitly `bfloat16`, not `auto`.
- Inference `quantization`: `None`.
- `quantize_in_weight_transfer`: `false`.
- Inference topology: TP=1 and PP=1 initially; DP replicas may receive the same update.
- Trainer topology: one trainer rank for the first end-to-end proof; sharded DTensor delta aggregation is the next expansion.
- Model: one audited dense architecture before adding MoE or model variants.
- Synchronization cadence: one published delta per optimizer step.
- Delta storage: GPU by default, under a strict configurable byte budget.
- Fallback: use the existing full NCCL broadcast if the encoded delta is too large, the buffer budget is exceeded, or any capability check fails before mutation.

Unsupported configurations must fail validation or select the existing full NCCL path. They must never enter a best-effort delta path.

## 3. Existing behavior that must remain intact

The current default is NCCL checkpoint-format transfer:

```text
trainer state dict
  -> layerwise Prime-to-HF conversion
  -> full tensors grouped and concatenated by dtype
  -> NCCL broadcast
  -> vLLM model.load_weights(...)
  -> TP/PP/EP slicing, fusion, casting, and processing
  -> copy into original resident kernel storage
```

The existing functions remain the control implementation:

- `src/prime_rl/trainer/rl/broadcast/nccl.py::broadcast_weights`
- `src/prime_rl/trainer/rl/broadcast/nccl.py::broadcast_state_dict`
- `src/prime_rl/inference/vllm/worker/nccl.py::receive_state_dict`
- `src/prime_rl/inference/vllm/worker/weight_transfer.py::load_weights_checkpoint_layerwise`

No delta change should alter their on-wire format or runtime behavior.

Prime already pauses all inference engines, drains in-flight work, performs a collective update, and resumes the engines. The delta path must use exactly the same lifecycle. It must not introduce a second pause/resume mechanism.

## 4. Core architecture: produce the delta during optimization and transform the delta at the receiver

The implementation must not require a persistent previous-model copy on either the trainer or inference worker. A CPU shadow model is permitted only in a short-lived measurement harness, not in the production architecture.

### 4.1 Trainer: capture old-to-new XOR while the old bytes still exist

The only natural place where both exact versions exist without retaining a previous model is the optimizer update itself:

```text
optimizer reads old parameter bytes
  -> computes and stores new parameter
  -> emits old_bits XOR new_bits
  -> compresses or queues that delta
```

The first correct implementation may snapshot one bounded parameter bucket immediately before updating that same bucket, then XOR/compress it immediately afterward. The eventual implementation should fuse delta emission into the AdamW update kernel.

The sender retains compressed update frames, not a previous model. This means optimizer integration is a core dependency of the real architecture rather than a late optional optimization.

### 4.2 Wire representation: training/checkpoint-layout XOR in the serving dtype

Initially target unquantized BF16 serving where Prime-to-HF conversion and vLLM loading consist only of bit-preserving routing operations:

- Rename.
- Slice/narrow/select.
- Reshape/view.
- Transpose/permute/contiguous copy.
- Split/chunk.
- Concatenation into fused QKV or gate/up destinations.
- TP/PP/EP selection.
- Same-dtype `copy_`.

For a byte rearrangement or selection `T`:

```text
T(old XOR new) = T(old) XOR T(new)
```

Therefore the trainer can emit deltas in its canonical wire layout and the receiver can route those delta bytes into the corresponding vLLM destination ranges without reconstructing the absolute checkpoint tensor.

This is valid only when every operation is bit preserving. Floating-point arithmetic, dtype conversion, quantization, scale generation, clamping, and value-dependent repacking invalidate this identity and must be rejected in the first mode.

### 4.3 Receiver: delta-aware vLLM loading with bounded scratch, not an old checkpoint

The receiver already owns the previous value in resident vLLM parameter storage. It needs a way to apply vLLM's existing name routing, fusion, and TP slicing to the incoming delta without letting the normal loader overwrite live weights.

The proposed receiver works one layer at a time:

1. Allocate zeroed scratch destination tensors for that vLLM layer, with the same shapes, dtypes, and weight-loader attributes as the resident parameters.
2. Temporarily route the layer's incoming delta tensors through the existing model-specific `load_weights()` and parameter `weight_loader` logic into scratch.
3. Restore the live parameter objects without changing their storage addresses.
4. XOR the populated scratch destination ranges into the live parameter bytes.
5. Release the layer scratch before processing the next layer.

Conceptually:

```python
destination_delta = route_with_existing_vllm_loaders(source_delta, layer_scratch)
live_parameter.view(integer_dtype).bitwise_xor_(destination_delta.view(integer_dtype))
```

This requires at most one layer or bounded destination bucket of scratch memory. It requires no previous checkpoint representation on the receiver and no second resident model.

Before implementation, build a compatibility audit that records every operation performed by the supported Prime conversion chain and vLLM weight loaders. The mode must fail closed if any non-bit-preserving operation occurs.

### 4.4 `convert_layer_to_vllm_kernel` is not the foundation

`PreTrainedModelPrimeRL.convert_layer_to_vllm_kernel()` is optional and raises `NotImplementedError` by default. The current concrete GLM-MoE-DSA exporter is hand-written: it concatenates particular projections and optionally performs blockwise FP8 quantization and architecture-specific postprocessing.

It does not provide a generic mapping from arbitrary Prime training tensors to arbitrary vLLM resident storage. It also does not by itself solve topology-specific TP/EP ownership for every model. Therefore the main BF16 delta design must not depend on it.

Kernel-format delta remains a later, explicitly model-specific path for quantized serving. Such a path must produce both old and new quantized blocks while the optimizer's bounded old-value snapshot is available, then XOR the final quantized weights, scales, and metadata. It cannot obtain a correct quantized delta by transforming a BF16 XOR.

### 4.5 Measurement-only shadow baseline

A temporary tool may retain one previous exported model on the trainer to measure XOR entropy before optimizer work is implemented. It must be labeled a benchmark harness, must not require a receiver-side shadow model, and must not become the architecture used by production synchronization.

## 5. Configuration design

Keep `type = "nccl"`. Add an explicit transfer mode rather than creating a new top-level transport type.

Proposed user-facing shape:

```toml
[weight_broadcast]
type = "nccl"
transfer_format = "checkpoint"  # existing default

[weight_broadcast.delta]
enabled = false
codec = "xor_blocks"
block_bytes = 131072
frame_bytes = 33554432
full_sync_threshold = 0.90
```

For early development, a simpler flat schema is acceptable if nested discriminated configuration is cumbersome:

```toml
[weight_broadcast]
type = "nccl"
transfer_format = "bf16_delta"
delta_block_bytes = 131072
delta_frame_bytes = 33554432
delta_full_sync_threshold = 0.90
```

Required semantic modes:

- `checkpoint`: current behavior and default.
- `bf16_delta`: optimizer-produced BF16 source deltas routed into resident vLLM destinations with a bit-preserving delta loader.
- `kernel_delta`: later model-specific final-kernel delta mode, permitted only for explicitly supported models/configurations.

Do not overload `quantize_in_weight_transfer` to mean delta mode. Quantization format and delta encoding are separate concerns.

Configuration changes will need to propagate through:

- `packages/prime-rl-configs/src/prime_rl/configs/rl.py`
- `packages/prime-rl-configs/src/prime_rl/configs/trainer.py`
- `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py`
- `packages/prime-rl-configs/src/prime_rl/configs/inference.py`
- `src/prime_rl/utils/client.py::init_nccl_broadcast`
- `src/prime_rl/inference/vllm/server.py::init_broadcaster`
- `src/prime_rl/inference/vllm/worker/nccl.py::init_broadcaster`

Validation must fail early for unsupported combinations, including:

- `kernel_delta` without model kernel-export support.
- A codec unavailable on the current platform.
- Invalid block or frame sizes.
- LoRA with an in-memory delta mode until explicitly supported.
- Quantized kernel mode whose scales or auxiliary tensors are not included in the exported representation.

## 6. Code organization

Keep the existing NCCL implementation readable by putting new functionality in separate modules.

Proposed files:

```text
src/prime_rl/weight_transfer/delta_protocol.py
    Shared protocol enums, dataclasses, serialization, validation, and checksums.

src/prime_rl/weight_transfer/delta_codec.py
    Reference XOR block encoder/decoder interface and CPU implementation.

src/prime_rl/trainer/rl/broadcast/nccl_delta.py
    Delta sender, baseline management, layer export, framing, and NCCL send.

src/prime_rl/inference/vllm/worker/nccl_delta.py
    Delta receiver, baseline verification, scratch routing/direct application,
    acknowledgement state, and metrics.
```

The existing `NCCLWeightBroadcast` may dispatch between separate methods:

```python
def broadcast_weights(self, model, step):
    if self.config.transfer_format == "checkpoint":
        return self.broadcast_weights_checkpoint(model, step)
    return self.broadcast_weights_delta(model, step)
```

The user-visible worker extension can remain `NCCLWeightUpdateWorker` because the server currently chooses the extension by transport type. Its update method should dispatch into the separate receiver implementation rather than mixing the protocols in one generator.

## 7. Versioned wire protocol

### 7.1 Required update header

Every update begins with a small fixed or length-prefixed header:

```python
DeltaUpdateHeader(
    magic=b"PRDL",
    protocol_version=1,
    update_kind=FULL | DELTA,
    representation=CHECKPOINT | VLLM_KERNEL,
    base_model_version=N,
    target_model_version=N + 1,
    frame_count=K,
    tensor_table_hash=...,
    update_checksum=...,
)
```

Rules:

- A delta is legal only when the receiver's current version equals `base_model_version`.
- `target_model_version` must be newer than the baseline and normally exactly one greater.
- A full update establishes a new baseline and can recover from any version mismatch.
- The tensor table hash must match the names, dtypes, shapes, and stable tensor IDs established for the baseline.
- The receiver must not change its visible model version until every frame has been validated and applied successfully.

### 7.2 Tensor identities

The initial full synchronization establishes stable integer tensor IDs:

```python
TensorDescriptor(
    tensor_id=17,
    name="model.layers.3.self_attn.qkv_proj.weight",
    dtype="bfloat16",
    shape=(...),
    byte_length=...,
)
```

Delta frames use tensor IDs rather than repeatedly serializing names.

For kernel-space deltas, a descriptor must identify the actual destination parameter and representation. Model names alone are insufficient if aliases or shared storage exist. The baseline builder must detect duplicate storage and ensure each byte range is updated exactly once.

### 7.3 Frame layout

Never launch one NCCL collective per block. Encode many independently decodable blocks into a large frame:

```text
FrameHeader
BlockDescriptor[block_count]
CompressedPayloadBytes
```

Each block descriptor contains at least:

```python
BlockDescriptor(
    tensor_id,
    byte_offset,
    uncompressed_bytes,
    compressed_offset,
    compressed_bytes,
    encoding,
    checksum,
)
```

Initial values to benchmark, not hard-code as universal constants:

- Compression block: 64 KiB, 128 KiB, or 256 KiB.
- NCCL frame: 16 MiB, 32 MiB, or 64 MiB.
- Receive staging buffers: one for correctness first, two for pipelined receive/apply later.

### 7.4 Encodings

At minimum support:

- `ZERO`: the entire XOR block is zero; no payload.
- `RAW_XOR`: uncompressed XOR bytes; useful when compression expands data.
- `COMPRESSED_XOR`: losslessly compressed XOR bytes.
- `ABSOLUTE`: absolute block bytes for recovery or mixed full/delta frames if later useful.

The encoder chooses the smallest valid representation per block after accounting for descriptor overhead.

### 7.5 Checksums and failure behavior

Use checksums to detect corrupt metadata and payloads. On failure:

- Do not acknowledge the target version.
- Mark the delta baseline invalid.
- Resume only after the orchestration layer has a clear failure or full-resync path.
- Never retry the same non-idempotent XOR application blindly; applying an XOR twice restores the old bytes.

For the first implementation, stage/reconstruct enough of an update that validation can happen before mutating visible resident weights. The direct streaming path will later require a transactional strategy, such as complete-frame checks before frame application plus a full-resync requirement on mid-update failure.

## 8. Codec design

### 8.1 Reference codec

Implement a simple CPU reference first:

```python
def encode_xor_blocks(previous: Tensor, current: Tensor, block_bytes: int): ...
def decode_xor_blocks(previous: Tensor, encoded: EncodedTensor): ...
```

Its purposes are:

- Define exact byte-level semantics.
- Generate golden test vectors.
- Measure raw XOR entropy and zero-block rates.
- Validate any GPU/AState implementation.

The reference implementation must treat tensors as raw bytes without floating-point arithmetic.

### 8.2 GPU codec

The GPU encoder should combine as much work as practical in one pass:

```text
load old and current words
  -> XOR
  -> block-wide all-zero reduction
  -> compression analysis/encoding
  -> emit descriptor and payload
```

Avoid separate full-tensor compare, dirty-marking, XOR, and compression passes unless profiling shows that separate kernels are faster.

The decoder should eventually fuse decompression with application:

```text
compressed block -> decoded word -> resident word XOR decoded word
```

This avoids allocating a full decompressed delta and avoids a second model-sized memory pass.

The codec API must report:

- Uncompressed bytes.
- Encoded bytes including descriptors.
- Zero block count.
- Raw fallback block count.
- Encode/decode time.
- Workspace bytes.

### 8.3 Compression fallback

For each whole update calculate:

```python
encoded_wire_bytes = header_bytes + descriptor_bytes + payload_bytes
full_wire_bytes = sum(tensor_bytes)
```

If:

```python
encoded_wire_bytes >= full_wire_bytes * full_sync_threshold
```

send a full synchronization instead. A starting threshold of `0.90` is reasonable for experiments, but must be configurable and benchmark-driven.

## 9. Baseline and memory model

### 9.1 Trainer

The production sender must not retain an absolute previous model. During one logical optimizer step it holds only:

- A bounded old-value bucket, until that bucket has been updated and encoded.
- Compression workspace.
- Encoded update frames waiting for the next policy synchronization.

After a bucket's delta has been encoded, release its old-value snapshot. The eventual fused AdamW path removes even that snapshot.

### 9.2 Inference worker

The previous baseline is the live resident vLLM model itself. The receiver holds only:

- Compressed NCCL receive frames.
- Decoder workspace.
- A zeroed scratch destination layer or bounded bucket used to run bit-preserving vLLM routing.

It does not retain previous HF/checkpoint tensors and does not retain a second vLLM model.

### 9.3 Version state

Both sides retain small metadata state:

- Current committed model version.
- Tensor table/schema hash.
- Pending encoded update version.
- Whether a full synchronization is required after failure.

An initial absolute NCCL update establishes the resident baseline. Every later XOR update names that exact baseline version.

## 10. BF16 source-delta sender and receiver

### 10.1 Sender

Add `broadcast_weights_delta()` as a distinct NCCL implementation, but consume deltas already produced by the optimizer rather than comparing two absolute state dicts.

For each optimizer bucket:

1. Snapshot only that bucket's old BF16 wire representation, or retain the old parameter values needed to produce it.
2. Apply the AdamW update for that bucket as part of one logical optimizer step.
3. Produce the exact new BF16 wire representation.
4. XOR old and new bytes.
5. Encode XOR blocks and release the old bucket.
6. Store encoded frames in bounded/pinned host buffers until synchronization.

At synchronization:

1. Pause inference through the existing orchestration path.
2. Broadcast the versioned header and encoded frames.
3. Wait for successful collective completion.
4. Release committed encoded frames.

The mapping from optimizer parameters to canonical wire tensors must handle fused Prime training parameters explicitly. If producing the HF wire representation itself requires multiple parameters, define optimizer buckets along those conversion boundaries so all required old and new inputs coexist only for that bounded conversion group.

### 10.2 Receiver delta loader

For each layer or bounded conversion group:

1. Receive and validate compressed delta frames.
2. Decode source-layout XOR tensors without reconstructing absolute source weights.
3. Create zeroed scratch parameters matching that vLLM destination layer.
4. Copy all custom `weight_loader` attributes required by vLLM onto scratch parameters.
5. Temporarily expose scratch to the model-specific loader for only the incoming layer.
6. Run Prime-to-HF conversion and vLLM name routing using only audited bit-preserving operations.
7. Restore live parameter objects.
8. XOR scratch bytes into the corresponding live parameter bytes.
9. Release scratch before continuing.

Do not call normal layerwise quantization or repacking on XOR values. The BF16 mode is valid only when the destination representation is already unquantized and the audited loader path is purely bit preserving.

### 10.3 Compatibility audit

Before enabling a model, test its conversion and loader path with adversarial byte patterns, including BF16 bit patterns representing NaNs. Compare:

```text
route(old) XOR route(new)
```

against:

```text
route(old XOR new)
```

for every destination byte. This catches implicit casts, arithmetic, padding behavior, and non-bit-preserving processing.

The first implementation supports a small audited model matrix. It is not automatically generic merely because it invokes `model.load_weights()`.

## 11. Later quantized/kernel-space sender and receiver

Quantized serving is a separate later problem. BF16 XOR cannot be passed through FP8 quantization because quantization and scale generation are nonlinear.

### 11.1 Supported-format contract

Introduce an explicit capability check, for example:

```python
model.supports_vllm_kernel_delta(vllm_config) -> bool
```

The capability contract must specify:

- Parameter names and stable ordering.
- Shapes and dtypes for each inference topology.
- TP and EP ownership rules.
- Quantized weight and scale tensors.
- Padding bytes that must be deterministically initialized.
- Derived tensors that are not transferred and must be recomputed.
- Aliased/shared storage.

Do not infer support merely because `convert_layer_to_vllm_kernel` exists; test it against the exact target vLLM topology and configuration.

### 11.2 Sender

For each supported quantization block or layer:

1. Retain a bounded old BF16 value immediately before its optimizer update.
2. Update the parameter.
3. Quantize both the bounded old and new values into the exact final vLLM representation.
4. Include every weight, scale, packed tensor, and auxiliary value needed by that kernel representation.
5. XOR the final old and new representations.
6. Encode the delta and release all old-value scratch.

This is intentionally model- and kernel-specific. It must not require a persistent previous exported kernel model.

### 11.3 Receiver

Build the resident destination table once during initialization:

```python
tensor_id -> parameter object, base pointer, byte length, dtype, shape
```

For each decoded block:

1. Validate tensor ID and byte range.
2. Ensure alignment required by the decoder kernel.
3. XOR decoded bytes directly into the resident range.
4. Never replace the `Parameter`, its `.data` storage, or a CUDA-graph-visible pointer.

After all transferred tensors are updated:

- Recompute MLA absorbed weights through the existing helper.
- Run any explicitly required model-specific derived-state refresh.
- Synchronize the CUDA stream before publishing the new model version.

The current direct `load_weights_kernel()` implementation is the semantic starting point, but delta application should live in a separate function rather than add conditionals inside its `param.copy_()` loop.

## 12. Optimizer integration strategy

Optimizer integration follows the protocol and receiver proof, but it is required for the first real end-to-end mode. The production architecture cannot generate exact deltas without either observing the update or retaining a previous model.

### 12.1 Why step pre/post hooks alone are insufficient

An optimizer pre-hook runs before the entire `optimizer.step()` and a post-hook runs after the entire step. Saving old parameters in the pre-hook therefore creates a second full model unless the optimizer itself can update a bounded subset between snapshots.

### 12.2 Why repeated mutation of `param_groups` is not the target design

Temporarily giving an unchanged optimizer one group or bucket and calling `step()` repeatedly can change observable or actual semantics:

- Step hooks execute multiple times.
- Closures can execute multiple times.
- AMP/GradScaler expects one logical optimizer step.
- Fused and foreach grouping decisions change.
- Optimizers may perform cross-parameter/global operations.
- Schedulers, profilers, and checkpoint code see altered optimizer structure.
- Muon and other non-separable optimizers are not equivalent to per-bucket invocation.

It may be used only as a constrained experiment with explicit equivalence tests, not advertised as optimizer-generic behavior.

### 12.3 Bounded-snapshot AdamW milestone

Prime's default optimizer is PyTorch AdamW. Implement an AdamW-specific experimental path that preserves one logical step while internally processing bounded buckets:

```text
snapshot bucket old bytes
  -> perform mathematically identical AdamW update for that bucket
  -> XOR/compress old and new bytes
  -> release snapshot
  -> process next bucket
```

Requirements:

- Same parameter values as configured PyTorch AdamW.
- Same `exp_avg`, `exp_avg_sq`, and step counters.
- Same behavior for parameters with `grad is None`.
- Same decoupled weight decay.
- Same skipped-step behavior under AMP.
- One externally visible optimizer step.
- Compatibility decision for `CPUOffloadOptimizer` stated explicitly.

### 12.4 Final fused optimizer design

The desired end state fuses delta production into the update kernel:

```text
old_bits = load(parameter)
new_value = adamw(old_value, gradient, optimizer_state)
store(parameter, new_value)
delta_bits = old_bits XOR bitcast(new_value)
encode delta_bits
```

This eliminates:

- Old-weight snapshots.
- A second parameter read for comparison.
- A standalone XOR pass.
- Full previous-model storage.

The optimizer produces compressed frames or bounded intermediate block buffers. Those buffers remain on GPU by default until the immediately following inference update.

### 12.5 Preserve the optimizer/transport boundary with an explicit artifact

The optimizer must not call NCCL and must not know about vLLM. It publishes an immutable `WeightDeltaUpdate` artifact that the later broadcast consumes:

```python
delta_recorder.begin(step=progress.step, enabled=should_broadcast)
optimizer.step()
delta_update = delta_recorder.finish()
optimizer.zero_grad()
scheduler.step()

if should_broadcast:
    weight_broadcast.broadcast_weights(
        model,
        step=progress.step,
        delta_update=delta_update,
    )
```

`WeightDeltaUpdate` contains model/base version, source parameter IDs, shard metadata, encoded GPU frames, byte counts, and checksums. The optimizer/delta recorder owns generation; the broadcaster owns transport and consumes/releases the artifact. This changes the data contract between the two phases without moving NCCL into `optimizer.step()`.

## 13. Buffering and pipelining

### 13.1 First implementation

- Encode sequentially.
- Store encoded frames in GPU memory.
- Enforce a configurable total delta-buffer budget.
- Fall back to the existing full broadcast if compression is uneconomical or the budget would be exceeded.
- Pause inference only at the existing synchronization point.
- NCCL broadcast the encoded GPU frames directly.
- Receive and apply it synchronously.

### 13.2 Optimized implementation

Use bounded double buffering:

```text
trainer:
  frame N NCCL broadcast
  frame N+1 ready in a GPU frame buffer

receiver:
  frame N decompress/apply
  frame N+1 NCCL receive
```

Use distinct CUDA streams with explicit events. Do not rely on implicit default-stream ordering.

Memory accounting must include:

- Encoder workspace.
- One bounded old-value snapshot if used.
- Encoded GPU frame buffers.
- Receiver compressed staging buffers.
- Receiver decompression workspace.
- Layer-bounded receiver scratch used for delta routing.

Pinned-CPU spill may be benchmarked later as an opt-in low-memory mode. It necessarily adds D2H storage and H2D staging because the current PyNccl path broadcasts CUDA tensors, so it is not part of the intended fast path.

## 14. Dirty blocks and expected sparsity

Do not assume that most blocks will be completely unchanged.

AdamW's decoupled weight decay can modify parameters even when gradient values are zero. For actively trained parameters, nearly every reasonably sized block may contain at least one changed element.

The likely benefit is lower entropy in XOR bytes, not necessarily sparse dirty blocks:

- Sign bits usually remain stable.
- Exponent bits often remain stable.
- Small updates primarily alter mantissa bits.
- High bytes or high-order bits may contain many zeros.
- Padding and some frozen/unused tensors may be exactly unchanged.

Zero-block detection should still be included because it is cheap when fused with XOR generation. The measurement phase must report zero-block rates separately from compression ratios.

## 15. Correctness and recovery semantics

### 15.1 Bit-exactness

For supported direct updates, the primary correctness condition is byte equality with a full synchronization:

```python
torch.equal(
    delta_updated_parameter.view(torch.uint8),
    full_sync_parameter.view(torch.uint8),
)
```

Logit equivalence is an additional end-to-end check, not a substitute for byte equality.

### 15.2 Skipped optimizer steps

If AMP/GradScaler skips an optimizer step:

- Do not increment the model version.
- Do not publish a nonempty delta.
- Do not cause inference to apply an update.

### 15.3 Multiple steps between broadcasts

For raw XOR:

```text
(v0 XOR v1) XOR (v1 XOR v2) = v0 XOR v2
```

However, independently compressed frames cannot generally be composed without decoding. The first implementation should require one delta synchronization per published optimizer version. Supporting accumulation across unpublished steps is deferred and must have explicit state handling.

### 15.4 Partial failures

Potential failures include:

- Version mismatch.
- Tensor table mismatch.
- Invalid tensor range.
- Checksum failure.
- NCCL failure mid-update.
- Decoder failure.
- Model-specific derived-state refresh failure.

Initial policy:

- Mark the current delta session invalid.
- Do not advertise the target version.
- Require a full synchronization before applying another delta.
- Resume inference only if the model is known to be in a coherent version; otherwise surface a fatal update error rather than serve a mixed model.

A future transactional or rollback scheme can be considered after the basic implementation is stable.

## 16. Testing plan

### 16.1 Protocol unit tests

- Header round-trip.
- Tensor table stable hashing.
- Frame serialization/deserialization.
- Unknown protocol version rejection.
- Base-version mismatch rejection.
- Tensor descriptor mismatch rejection.
- Corrupt header, descriptor, and payload checksum rejection.
- Duplicate/overlapping block rejection.
- Out-of-range block rejection.
- Zero, raw, compressed, and absolute block encodings.
- Full-sync fallback threshold.

### 16.2 Codec unit tests

- Random byte tensors.
- Identical tensors.
- One changed byte per block.
- All bytes changed.
- BF16, FP32, FP8, scale tensors, and odd byte lengths where supported.
- Non-contiguous inputs canonicalized correctly.
- CPU reference versus GPU encoder bit equality.
- CPU reference versus GPU decoder bit equality.
- In-place decode-and-XOR versus materialized decode.
- Reapplying the same XOR returns the original bytes, used only as a codec property test and never as retry behavior.

### 16.3 BF16 source-delta integration tests

- Initial full baseline followed by one delta.
- Several consecutive deltas.
- Forced full fallback followed by further deltas.
- Receiver restart followed by full resynchronization.
- Comparison against the existing full NCCL/checkpoint loader.
- TP=1 and TP>1.
- Pipeline-parallel ownership/skips.
- Dense and MoE models.
- Tied embeddings/lm head.
- Model-specific conversion chains.

### 16.4 Kernel-space integration tests

- Exact resident bytes versus ordinary vLLM full reload.
- Q/K/V fused regions.
- Gate/up fused regions.
- Row- and column-parallel shards.
- KV head replication.
- Local expert selection and EPLB where supported.
- Quantized weights and every corresponding scale.
- Padding bytes.
- Shared/aliased parameter storage.
- MLA absorbed weight recomputation.
- CUDA graph pointer stability.

### 16.5 Optimizer equivalence tests

For the AdamW-specific implementation, compare every step with PyTorch AdamW:

- Parameters.
- Gradients unchanged by the wrapper.
- `exp_avg`.
- `exp_avg_sq`.
- Per-parameter step counters.
- Multiple parameter groups and hyperparameters.
- `grad is None`.
- Weight decay on/off.
- Gradient accumulation.
- AMP success and overflow/skip.
- State-dict save/load and resume.
- CPU optimizer-state offload if supported.

### 16.6 End-to-end generation tests

- Full sync and delta sync produce identical logits on fixed prompts.
- Prefix-cache version salting remains correct.
- Engines remain paused throughout mutation.
- No request observes a partially updated model.
- Resume occurs after successful update.
- Failure behavior does not silently resume a mixed model.

## 17. Benchmark plan

Collect timings with CUDA events for GPU work and wall-clock timings for orchestration and communication.

For every update report:

```text
model/version
full tensor bytes
zero block count and percentage
raw-XOR block count
compressed block count
metadata bytes
compressed payload bytes
overall wire bytes
compression ratio
baseline/export time
XOR/encode time
CPU/GPU staging time
NCCL transfer time
receive time
decode/apply time
vLLM postprocessing time
total inference pause time
peak temporary GPU bytes
peak pinned CPU bytes
```

Compare at least:

1. Existing checkpoint-format NCCL broadcast.
2. Existing kernel-format NCCL broadcast where supported.
3. BF16 source-layout XOR with reference codec and delta-aware routing.
4. BF16 source-layout XOR with GPU codec and delta-aware routing.
5. Model-specific quantized kernel-space direct XOR, when implemented.
6. Fused decode-and-XOR.

Vary:

- Model size and architecture.
- BF16 versus supported quantized serving.
- TP and DP size.
- Inter-node fabric.
- Block size.
- Frame size.
- Number of optimizer steps/model update magnitude.
- Compression codec parameters.

The go/no-go result is total pause-time and training-throughput improvement, not compression ratio in isolation.

## 18. Implementation milestones and acceptance gates

### Milestone 0: measurement harness

Deliverables:

- Layerwise consecutive-weight XOR statistics.
- Zero-block and entropy/compression measurements for several block sizes.
- Timing breakdown for current full NCCL synchronization.
- No production protocol changes.

Gate:

- Demonstrate enough expected end-to-end savings to justify transport implementation.

### Milestone 1: protocol and CPU codec

Deliverables:

- Versioned headers, tensor table, frames, block descriptors, and checksums.
- CPU reference encoder/decoder.
- Golden tests and malformed-input tests.

Gate:

- Exact round-trip for all supported tensor dtypes and shapes.

### Milestone 2: bit-preserving vLLM delta-loader proof

Deliverables:

- One explicitly scoped BF16 model/configuration.
- Layer-bounded scratch destination loading.
- Automated audit of Prime conversion and vLLM loader operations.
- A test proving `route(old XOR new) == route(old) XOR route(new)` byte for byte.
- No receiver-side previous checkpoint or second model.

Gate:

- Source-layout XOR is routed into every resident destination byte exactly like two ordinary absolute loads for the supported configuration.

### Milestone 3: bounded-snapshot DeltaAdamW

Deliverables:

- AdamW-specific bounded-bucket update implementation.
- Exact delta production while old bucket bytes still exist.
- Compressed frame storage in a budgeted GPU update artifact.
- Full optimizer-state equivalence tests.
- No persistent previous model on the trainer.

Gate:

- Repeated steps and checkpoint resume are equivalent to PyTorch AdamW within the exact guarantees expected for the chosen kernels.

### Milestone 4: end-to-end BF16 NCCL delta

Deliverables:

- Separate `broadcast_weights_delta()` NCCL sender.
- Separate delta-aware receiver path.
- Full-sync initialization, versioning, checksums, and recovery.
- Direct resident XOR after bounded loader routing.
- Timing and memory metrics.

Gate:

- Several consecutive updates match existing full synchronization exactly without a persistent previous model on either side.

### Milestone 5: GPU/AState codec

Deliverables:

- GPU XOR, zero detection, and compression.
- GPU decompression.
- CPU reference cross-checks.
- Detailed workspace and timing metrics.

Gate:

- Bit-exact output and a measured end-to-end improvement over the CPU reference path.

### Milestone 6: pipelined NCCL frames

Deliverables:

- Large framed broadcasts.
- Double-buffered receive/apply.
- Explicit CUDA streams/events.
- Bounded memory accounting.

Gate:

- Pipelining reduces or does not regress total pause time and remains stable under multi-node tests.

### Milestone 7: fused optimizer emission

Deliverables:

- Optimizer kernel emits XOR/encoded blocks while updating parameters.
- No old-model baseline and no bucket clone.
- Profiling of added optimizer-step overhead versus synchronization savings.

Gate:

- Net training throughput improves, not merely weight-transfer time.

### Milestone 8: first quantized kernel-space integration

Deliverables:

- Explicitly scoped supported model, topology, GPU architecture, and vLLM quantization configuration.
- Bounded old/new final-kernel production.
- Weight, scale, packed-metadata, and derived-state handling.
- Direct resident parameter XOR.
- Full-update oracle comparison.

Gate:

- Every resident byte and fixed-prompt logit matches the ordinary quantized full loader for repeated updates.

### Milestone 9: hardening and broader support

Deliverables:

- Additional model exporters.
- Quantized formats and scale handling.
- Failure/restart testing.
- Operational metrics and clear fallback logs.
- Documentation of supported and unsupported configurations.

Gate:

- Opt-in production trials complete without baseline divergence or mixed-version serving.

## 19. Initial file-level work list

The first implementation PR should prove the protocol and receiver mapping without yet changing optimizer behavior.

1. Add shared protocol and CPU reference codec modules.
2. Add synthetic old/new tensor generators for measurement and loader-equivalence tests.
3. Implement layer-bounded scratch loading on the inference side for one audited BF16 model/configuration.
4. Prove `route(old XOR new) == route(old) XOR route(new)` byte for byte.
5. Add malformed-frame, versioning, and checksum tests.
6. Add timing and temporary-memory instrumentation.
7. Add unit tests beside `tests/unit/train/rl/test_nccl_broadcast.py` and new inference/protocol tests.
8. Do not expose the mode as a working end-to-end configuration yet.
9. Do not touch NIXL.

The second implementation PR adds the bounded-snapshot AdamW path and end-to-end `bf16_delta` configuration. This avoids landing a production-looking mode that secretly depends on a previous-model baseline.

## 20. Open decisions to resolve with measurements

- Exact initial codec: custom AState codec, general-purpose codec, or both for comparison.
- Compression block size.
- NCCL frame size.
- Whether full fallback is selected per tensor, layer, frame, or whole update.
- Exact acknowledgement mechanism between trainer/orchestrator and every inference worker.
- Transactional behavior for a failure after some direct resident blocks have been applied.
- First model and vLLM configuration for kernel-space support.
- Whether unquantized kernel export should be implemented before FP8 kernel delta.
- Whether duplicate tensor storage is represented once or through explicit aliases.
- Whether optimizer-emitted frames can be transferred before the normal pause point into non-visible receiver staging without excessive receiver memory.

## 21. Recommended first experiment

Before modifying optimizer behavior, record two consecutive post-AdamW model versions from a representative Prime-RL run and evaluate them layer by layer in the exact planned wire dtype.

For block sizes 64 KiB, 128 KiB, and 256 KiB, measure:

- Fraction of blocks with zero XOR.
- Byte histogram and zero-byte fraction of nonzero XOR blocks.
- Encoded size including descriptors.
- GPU or CPU encode/decode throughput.
- Estimated and actual NCCL time saved.
- Current vLLM reload time avoided by the delta-aware resident update path.

This experiment determines whether the primary opportunity is:

- Network compression plus bit-preserving BF16 delta routing.
- Avoiding vLLM reload processing through direct resident XOR.
- Optimizer-integrated delta generation.
- Or some combination of the three.

Only after those measurements should optimizer integration become an implementation dependency.
