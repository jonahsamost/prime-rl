# NIXL XOR Delta Weight Transfer Plan

## Objective

Add exact, lossless XOR-delta weight updates to the NIXL weight-transfer path so
that large dense and non-FP8 MoE policies can synchronize from sharded trainers
to sharded vLLM inference workers without gathering the update on trainer rank
zero or sending every source shard to every inference rank.

The intended steady-state path is:

```text
trainer-local optimizer update
    -> trainer-local XOR bytes
    -> trainer-local nvCOMP frames
    -> NIXL pull by only the inference workers that need those frames
    -> receiver-local decode and layout replay
    -> XOR into resident vLLM parameters
```

The existing full NIXL transfer remains the authoritative startup and recovery
path. The existing NCCL XOR implementation remains a useful dense-model
reference and fallback for small deployments, but it will not be extended into
a second rank-aware, MoE-aware transport system.

## Scope

Initial model scope:

- Text-only dense transformer models already supported by the NIXL load graph.
- Text-only, non-FP8 MoE models, beginning with the repository's miniature GLM
  MoE configuration and then GLM-5.x-style models.
- BF16, FP16, and FP32 tensor storage where the trainer wire representation and
  inference destination representation match exactly for each tensor.
- Tensor parallelism and expert parallelism on the inference side.
- FSDP `Shard(0)` source parameters on the trainer side.

Explicitly out of scope for the first implementation:

- FP8 or other quantized inference parameters.
- Multimodal/vision models.
- Arithmetic or lossy conversions between delta generation and destination
  storage.
- Asynchronous policy updates while inference continues serving requests.
- Replacing the existing NCCL XOR path.
- Implementing direct trainer-to-inference routing on top of NCCL.
- Changing `optimization_dtype` or `reduce_dtype`.

## Why NIXL Is the Production Transport

The NCCL XOR prototype proves that exact optimizer-time XOR capture, nvCOMP LZ4
compression, versioned updates, and in-place inference application work. Its
transport topology is nevertheless the wrong foundation for very large models:

```text
all trainer ranks
    -> gather complete compressed update on trainer rank 0
    -> broadcast every compressed trainer shard to every inference rank
```

This creates three scaling problems:

1. Trainer rank zero holds the complete compressed model update.
2. Every inference rank holds the complete compressed model update before
   applying it.
3. Every inference rank receives data for TP slices and MoE experts it does not
   own.

The NIXL backend already solves the underlying topology and lifetime problems:

- Every serving trainer rank owns a NIXL agent.
- Registered CUDA staging arenas have stable remote addresses.
- The trainer tensor table describes which agent owns each logical source
  range.
- Each inference worker traces its own vLLM loading path and builds a local
  TP/EP-aware transfer plan.
- Inference workers pull directly from the trainer agents that own their source
  ranges.
- Layer groups and acknowledgements bound staging-buffer lifetime.
- Receive-buffer pipelining overlaps the next pull with current-group replay.

The delta implementation should add a compressed, versioned representation to
that system rather than recreate the system inside the NCCL broadcaster.

## Architectural Boundaries

The implementation must keep four concerns separate.

### 1. Delta generation

Delta generation observes old and new trainer-local parameter values and emits
exact XOR bytes. It owns:

- Parameter identity and source-shard descriptions.
- Exact storage dtype.
- Old/new byte comparison.
- Bucketing and nvCOMP encoding.
- Frame boundaries and frame-local tensor spans.
- Compression statistics.

It must not know about:

- NCCL communicators.
- NIXL agents or remote addresses.
- ModelExpress sessions.
- vLLM modules or inference ranks.

The existing transport-neutral pieces in
`src/prime_rl/weight_sync/xor_delta.py` are the starting point. Any types that
currently exist only to satisfy the NCCL gather protocol should be generalized
or moved so the core layer describes a rank-local delta independently of how it
will be transported.

### 2. Delta metadata

Logical delta metadata must be serializable independently of both NCCL and
NIXL. It should describe:

- `base_step` and `step`.
- Source tensor name.
- Tensor dtype.
- Global shape.
- Trainer-local shard offset and extent.
- Compressed frame ID.
- Uncompressed byte range within the decoded frame.
- Compressed and uncompressed sizes.
- Any alignment required by nvCOMP.
- Transfer-group identity.

Transport publication then decorates that logical metadata with physical
locations. For NIXL this includes:

- Trainer agent identity.
- Registered staging-buffer index.
- Remote CUDA address.
- Offset and length of the compressed frame.
- Generation identifier used to prevent stale-buffer reads.

This gives us a clean layering:

```text
logical DeltaUpdate / DeltaManifest
    + NCCL adapter: collective serialization and broadcast
    + NIXL adapter: registered addresses and pull descriptors
```

The NIXL manifest should live with the NIXL broadcast implementation; the
logical delta structures and validation should remain under
`prime_rl.weight_sync`.

### 3. Transport and synchronization

The transport owns movement and lifetime, not model conversion. The NIXL delta
transport is responsible for:

- Registering persistent compressed staging arenas.
- Publishing the current manifest through ModelExpress.
- Keeping a staging generation alive until all required inference workers
  acknowledge it.
- Allowing inference ranks to pull selected compressed frames directly from
  their owning trainer agents.
- Bounding trainer and receiver scratch memory by transfer group rather than
  total model size.
- Reporting transfer failures without advancing the policy version.

### 4. Inference layout routing and application

The inference layer owns the mapping from decoded source bytes to live vLLM
parameters. It is responsible for:

- Reusing the existing symbolically traced NIXL load graph.
- Determining which source ranges a particular TP/EP rank needs.
- Mapping required source ranges to compressed frames.
- Replaying byte-preserving layout operations.
- Applying XOR directly to the matching destination byte ranges.
- Rejecting transformations for which source XOR is not equivalent to
  destination XOR.

This logic must not be implemented as a growing list of model-name-specific
special cases.

## Core Correctness Invariants

### Exact representation

XOR must be generated in the exact representation stored by inference. A BF16
delta cannot be cast to FP16, and an FP32 optimizer-master delta cannot be
applied to a BF16 inference parameter.

For the initial implementation, every logical source tensor must have an exact
wire/destination dtype match. Mixed-dtype models are allowed: for example, BF16
model weights and FP32 router weights can coexist as separately typed tensors.
There must be no single global `model_dtype` assumption.

If the full NIXL path casts a trainer tensor before loading it into inference,
the delta must be computed from the pre-update and post-update values after that
same cast. It is incorrect to XOR FP32 optimizer values and then cast the XOR
bytes to BF16.

### Byte-preserving transformations

Source XOR can be replayed through operations that only select, reorder, or
copy bytes, including:

- Narrow/select/slice.
- Reshape/view/flatten.
- Transpose/permute.
- Split/chunk/unbind.
- Concatenation or packing expressed as copies into destination ranges.
- Exact-dtype contiguous copies.

It cannot in general be replayed through:

- Dtype conversion.
- Quantization or dequantization.
- Scaling or other arithmetic.
- Nonlinear weight preprocessing.
- A kernel conversion that changes numerical representation rather than byte
  layout.

Transfer-plan construction must classify the traced operation chain and reject
an unsafe delta plan before the first delta update. Rejection should select or
require a full transfer, not silently produce an approximate update.

### Versioning

Every delta names exactly one transition:

```text
base_step -> step, where step == base_step + 1
```

An inference worker applies the delta only when its resident version equals
`base_step`. The resident version advances only after every transfer group has
been decoded and applied successfully.

Because application is in-place and groupwise, a failure after the first group
may leave a partially updated model. Such a worker must not resume serving. It
must be marked as requiring a full NIXL resynchronization.

### Buffer lifetime

A compressed frame's remote address is valid from publication of its generation
until every inference worker that may read it has acknowledged completion. A
trainer must never overwrite or reallocate a published buffer early.

### Bounded memory

No component may allocate storage proportional to the complete compressed model
update except the rank-local producer payload if retained temporarily during an
initial migration phase. The final design must bound memory by a configured
number of transfer groups/frames:

```text
trainer:   O(staging_group_bytes * staging_buffer_count)
receiver:  O(receive_group_bytes * receive_buffer_count + decode scratch)
rank zero: O(metadata), not O(compressed_model_bytes)
```

## Frame and Group Design

Compression prevents arbitrary byte-range reads inside a frame. If a frame
contains ten experts and an inference rank owns one of them, that rank must pull
and decode the entire frame. Frame boundaries therefore affect both compression
ratio and routing efficiency.

The initial policy should be:

- Use the same high-level transfer groups as the full NIXL path: non-layer
  tensors followed by individual transformer layers.
- Keep frames bounded by a configurable target size.
- Do not mix tensors from different transfer groups in one frame.
- Split an oversized contiguous source shard into independently compressed
  spans when the delta generator can do so without retaining an unbounded old
  copy.
- Prefer expert-aligned spans for packed MoE tensors when expert boundaries are
  represented by a stable leading dimension.
- Preserve enough source offset metadata to route a decoded span without
  reconstructing the complete global tensor.

We should measure two competing quantities:

```text
compression ratio = uncompressed bytes / compressed bytes
pull amplification = pulled compressed bytes / compressed bytes actually needed
```

Frame size should be tuned from those measurements rather than hard-coded for a
specific model.

## Detailed Implementation Plan

### Step 1: Separate delta generation and metadata from NCCL

Refactor without changing the behavior of the existing NCCL XOR path.

1. Audit `prime_rl.weight_sync.xor_delta` and keep the following concepts
   transport-neutral:
   - Supported bytewise dtypes.
   - Integer/byte views.
   - nvCOMP codec wrapper.
   - Tensor shard metadata.
   - Frame metadata.
   - Rank-local delta update.
   - Encoder and validators.
2. Remove assumptions that a complete distributed update must be represented as
   one rank-zero `ShardedDeltaUpdate`. A distributed manifest should be able to
   reference rank-local updates without owning all payload tensors.
3. Keep `src/prime_rl/trainer/rl/broadcast/nccl_delta.py` as an adapter:
   - Gather rank-local metadata and payloads.
   - Serialize them for NCCL.
   - Broadcast them using the existing protocol.
   - Do not define canonical delta semantics there.
4. Remove the inference-side dependency on NCCL-specific aggregate types. The
   decoder/application API should accept one logical frame or group plus source
   shard metadata.
5. Replace the current global model-dtype parameter with per-tensor dtype
   validation where required.
6. Preserve existing dense NCCL behavior and tests throughout this refactor so
   that subsequent NIXL work has a stable correctness reference.

Deliverable: a rank-local `DeltaUpdate` and logical manifest can be produced,
validated, decoded, and applied without importing NCCL, NIXL, ModelExpress, or
vLLM transport code.

### Step 2: Define the NIXL delta manifest

Add a versioned, serializable manifest for one NIXL policy transition.

The manifest should contain:

- Protocol/schema version.
- `base_step` and `step`.
- Full versus XOR update kind.
- Ordered transfer groups.
- Trainer agents participating in the update.
- Per-agent registered buffer generations.
- Per-frame source tensor spans.
- Per-frame compressed address, compressed size, and uncompressed size.
- Dtype and alignment information.
- Optional compression metrics useful for logging.

Manifest validation must establish before publication that:

- Every tensor span is in bounds.
- Frame payload ranges are non-overlapping and within registered arenas.
- All source shards needed to describe each logical tensor are present.
- Tensor dtype and shape metadata agree across trainer ranks.
- Each group references a live staging generation.
- Policy versions are consecutive.

Trainer rank zero may gather small manifest fragments from trainer ranks and
publish the combined manifest through ModelExpress. It must not gather the
compressed CUDA payloads.

Deliverable: inference workers can discover all remote compressed frames and
their logical contents from metadata alone.

### Step 3: Produce persistent registered compressed arenas

Connect the optimizer-time delta producer to NIXL-owned CUDA memory.

1. Allocate a bounded set of persistent CUDA arenas on each serving trainer
   rank and register them once with that rank's NIXL agent.
2. Divide arenas into generation/group slots compatible with the existing NIXL
   staging-buffer handshake.
3. Arrange for nvCOMP output to land in those arenas, or initially copy completed
   frames from the existing `DeltaUpdate.payload` into them as an incremental
   implementation step.
4. Record the exact compressed offset and size of every frame in the local
   manifest fragment.
5. Do not reuse a slot until all inference workers have acknowledged the group.
6. Retain the current full-update fallback when compression is not beneficial or
   a delta arena cannot represent the update safely.

The first integration may retain one rank-local complete compressed payload
before copying it groupwise into registered memory. That is acceptable for
bringing up the protocol because rank-local size is divided by the trainer FSDP
world size. It is not the final memory target. The final producer should expose
or emit completed frames groupwise so model-wide rank-local retention can also
be removed.

The optimizer must remain numerically identical. This work must not change
optimizer dtype, reduction dtype, AdamW ordering, or update equations.

Deliverable: each trainer agent owns stable, remotely readable compressed frames
for the current policy transition, with bounded and acknowledged reuse.

### Step 4: Extend the NIXL load graph into an XOR routing plan

Reuse the existing full-weight tracing machinery rather than adding a separate
model-specific MoE delta router.

1. Trace the normal PrimeRL-to-HF conversion and vLLM `load_weights` path exactly
   as the full NIXL backend does.
2. For every recorded destination copy, derive:
   - Logical source tensor.
   - Source offset/shape/stride.
   - Destination parameter and byte range.
   - Required byte-preserving replay operations.
3. Classify the operation chain as XOR-safe or XOR-unsafe.
4. Intersect each inference worker's source requirements with trainer FSDP shard
   ranges.
5. Map those intersections to compressed frame dependencies from the manifest.
6. Cache the structural routing plan across policy versions. Refresh only frame
   addresses, sizes, and generations for each update.
7. Ensure aliases, tied parameters, persistent parameters, and vLLM packed
   parameters are applied exactly once where appropriate.

The plan must support mixed per-tensor dtypes, but each individual route must
preserve dtype and byte representation from XOR generation to destination.

Deliverable: each inference rank has a static plan describing which decoded
source spans affect which local destination bytes and a dynamic list of frames
to pull for the current update.

### Step 5: Pull, decode, and apply bounded groups

Implement the receiver-side NIXL delta execution pipeline.

For each transfer group:

1. Wait for the trainer to publish the group generation as ready.
2. Pull only the compressed frames required by this inference rank from their
   owning trainer agents.
3. Decode frames with nvCOMP into reusable receive/decode arenas.
4. Replay the cached byte-routing plan into local vLLM destination views.
5. XOR the decoded bytes into the resident parameters.
6. Run only required post-load bookkeeping that does not numerically transform
   the already updated weights.
7. Synchronize the relevant CUDA work.
8. Acknowledge the group so trainers may reuse its staging slot.
9. Release/reuse the receive and decode buffers before advancing beyond the
   configured pipeline depth.

When memory permits, preserve the existing NIXL overlap pattern:

```text
pull/decode group N+1  ||  replay/apply group N
```

The first correct implementation may serialize pull, decode, and application.
Pipelining should be enabled after byte-exact correctness and buffer lifetime
are verified.

Deliverable: peak receiver memory depends on bounded groups, not total
compressed model size, and rank zero never carries payload bytes.

### Step 6: Add MoE and expert-parallel routing

Extend the generic routing plan to cover packed expert weights and expert
parallelism.

1. Use the traced vLLM loaders to identify expert packing, permutation, and
   destination ownership instead of branching on model names.
2. Determine the local expert set and TP slice for each inference worker.
3. Route intersections of trainer source shards directly into those local
   destinations.
4. Avoid reconstructing a complete global expert tensor or complete global MoE
   layer on any inference rank.
5. Support FP32 router/gate tensors alongside BF16/FP16 expert and attention
   tensors through per-tensor dtype metadata.
6. Ensure replicated tensors are updated consistently without transferring
   duplicate trainer copies unnecessarily.
7. Validate shared experts, routed experts, router weights, sparse-attention
   indexer weights, and model-specific packed projections through the same
   traced-plan abstraction.
8. Remove configuration guards against MoE and expert parallelism only after
   plan construction validates the actual model.

The miniature GLM MoE model should be the first end-to-end target. GLM-5.x
support should follow from the same generic operations; any missing operation
should be added to the trace/replay vocabulary rather than handled by a
`model_type == ...` branch unless the underlying model representation is truly
unique.

Deliverable: a multi-rank inference deployment applies byte-exact deltas to only
its locally owned dense, TP, and expert-parallel parameters.

### Step 7: Preserve full NIXL startup, fallback, and recovery

Delta transfer is an optimization over a known full-policy state, not a
replacement for full synchronization.

Use full NIXL transfer when:

- Inference starts and has no resident base version.
- An inference worker reports a base-version mismatch.
- A delta is not smaller than the corresponding full wire representation.
- Transfer-plan construction finds an XOR-unsafe operation.
- A model contains an unsupported dtype or quantized representation.
- A delta transfer or application fails after partial progress.
- The trainer cannot safely publish a complete manifest/generation.

The update state machine should distinguish at least:

```text
UNINITIALIZED
FULL_SYNC_REQUIRED
READY(base_step)
DELTA_IN_PROGRESS(base_step -> step)
READY(step)
FAILED_REQUIRES_FULL_SYNC
```

The orchestrator must not resume rollouts until all inference workers report the
same successfully applied policy version. A partially updated worker must be
removed from service or fully reloaded; it must never continue from an assumed
version.

The NIXL config should expose delta selection coherently, for example
`delta_mode = "none" | "xor"`, without importing NCCL-specific tuning fields.
Compression frame/group sizing and pipeline depth should be named for their
actual NIXL delta roles.

Deliverable: every failure mode has an explicit safe route back to a full,
version-aligned policy.

### Step 8: Validate correctness, memory, and performance

Testing should proceed from pure logic to real distributed models.

#### Unit validation

- Manifest encode/decode and schema-version rejection.
- Source shard and frame range validation.
- Frame-to-route dependency calculation.
- XOR-safe operation classification.
- Mixed BF16/FP16/FP32 tensors.
- Uneven FSDP `Shard(0)` ranges.
- Expert-aligned and tensor-sliced frames.
- Version mismatch and recovery-state transitions.
- Buffer generation and acknowledgement rules.

#### Small GPU integration

- Dense model: full NIXL startup followed by multiple XOR updates.
- Mini GLM MoE without EP.
- Mini GLM MoE with EP and TP.
- Byte-for-byte comparison against a separately full-reloaded model after every
  update.
- Multiple consecutive deltas to catch base-version drift.
- Forced incompressible update selecting a full fallback.
- Injected version mismatch selecting a full recovery.

#### Supported-model smoke tests

- Qwen dense model.
- Llama dense model.
- Gemma dense model.
- Mistral dense model.
- Repository mini-GLM MoE model.
- A practical intermediate-size GLM/DeepSeek-style MoE before GLM-5.2.

#### Scale validation

For a GLM-5.2-like deployment, record:

- Uncompressed and compressed bytes by trainer rank and transfer group.
- Compression ratio by group.
- Bytes pulled by each inference rank.
- Pull amplification caused by frame granularity.
- NIXL transfer time.
- Decode time.
- Route/apply time.
- End-to-end optimizer-start to inference-apply latency.
- Peak trainer staging memory.
- Peak inference receive/decode memory.
- Per-link and per-agent bandwidth balance.
- Time spent waiting for the slowest trainer or inference rank.

Compare against:

- Full NIXL updates.
- Existing full NCCL updates where runnable.
- Existing NCCL XOR on smaller dense models as a correctness/performance
  reference.

Deliverable: demonstrated byte-exactness and evidence that memory is bounded by
configured transfer groups while network traffic is distributed across trainer
agents and filtered by inference ownership.

## Implementation Phases and Gates

### Phase A: Transport-neutral cleanup

- Complete Step 1.
- Preserve all current NCCL XOR behavior.
- Gate: existing dense XOR tests and benchmarks remain unchanged within normal
  variance.

### Phase B: NIXL dense XOR bring-up

- Complete Steps 2 through 5 for dense models.
- Start with serialized group execution.
- Gate: multiple consecutive NIXL XOR updates are byte-identical to full reloads
  and no payload reaches trainer rank zero.

### Phase C: MoE/EP support

- Complete Step 6.
- Gate: mini-GLM MoE passes with TP and EP, including FP32 router tensors, without
  reconstructing global expert layers.

### Phase D: Recovery and production hardening

- Complete Step 7.
- Add cancellation, timeout, stale-generation, and partial-application tests.
- Gate: every induced failure either leaves the old version intact or forces a
  full resynchronization before serving.

### Phase E: Scale and optimize

- Complete Step 8 at increasing model sizes.
- Tune frame size, group size, arena count, and pipeline depth from measurements.
- Gate: GLM-5.2-like runs stay within the predicted memory bound and improve
  update latency materially over full NIXL transfer.

## Expected Code Ownership

The exact file split may evolve, but dependencies should point in this
direction:

```text
prime_rl.weight_sync
    XOR representation, codec, encoder, manifests, validation
        ^
        |
trainer.delta_adamw
    optimizer-time producer of rank-local logical deltas
        ^
        |
trainer.rl.broadcast.nccl_delta
    legacy NCCL adapter only

trainer.rl.broadcast.nixl
    registered delta arenas, NIXL manifest publication, lifecycle
        |
        v
inference.vllm.worker.nixl
    pull planning, decode scheduling, layout replay, XOR application
```

Neither `prime_rl.weight_sync` nor `DeltaAdamW` should import the NCCL or NIXL
transport implementations. The NCCL and NIXL adapters may depend on the common
delta layer.

## Observability

Every update should report enough information to distinguish compression,
transport, and application bottlenecks:

- Policy transition and update kind.
- Raw and compressed bytes.
- Compression ratio.
- Published and pulled frame counts.
- Useful versus pulled bytes per inference rank.
- Trainer staging duration.
- NIXL pull duration.
- Decode duration.
- Replay/XOR duration.
- Full end-to-end synchronization duration.
- Fallback or recovery reason.

Metrics must be aggregated without requiring rank zero to inspect payloads.

## Completion Criteria

The project is complete when:

1. NIXL supports full startup followed by consecutive XOR updates.
2. Delta generation and logical metadata have no NCCL dependency.
3. Trainer rank zero handles metadata only for NIXL delta updates.
4. Inference workers pull compressed data directly from owning trainer agents.
5. Memory is bounded by configured transfer groups rather than model size.
6. Dense and non-FP8 MoE models use the same generic traced routing system.
7. TP and EP workers apply only their owned destination ranges without building
   complete global MoE layers.
8. Mixed BF16/FP16/FP32 storage is byte-exact where source and destination
   representations match.
9. Version mismatch, unsafe conversion, incompressible delta, and transfer
   failure reliably select full NIXL recovery.
10. Small-model tests are byte-exact and large-model measurements demonstrate
    useful bandwidth and latency improvements over full weight transfer.
