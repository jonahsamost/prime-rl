from __future__ import annotations

import pytest
import torch
from torch import nn

from prime_rl.trainer.delta_adamw import DeltaAdamW
from prime_rl.weight_sync.xor_delta import (
    DeltaEncoder,
    NvcompLZ4Codec,
    ShardedDeltaUpdate,
    WeightUpdateHeader,
    WeightUpdateKind,
    decode_delta_tensors,
    decode_weight_update_header,
    delta_dtype_nbytes,
    encode_weight_update_header,
    integer_view,
    packed_delta_nbytes,
    reconstruct_delta_tensors,
    validate_sharded_delta_update,
)

pytestmark = [pytest.mark.gpu]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return integer_view(tensor)


def _random_bits(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    nbytes = int(torch.tensor(shape).prod().item()) * delta_dtype_nbytes(dtype)
    return (
        torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device, generator=generator).view(dtype).view(shape)
    )


def _decode_update(update) -> dict[str, torch.Tensor]:
    decoder = NvcompLZ4Codec(update.payload.device)
    decoded = decode_delta_tensors(
        decoder,
        update.tensors,
        update.frames,
        list(update.frame_payloads()),
    )
    decoder.synchronize()
    return dict(decoded)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nvcomp_lz4_cuda_encode_decode_round_trip_is_byte_exact(dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(7)
    old = _random_bits((1024, 1024), dtype, device=device, generator=generator)
    new = _random_bits((1024, 1024), dtype, device=device, generator=generator)
    delta = torch.bitwise_xor(_bits(old), _bits(new)).view(dtype)
    second_delta = torch.bitwise_xor(_bits(old[:256]), _bits(new[:256])).view(dtype)
    third_delta = torch.bitwise_xor(_bits(old[256:384]), _bits(new[256:384])).view(dtype)
    encoder = DeltaEncoder(
        base_step=4,
        step=5,
        codec=NvcompLZ4Codec(device),
        pipeline_depth=2,
    )

    encoder.append_batch([("weight", delta), ("second_weight", second_delta)])
    encoder.append("third_weight", third_delta)
    update = encoder.finish()
    decoded = _decode_update(update)

    assert update.base_step == 4
    assert update.step == 5
    assert update.payload.device == device
    assert update.payload.dtype == torch.uint8
    assert len(update.tensors) == 3
    assert len(update.frames) == 2
    assert update.frames[0].first_tensor_index == 0
    assert update.frames[0].tensor_count == 2
    assert update.frames[1].first_tensor_index == 2
    assert update.frames[1].tensor_count == 1
    assert update.payload.numel() == packed_delta_nbytes(update.frames)
    assert all(payload.data_ptr() % 256 == 0 for payload in update.frame_payloads())
    assert torch.equal(_bits(decoded["weight"]), _bits(delta))
    assert torch.equal(_bits(decoded["second_weight"]), _bits(second_delta))
    assert torch.equal(_bits(decoded["third_weight"]), _bits(third_delta))
    assert torch.equal(torch.bitwise_xor(_bits(old), _bits(decoded["weight"])), _bits(new))


def test_four_compressed_dimension_zero_shards_reconstruct_byte_exactly():
    device = torch.device("cuda", torch.cuda.current_device())
    full = torch.arange(80, dtype=torch.int16, device=device).view(10, 8).view(torch.bfloat16)
    local_rows = (3, 3, 2, 2)
    updates = []
    offset = 0
    for shard_index, rows in enumerate(local_rows):
        value = full.narrow(0, offset, rows).clone()
        encoder = DeltaEncoder(
            base_step=8,
            step=9,
            codec=NvcompLZ4Codec(device),
        )
        encoder.append_sharded_bucket(
            [("weight", value)],
            value.reshape(-1),
            global_shapes=[tuple(full.shape)],
            shard_dim=0,
            shard_index=shard_index,
            shard_count=4,
        )
        updates.append(encoder.finish())
        offset += rows

    distributed = ShardedDeltaUpdate(base_step=8, step=9, shards=tuple(updates))
    validate_sharded_delta_update(distributed)
    decoded_shards = []
    metadata_shards = []
    for update in distributed.shards:
        codec = NvcompLZ4Codec(device)
        decoded = decode_delta_tensors(
            codec,
            update.tensors,
            update.frames,
            list(update.frame_payloads()),
        )
        codec.synchronize()
        decoded_shards.append(decoded)
        metadata_shards.append(update.tensors)
    reconstructed = dict(reconstruct_delta_tensors(decoded_shards, metadata_shards))

    assert torch.equal(_bits(reconstructed["weight"]), _bits(full))


def test_distributed_delta_rejects_a_missing_trainer_shard():
    device = torch.device("cuda", torch.cuda.current_device())
    updates = []
    for shard_index in range(3):
        value = torch.zeros((2, 4), dtype=torch.bfloat16, device=device)
        encoder = DeltaEncoder(base_step=1, step=2, codec=NvcompLZ4Codec(device))
        encoder.append_sharded_bucket(
            [("weight", value)],
            value.reshape(-1),
            global_shapes=[(8, 4)],
            shard_dim=0,
            shard_index=shard_index,
            shard_count=4,
        )
        updates.append(encoder.finish())

    with pytest.raises(ValueError, match="expected 0/3|expected 1/3|expected 2/3"):
        validate_sharded_delta_update(ShardedDeltaUpdate(base_step=1, step=2, shards=tuple(updates)))


def test_distributed_delta_rejects_divergent_frame_manifests():
    device = torch.device("cuda", torch.cuda.current_device())
    updates = []
    for shard_index in range(2):
        values = [
            ("first", torch.zeros((2, 4), dtype=torch.bfloat16, device=device)),
            ("second", torch.ones((2, 4), dtype=torch.bfloat16, device=device)),
        ]
        encoder = DeltaEncoder(base_step=3, step=4, codec=NvcompLZ4Codec(device))
        if shard_index == 0:
            bucket = torch.cat([value.reshape(-1) for _, value in values])
            bucket_values = [
                ("first", bucket[:8].view(2, 4)),
                ("second", bucket[8:].view(2, 4)),
            ]
            encoder.append_sharded_bucket(
                bucket_values,
                bucket,
                global_shapes=[(4, 4), (4, 4)],
                shard_dim=0,
                shard_index=shard_index,
                shard_count=2,
            )
        else:
            for name, value in values:
                encoder.append_sharded_bucket(
                    [(name, value)],
                    value.reshape(-1),
                    global_shapes=[(4, 4)],
                    shard_dim=0,
                    shard_index=shard_index,
                    shard_count=2,
                )
        updates.append(encoder.finish())

    with pytest.raises(ValueError, match="incompatible tensor/frame manifest"):
        validate_sharded_delta_update(ShardedDeltaUpdate(base_step=3, step=4, shards=tuple(updates)))


def test_distributed_delta_supports_repeated_unsharded_metadata():
    device = torch.device("cuda", torch.cuda.current_device())
    value = torch.arange(16, dtype=torch.int16, device=device).view(4, 4).view(torch.bfloat16)
    updates = []
    decoded_shards = []
    metadata_shards = []
    for _rank in range(4):
        local = value.clone()
        encoder = DeltaEncoder(base_step=6, step=7, codec=NvcompLZ4Codec(device))
        encoder.append_sharded_bucket([("weight", local)], local.reshape(-1))
        update = encoder.finish()
        updates.append(update)
        decoded_shards.append(list(_decode_update(update).items()))
        metadata_shards.append(update.tensors)

    distributed = ShardedDeltaUpdate(base_step=6, step=7, shards=tuple(updates))
    validate_sharded_delta_update(distributed)
    reconstructed = dict(reconstruct_delta_tensors(decoded_shards, metadata_shards))

    assert torch.equal(_bits(reconstructed["weight"]), _bits(value))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_delta_adamw_matches_adamw_and_emits_exact_parameter_xor(dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    reference = nn.Sequential(
        nn.Linear(32, 32, bias=False, dtype=dtype, device=device),
        nn.Linear(32, 32, dtype=dtype, device=device),
    )
    delta_model = nn.Sequential(
        nn.Linear(32, 32, bias=False, dtype=dtype, device=device),
        nn.Linear(32, 32, dtype=dtype, device=device),
    )
    for parameter in reference.parameters():
        parameter.data.zero_()
    delta_model.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.1)
    wrapped_named_parameters = [
        (name.replace(".", "._checkpoint_wrapped_module.", 1), parameter)
        for name, parameter in delta_model.named_parameters()
    ]
    delta_optimizer = DeltaAdamW(
        params=wrapped_named_parameters,
        lr=1e-3,
        weight_decay=0.1,
        delta_adam_bucket_bytes=1024 * 1024,
    )

    old_state = {name: value.clone() for name, value in delta_model.state_dict().items()}
    for reference_parameter, delta_parameter in zip(reference.parameters(), delta_model.parameters(), strict=True):
        gradient = torch.ones_like(reference_parameter)
        reference_parameter.grad = gradient.clone()
        delta_parameter.grad = gradient.clone()

    with pytest.raises(RuntimeError, match=r"DeltaAdamW\.step\(\) requires begin_delta\(\)"):
        delta_optimizer.step()

    reference_optimizer.step()
    delta_optimizer.begin_delta(base_step=2, step=3)
    delta_optimizer.step()
    update = delta_optimizer.take_delta_update()

    assert update is not None
    assert len(update.frames) == 1
    assert update.frames[0].tensor_count == len(old_state)
    decoded = _decode_update(update)
    assert decoded.keys() == old_state.keys()
    for (name, delta_parameter), reference_parameter in zip(
        delta_model.named_parameters(), reference.parameters(), strict=True
    ):
        assert torch.equal(_bits(delta_parameter), _bits(reference_parameter)), name
        expected = torch.bitwise_xor(_bits(old_state[name]), _bits(delta_parameter))
        assert torch.equal(_bits(decoded[name]), expected), name

    for delta_parameter, reference_parameter in zip(delta_model.parameters(), reference.parameters(), strict=True):
        delta_state = delta_optimizer.state[delta_parameter]
        reference_state = reference_optimizer.state[reference_parameter]
        assert torch.equal(delta_state["step"], reference_state["step"])
        assert torch.equal(delta_state["exp_avg"], reference_state["exp_avg"])
        assert torch.equal(delta_state["exp_avg_sq"], reference_state["exp_avg_sq"])


def test_encoder_rejects_unsupported_dtype_and_nonconsecutive_steps():
    device = torch.device("cuda", torch.cuda.current_device())
    codec = NvcompLZ4Codec(device)
    with pytest.raises(ValueError, match="must be consecutive"):
        DeltaEncoder(base_step=1, step=3, codec=codec)

    encoder = DeltaEncoder(base_step=1, step=2, codec=codec)
    with pytest.raises(TypeError, match="supports"):
        encoder.append("weight", torch.zeros(2, dtype=torch.int32, device=device))

    with pytest.raises(ValueError, match="requires CUDA tensors"):
        encoder.append("weight", torch.zeros(2, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "header",
    [
        WeightUpdateHeader(WeightUpdateKind.FULL, base_step=-1, step=7, optimizer_start_ns=123456789),
        WeightUpdateHeader(WeightUpdateKind.XOR, base_step=7, step=8, optimizer_start_ns=123456789),
    ],
)
def test_update_header_round_trip(header: WeightUpdateHeader):
    encoded = encode_weight_update_header(header, device="cpu")

    assert decode_weight_update_header(encoded) == header


def test_update_header_rejects_malformed_or_nonconsecutive_values():
    with pytest.raises(ValueError, match="optimizer_start_ns must be non-negative"):
        encode_weight_update_header(
            WeightUpdateHeader(WeightUpdateKind.FULL, base_step=-1, step=2, optimizer_start_ns=-1),
            device="cpu",
        )

    with pytest.raises(ValueError, match="consecutive versions"):
        encode_weight_update_header(
            WeightUpdateHeader(WeightUpdateKind.XOR, base_step=2, step=4),
            device="cpu",
        )

    encoded = encode_weight_update_header(
        WeightUpdateHeader(WeightUpdateKind.FULL, base_step=-1, step=2),
        device="cpu",
    )
    encoded[0] = 0
    with pytest.raises(ValueError, match="protocol magic"):
        decode_weight_update_header(encoded)

    encoded = encode_weight_update_header(
        WeightUpdateHeader(WeightUpdateKind.FULL, base_step=-1, step=2),
        device="cpu",
    )
    encoded[1] = 99
    with pytest.raises(ValueError, match="weight update kind"):
        decode_weight_update_header(encoded)
