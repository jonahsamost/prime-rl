from __future__ import annotations

import pytest
import torch
from torch import nn

from prime_rl.trainer.delta_adamw import DeltaAdamW
from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaEncoder,
    NvcompLZ4Codec,
    WeightUpdateHeader,
    WeightUpdateKind,
    decode_delta_tensors,
    decode_weight_update_header,
    encode_weight_update_header,
    packed_delta_nbytes,
)
from prime_rl.weight_sync.profiling import WeightSyncMetrics

pytestmark = [pytest.mark.gpu]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16)


def _decode_update(update) -> dict[str, torch.Tensor]:
    decoder = NvcompLZ4Codec(update.payload.device)
    decoded, _events = decode_delta_tensors(
        decoder,
        update.tensors,
        update.frames,
        list(update.frame_payloads()),
    )
    decoder.synchronize()
    return dict(decoded)


def test_nvcomp_lz4_cuda_encode_decode_round_trip_is_byte_exact():
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(7)
    old = torch.randint(-(2**15), 2**15, (1024, 1024), dtype=torch.int16, device=device, generator=generator)
    new = torch.randint(-(2**15), 2**15, (1024, 1024), dtype=torch.int16, device=device, generator=generator)
    delta = torch.bitwise_xor(old, new).view(torch.bfloat16)
    second_delta = torch.bitwise_xor(old[:256], new[:256]).view(torch.bfloat16)
    third_delta = torch.bitwise_xor(old[256:384], new[256:384]).view(torch.bfloat16)
    profile = WeightSyncMetrics()
    encoder = BF16DeltaEncoder(
        base_step=4,
        step=5,
        codec=NvcompLZ4Codec(device),
        profile=profile,
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
    assert profile.raw_bytes == (delta.numel() + second_delta.numel() + third_delta.numel()) * delta.element_size()
    assert profile.compressed_bytes == update.compressed_nbytes
    assert profile.nvcomp_batch_count == 2
    assert profile.nvcomp_peak_pending_batches == 2
    assert profile.frame_count == 2
    assert profile.nvcomp_compress_gpu_ms > 0
    assert profile.nvcomp_encode_gpu_ms > 0
    assert profile.nvcomp_clone_gpu_ms > 0
    assert profile.nvcomp_compress_gpu_ms == pytest.approx(
        profile.nvcomp_encode_gpu_ms + profile.nvcomp_clone_gpu_ms
    )
    assert profile.nvcomp_compress_wall_ms > 0
    assert profile.nvcomp_output_alloc_wall_ms > 0
    assert profile.nvcomp_encode_call_wall_ms > 0
    assert profile.nvcomp_buffer_size_read_wall_ms >= 0
    assert profile.nvcomp_clone_enqueue_wall_ms > 0
    assert torch.equal(_bits(decoded["weight"]), _bits(delta))
    assert torch.equal(_bits(decoded["second_weight"]), _bits(second_delta))
    assert torch.equal(_bits(decoded["third_weight"]), _bits(third_delta))
    assert torch.equal(torch.bitwise_xor(old, _bits(decoded["weight"])), new)


def test_delta_adamw_matches_adamw_and_emits_exact_parameter_xor():
    device = torch.device("cuda", torch.cuda.current_device())
    reference = nn.Sequential(
        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device),
        nn.Linear(32, 32, dtype=torch.bfloat16, device=device),
    )
    delta_model = nn.Sequential(
        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device),
        nn.Linear(32, 32, dtype=torch.bfloat16, device=device),
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


def test_encoder_rejects_non_bf16_and_invalid_versions():
    device = torch.device("cuda", torch.cuda.current_device())
    codec = NvcompLZ4Codec(device)
    with pytest.raises(ValueError, match="must be consecutive"):
        BF16DeltaEncoder(base_step=1, step=3, codec=codec)

    encoder = BF16DeltaEncoder(base_step=1, step=2, codec=codec)
    with pytest.raises(TypeError, match="requires BF16 parameters"):
        encoder.append("weight", torch.zeros(2, dtype=torch.float32, device=device))

    with pytest.raises(ValueError, match="requires CUDA tensors"):
        encoder.append("weight", torch.zeros(2, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "header",
    [
        WeightUpdateHeader(WeightUpdateKind.FULL, base_step=-1, step=7),
        WeightUpdateHeader(WeightUpdateKind.BF16_XOR, base_step=7, step=8),
    ],
)
def test_update_header_round_trip(header: WeightUpdateHeader):
    encoded = encode_weight_update_header(header, device="cpu")

    assert decode_weight_update_header(encoded) == header


def test_update_header_rejects_malformed_or_nonconsecutive_values():
    with pytest.raises(ValueError, match="consecutive versions"):
        encode_weight_update_header(
            WeightUpdateHeader(WeightUpdateKind.BF16_XOR, base_step=2, step=4),
            device="cpu",
        )

    encoded = encode_weight_update_header(
        WeightUpdateHeader(WeightUpdateKind.FULL, base_step=-1, step=2),
        device="cpu",
    )
    encoded[0] = 0
    with pytest.raises(ValueError, match="protocol magic"):
        decode_weight_update_header(encoded)
