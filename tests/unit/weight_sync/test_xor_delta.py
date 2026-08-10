from __future__ import annotations

import pytest
import torch
from torch import nn

from prime_rl.trainer.delta_adamw import DeltaAdamW
from prime_rl.weight_sync.xor_delta import (
    DeltaEncoder,
    NvcompLZ4Codec,
    decode_delta_tensors,
    delta_dtype_nbytes,
    integer_view,
    packed_delta_nbytes,
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
    assert update.payload.numel() % 256 == 0
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nvcomp_lz4_cuda_decode_into_reuses_outputs_byte_exactly(dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(17)
    batches = [
        [
            _random_bits((1024, 128), dtype, device=device, generator=generator),
            _random_bits((257,), dtype, device=device, generator=generator),
        ]
        for _ in range(2)
    ]
    codec = NvcompLZ4Codec(device)
    outputs = [
        torch.empty(value.numel() * value.element_size(), dtype=torch.uint8, device=device) for value in batches[0]
    ]

    for values in batches:
        codec.decode_into(codec.encode(values), outputs)
        codec.synchronize()
        for value, output in zip(values, outputs, strict=True):
            assert torch.equal(value.view(torch.uint8).reshape(-1), output)


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
    assert len(update.frames) == len(old_state)
    assert all(frame.tensor_count == 1 for frame in update.frames)
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
