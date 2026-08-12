"""TP-local FP8 tensors in the representation consumed by vLLM kernels."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.weight_sync.fp8 import FP8_BLOCK_SIZE, FP8ScaleFormat, quantize_to_fp8_blockwise

_RANK_PREFIX_RE = re.compile(r"^__inference_rank_(\d+)__\.(.+)$")


def resident_tensor_name(inference_rank: int, name: str) -> str:
    return f"__inference_rank_{inference_rank}__.{name}"


def parse_resident_tensor_name(name: str) -> tuple[int, str]:
    match = _RANK_PREFIX_RE.match(name)
    if match is None:
        raise ValueError(f"invalid rank-local resident tensor name {name!r}")
    return int(match.group(1)), match.group(2)


@torch.no_grad()
def build_qwen3_fp8_resident_layer(
    tensors: dict[str, Tensor],
    *,
    layer_index: int,
    config: object,
    tp_size: int,
    scale_format: FP8ScaleFormat,
) -> dict[str, Tensor]:
    """Build every inference TP rank's resident tensors for one Qwen3 layer."""
    if tp_size <= 0:
        raise ValueError(f"inference TP size must be positive, got {tp_size}")

    prefix = f"model.layers.{layer_index}."
    relative = {name.removeprefix(prefix): value for name, value in tensors.items() if name.startswith(prefix)}
    expected = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    missing = expected - relative.keys()
    if missing:
        raise ValueError(f"Qwen3 FP8 resident conversion is missing tensors: {sorted(missing)}")

    num_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads"))
    configured_head_dim = getattr(config, "head_dim", None)
    head_dim = int(configured_head_dim or getattr(config, "hidden_size") // num_heads)
    if num_heads % tp_size:
        raise ValueError(f"Qwen3 attention heads {num_heads} are not divisible by TP={tp_size}")
    if num_kv_heads >= tp_size and num_kv_heads % tp_size:
        raise ValueError(f"Qwen3 KV heads {num_kv_heads} are not divisible by TP={tp_size}")

    output: dict[str, Tensor] = {}
    for tp_rank in range(tp_size):
        q = _head_shard(relative["self_attn.q_proj.weight"], num_heads, tp_rank, tp_size, head_dim)
        k = _kv_head_shard(relative["self_attn.k_proj.weight"], num_kv_heads, tp_rank, tp_size, head_dim)
        v = _kv_head_shard(relative["self_attn.v_proj.weight"], num_kv_heads, tp_rank, tp_size, head_dim)
        gate = _row_shard(relative["mlp.gate_proj.weight"], tp_rank, tp_size)
        up = _row_shard(relative["mlp.up_proj.weight"], tp_rank, tp_size)

        linears = {
            "self_attn.qkv_proj": torch.cat((q, k, v), dim=0),
            "self_attn.o_proj": _column_shard(relative["self_attn.o_proj.weight"], tp_rank, tp_size),
            "mlp.gate_up_proj": torch.cat((gate, up), dim=0),
            "mlp.down_proj": _column_shard(relative["mlp.down_proj.weight"], tp_rank, tp_size),
        }
        for name, weight in linears.items():
            resident_weight, resident_scale = _pack_fp8_linear(weight, scale_format)
            full_name = f"{prefix}{name}"
            output[resident_tensor_name(tp_rank, f"{full_name}.weight")] = resident_weight
            output[resident_tensor_name(tp_rank, f"{full_name}.weight_scale_inv")] = resident_scale

        for name, value in relative.items():
            if name not in expected:
                output[resident_tensor_name(tp_rank, prefix + name)] = value.detach().contiguous()
    return output


def _head_shard(weight: Tensor, heads: int, rank: int, world_size: int, head_dim: int) -> Tensor:
    if weight.shape[0] != heads * head_dim:
        raise ValueError(f"attention projection shape {tuple(weight.shape)} does not match {heads}x{head_dim} heads")
    heads_per_rank = heads // world_size
    return weight.narrow(0, rank * heads_per_rank * head_dim, heads_per_rank * head_dim).contiguous()


def _kv_head_shard(weight: Tensor, heads: int, rank: int, world_size: int, head_dim: int) -> Tensor:
    if heads >= world_size:
        return _head_shard(weight, heads, rank, world_size, head_dim)
    if world_size % heads:
        raise ValueError(f"TP={world_size} cannot replicate {heads} Qwen3 KV heads evenly")
    source_head = rank // (world_size // heads)
    return weight.narrow(0, source_head * head_dim, head_dim).contiguous()


def _row_shard(weight: Tensor, rank: int, world_size: int) -> Tensor:
    if weight.shape[0] % world_size:
        raise ValueError(f"row-parallel output {weight.shape[0]} is not divisible by TP={world_size}")
    rows = weight.shape[0] // world_size
    return weight.narrow(0, rank * rows, rows).contiguous()


def _column_shard(weight: Tensor, rank: int, world_size: int) -> Tensor:
    if weight.shape[1] % world_size:
        raise ValueError(f"column-parallel input {weight.shape[1]} is not divisible by TP={world_size}")
    columns = weight.shape[1] // world_size
    return weight.narrow(1, rank * columns, columns).contiguous()


def _pack_fp8_linear(weight: Tensor, scale_format: FP8ScaleFormat) -> tuple[Tensor, Tensor]:
    quantized, scales = quantize_to_fp8_blockwise(weight, FP8_BLOCK_SIZE)
    capability = torch.cuda.get_device_capability(weight.device)
    if capability < (8, 9):
        if scale_format != "float32":
            raise ValueError("FP8 Marlin resident tensors require float32 checkpoint scales")
        return _pack_fp8_marlin(quantized, scales, weight.dtype)

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_e8m0_used,
        is_deep_gemm_supported,
        should_use_deepgemm_for_fp8_linear,
    )

    quantized, scales = process_fp8_weight_block_strategy(quantized, scales)
    if is_deep_gemm_supported() and should_use_deepgemm_for_fp8_linear(torch.bfloat16, quantized.shape):
        use_e8m0 = is_deep_gemm_e8m0_used()
        if scale_format == "ue8m0" and not use_e8m0:
            raise ValueError("configured UE8M0 scales do not match the selected vLLM DeepGEMM kernel")
        return deepgemm_post_process_fp8_weight_block(
            wq=quantized,
            ws=scales,
            quant_block_shape=(FP8_BLOCK_SIZE, FP8_BLOCK_SIZE),
            use_e8m0=use_e8m0,
        )
    if scale_format != "float32":
        raise ValueError("UE8M0 resident scales require a vLLM DeepGEMM kernel")
    return quantized, scales


def _pack_fp8_marlin(weight: Tensor, scales: Tensor, source_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )

    layer = nn.Module()
    layer.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
    layer.register_parameter("weight_scale_inv", nn.Parameter(scales, requires_grad=False))
    layer.output_size_per_partition = weight.shape[0]
    layer.input_size_per_partition = weight.shape[1]
    layer.weight_block_size = [FP8_BLOCK_SIZE, FP8_BLOCK_SIZE]
    layer.orig_dtype = source_dtype
    prepare_fp8_layer_for_marlin(layer, size_k_first=False, input_dtype=None)
    return layer.weight.detach().contiguous(), layer.weight_scale_inv.detach().contiguous()


__all__ = [
    "build_qwen3_fp8_resident_layer",
    "parse_resident_tensor_name",
    "resident_tensor_name",
]
