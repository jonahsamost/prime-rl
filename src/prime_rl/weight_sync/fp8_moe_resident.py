"""TP/EP-local Qwen3-MoE tensors in the representation consumed by vLLM."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor

from prime_rl.weight_sync.fp8 import FP8_BLOCK_SIZE, FP8ScaleFormat, quantize_to_fp8_blockwise
from prime_rl.weight_sync.fp8_resident import (
    build_qwen3_fp8_resident_attention,
    build_qwen3_fp8_resident_layer,
    resident_tensor_name,
)

_EXPERT_KEYS = (
    "mlp.experts.w1",
    "mlp.experts.w2",
    "mlp.experts.w3",
)
_ATTENTION_KEYS = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
}


@torch.no_grad()
def build_qwen3_moe_fp8_resident_layer(
    tensors: dict[str, Tensor],
    *,
    layer_index: int,
    config: object,
    inference_world_size: int,
    scale_format: FP8ScaleFormat,
) -> dict[str, Tensor]:
    """Build every inference rank's resident tensors for one Qwen3-MoE layer."""
    if inference_world_size <= 0:
        raise ValueError(f"inference world size must be positive, got {inference_world_size}")
    if scale_format != "float32":
        raise ValueError("Qwen3-MoE resident FP8 transfer requires float32 checkpoint scales")

    prefix = f"model.layers.{layer_index}."
    relative = {name.removeprefix(prefix): value for name, value in tensors.items() if name.startswith(prefix)}
    present_expert_keys = {name for name in _EXPERT_KEYS if name in relative}
    if not present_expert_keys:
        return build_qwen3_fp8_resident_layer(
            tensors,
            layer_index=layer_index,
            config=config,
            tp_size=inference_world_size,
            scale_format=scale_format,
        )
    missing = set(_EXPERT_KEYS) - present_expert_keys
    if missing:
        raise ValueError(f"Qwen3-MoE FP8 resident conversion is missing tensors: {sorted(missing)}")

    w1 = relative["mlp.experts.w1"]
    w2 = relative["mlp.experts.w2"]
    w3 = relative["mlp.experts.w3"]
    _validate_expert_tensors(w1, w2, w3, inference_world_size)

    output = build_qwen3_fp8_resident_attention(
        relative,
        prefix=prefix,
        config=config,
        tp_size=inference_world_size,
        scale_format=scale_format,
    )
    num_experts = w1.shape[0]
    experts_per_rank = num_experts // inference_world_size
    for inference_rank in range(inference_world_size):
        expert_start = inference_rank * experts_per_rank
        local_w13, local_w2 = partition_qwen3_moe_experts(
            w1,
            w2,
            w3,
            expert_start=expert_start,
            expert_count=experts_per_rank,
        )
        resident_w13, resident_w2, resident_w13_scale, resident_w2_scale = _pack_fp8_moe(
            local_w13,
            local_w2,
            num_global_experts=num_experts,
            experts_per_token=int(getattr(config, "num_experts_per_tok")),
        )
        expert_prefix = f"{prefix}mlp.experts.routed_experts"
        resident = {
            f"{expert_prefix}.w13_weight": resident_w13,
            f"{expert_prefix}.w2_weight": resident_w2,
            f"{expert_prefix}.w13_weight_scale_inv": resident_w13_scale,
            f"{expert_prefix}.w2_weight_scale_inv": resident_w2_scale,
        }
        for name, value in resident.items():
            output[resident_tensor_name(inference_rank, name)] = value

        for name, value in relative.items():
            if name in _ATTENTION_KEYS or name in _EXPERT_KEYS:
                continue
            target_name = "mlp.gate.weight" if name == "mlp.router.gate.weight" else name
            output[resident_tensor_name(inference_rank, prefix + target_name)] = value.detach().contiguous()
    return output


def partition_qwen3_moe_experts(
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    *,
    expert_start: int,
    expert_count: int,
) -> tuple[Tensor, Tensor]:
    """Select one EP shard and fuse gate/up weights into canonical vLLM W13."""
    if expert_start < 0 or expert_count <= 0 or expert_start + expert_count > w1.shape[0]:
        raise ValueError(
            f"invalid expert range [{expert_start}, {expert_start + expert_count}) for {w1.shape[0]} experts"
        )
    local_w1 = w1.narrow(0, expert_start, expert_count)
    local_w3 = w3.narrow(0, expert_start, expert_count)
    local_w2 = w2.narrow(0, expert_start, expert_count).contiguous()
    return torch.cat((local_w1, local_w3), dim=1).contiguous(), local_w2


def _validate_expert_tensors(w1: Tensor, w2: Tensor, w3: Tensor, inference_world_size: int) -> None:
    if w1.ndim != 3 or w2.ndim != 3 or w3.ndim != 3:
        raise ValueError(
            "Qwen3-MoE expert weights must be three-dimensional, "
            f"got w1={tuple(w1.shape)}, w2={tuple(w2.shape)}, w3={tuple(w3.shape)}"
        )
    if w1.shape != w3.shape:
        raise ValueError(f"Qwen3-MoE gate/up shapes differ: w1={tuple(w1.shape)}, w3={tuple(w3.shape)}")
    if w2.shape != (w1.shape[0], w1.shape[2], w1.shape[1]):
        raise ValueError(f"Qwen3-MoE down shape {tuple(w2.shape)} is incompatible with {tuple(w1.shape)}")
    if w1.shape[0] % inference_world_size:
        raise ValueError(f"Qwen3-MoE experts {w1.shape[0]} are not divisible by EP={inference_world_size}")


def _quantize_fp8_moe_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    quantized: list[Tensor] = []
    scales: list[Tensor] = []
    for expert in weight.unbind(0):
        expert_weight, expert_scale = quantize_to_fp8_blockwise(expert, FP8_BLOCK_SIZE)
        quantized.append(expert_weight)
        scales.append(expert_scale)
    return torch.stack(quantized).contiguous(), torch.stack(scales).contiguous()


def _pack_fp8_moe(
    w13: Tensor,
    w2: Tensor,
    *,
    num_global_experts: int,
    experts_per_token: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not w13.is_cuda or not w2.is_cuda:
        raise ValueError("vLLM FP8 MoE resident post-processing requires CUDA tensors")

    w13_quantized, w13_scale = _quantize_fp8_moe_weight(w13)
    w2_quantized, w2_scale = _quantize_fp8_moe_weight(w2)

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        convert_to_fp8_moe_kernel_format,
        select_fp8_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    ep_size = num_global_experts // w13.shape[0]
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=ep_size > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    activation = MoEActivation.SILU
    moe_config = FusedMoEConfig(
        num_experts=num_global_experts,
        experts_per_token=experts_per_token,
        hidden_dim=w13.shape[2],
        intermediate_size=w2.shape[2],
        num_local_experts=w13.shape[0],
        num_logical_experts=num_global_experts,
        activation=activation,
        device=w13.device,
        routing_method=RoutingMethodType.RenormalizeNaive,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    backend, _experts_cls = select_fp8_moe_backend(
        config=moe_config,
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
        allow_vllm_cutlass=False,
    )
    supported = {
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.DEEPGEMM,
        Fp8MoeBackend.MARLIN,
        Fp8MoeBackend.TRITON,
    }
    if backend not in supported:
        raise ValueError(f"Qwen3-MoE resident FP8 transfer does not support vLLM backend {backend.value}")

    layer = SimpleNamespace(
        moe_config=moe_config,
        activation=activation,
        weight_block_size=[FP8_BLOCK_SIZE, FP8_BLOCK_SIZE],
        num_experts=w13.shape[0],
        hidden_size=w13.shape[2],
        intermediate_size_per_partition=w2.shape[2],
        orig_dtype=torch.bfloat16,
        w13_weight=w13_quantized,
    )
    packed_w13, packed_w2, packed_w13_scale, packed_w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=backend,
        layer=layer,
        w13=w13_quantized,
        w2=w2_quantized,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
    )
    return (
        packed_w13.detach().contiguous(),
        packed_w2.detach().contiguous(),
        packed_w13_scale.detach().contiguous(),
        packed_w2_scale.detach().contiguous(),
    )


__all__ = [
    "build_qwen3_moe_fp8_resident_layer",
    "partition_qwen3_moe_experts",
]
