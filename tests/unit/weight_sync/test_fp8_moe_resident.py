from types import SimpleNamespace

import pytest
import torch

from prime_rl.weight_sync.fp8_moe_resident import (
    build_qwen3_moe_fp8_resident_layer,
    partition_qwen3_moe_experts,
)


def test_partition_qwen3_moe_experts_selects_and_fuses_ep_shard():
    w1 = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    w3 = w1 + 100
    w2 = torch.arange(4 * 3 * 2).reshape(4, 3, 2) + 200

    w13, local_w2 = partition_qwen3_moe_experts(w1, w2, w3, expert_start=2, expert_count=2)

    assert torch.equal(w13[:, :2], w1[2:])
    assert torch.equal(w13[:, 2:], w3[2:])
    assert torch.equal(local_w2, w2[2:])
    assert w13.is_contiguous()
    assert local_w2.is_contiguous()


def test_partition_qwen3_moe_experts_rejects_out_of_range_shard():
    weights = torch.empty(4, 2, 2)

    with pytest.raises(ValueError, match="invalid expert range"):
        partition_qwen3_moe_experts(weights, weights, weights, expert_start=3, expert_count=2)


@pytest.mark.gpu
def test_build_qwen3_moe_fp8_resident_layer_uses_rank_local_vllm_names():
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-MoE resident FP8 conversion requires CUDA")

    device = torch.device("cuda", torch.cuda.current_device())
    layer = "model.layers.0."
    tensors = {
        layer + "self_attn.q_proj.weight": torch.randn(256, 256, dtype=torch.bfloat16, device=device),
        layer + "self_attn.k_proj.weight": torch.randn(256, 256, dtype=torch.bfloat16, device=device),
        layer + "self_attn.v_proj.weight": torch.randn(256, 256, dtype=torch.bfloat16, device=device),
        layer + "self_attn.o_proj.weight": torch.randn(256, 256, dtype=torch.bfloat16, device=device),
        layer + "self_attn.q_norm.weight": torch.randn(128, dtype=torch.bfloat16, device=device),
        layer + "self_attn.k_norm.weight": torch.randn(128, dtype=torch.bfloat16, device=device),
        layer + "mlp.router.gate.weight": torch.randn(4, 256, dtype=torch.bfloat16, device=device),
        layer + "mlp.experts.w1": torch.randn(4, 128, 256, dtype=torch.bfloat16, device=device),
        layer + "mlp.experts.w2": torch.randn(4, 256, 128, dtype=torch.bfloat16, device=device),
        layer + "mlp.experts.w3": torch.randn(4, 128, 256, dtype=torch.bfloat16, device=device),
        layer + "input_layernorm.weight": torch.randn(256, dtype=torch.bfloat16, device=device),
        layer + "post_attention_layernorm.weight": torch.randn(256, dtype=torch.bfloat16, device=device),
    }
    config = SimpleNamespace(
        hidden_size=256,
        head_dim=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts_per_tok=2,
    )

    resident = build_qwen3_moe_fp8_resident_layer(
        tensors,
        layer_index=0,
        config=config,
        inference_world_size=2,
        scale_format="float32",
    )

    for rank in range(2):
        prefix = f"__inference_rank_{rank}__.model.layers.0."
        expert_prefix = prefix + "mlp.experts.routed_experts."
        assert resident[expert_prefix + "w13_weight"].shape[0] == 2
        assert resident[expert_prefix + "w2_weight"].shape[0] == 2
        assert resident[expert_prefix + "w13_weight"].dtype in (torch.float8_e4m3fn, torch.int32)
        assert resident[expert_prefix + "w2_weight"].dtype in (torch.float8_e4m3fn, torch.int32)
        assert resident[expert_prefix + "w13_weight_scale_inv"].shape[0] == 2
        assert resident[expert_prefix + "w2_weight_scale_inv"].shape[0] == 2
        assert resident[prefix + "mlp.gate.weight"].dtype == torch.bfloat16
