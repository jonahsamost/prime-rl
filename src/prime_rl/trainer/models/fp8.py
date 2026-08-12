import torch
from torch import Tensor

from prime_rl.weight_sync.fp8 import (
    FP8ScaleFormat,
    quantize_to_fp8_blockwise,
    quantize_to_vllm_kernel_format as _quantize_to_vllm_kernel_format,
)


def quantize_to_vllm_kernel_format(
    weight: Tensor,
    block_size: int = 128,
    *,
    scale_format: FP8ScaleFormat | None = None,
) -> tuple[Tensor, Tensor]:
    """Preserve the model-conversion API while selecting the local kernel scale format."""
    use_ue8m0 = weight.is_cuda and torch.cuda.get_device_capability(weight.device) == (10, 0)
    return _quantize_to_vllm_kernel_format(
        weight,
        block_size,
        scale_format=scale_format or ("ue8m0" if use_ue8m0 else "float32"),
    )

__all__ = ["quantize_to_fp8_blockwise", "quantize_to_vllm_kernel_format"]
