from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TEST_SUPPORT_DIR = Path(__file__).resolve().parent
if str(_TEST_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_SUPPORT_DIR))

from moe_xor_audit import audit_glm4_moe_xor  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_glm4_moe_source_xor_routes_and_applies_byte_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    worker_pythonpath = str(_TEST_SUPPORT_DIR)
    if existing_pythonpath:
        worker_pythonpath = f"{worker_pythonpath}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", worker_pythonpath)

    from vllm import LLM

    tensor_parallel_size = int(os.environ.get("MOE_XOR_TP_SIZE", "2"))
    llm = LLM(
        model=os.environ.get("MOE_XOR_GLM4_MODEL", "samsja/mini-glm-moe"),
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=True,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=64,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
    )

    results = llm.apply_model(audit_glm4_moe_xor)
    assert len(results) == tensor_parallel_size
    for result in results:
        assert result["family"] == "glm4_moe"
        assert result["layer"] == 1
        assert result["destination_parameters_checked"] > 0
        assert result["source_tensors_checked"] > 0
        assert result["local_expert_parameters_checked"] > 0
