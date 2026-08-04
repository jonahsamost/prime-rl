from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TEST_SUPPORT_DIR = Path(__file__).resolve().parent
if str(_TEST_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_SUPPORT_DIR))

from qwen_delta_audit import audit_qwen3_xor_delta  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_MODEL = "Qwen/Qwen3-0.6B-Base"


def test_qwen3_source_xor_routes_and_applies_byte_exactly(monkeypatch: pytest.MonkeyPatch):
    # vLLM's apply_model API transports a Python callable to the worker. The
    # secure serializer in vLLM 0.26 requires explicit pickle opt-in for that.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    worker_pythonpath = str(_TEST_SUPPORT_DIR)
    if existing_pythonpath:
        worker_pythonpath = f"{worker_pythonpath}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", worker_pythonpath)

    from vllm import LLM

    tensor_parallel_size = int(os.environ.get("QWEN_DELTA_TP_SIZE", "1"))
    llm = LLM(
        model=_MODEL,
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=64,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
    )

    assert (
        llm.apply_model(audit_qwen3_xor_delta)
        == [
            {
                "layer": 0,
                "destination_parameters_checked": 8,
                "source_tensors_checked": 11,
                "non_layer_parameters_checked": 3,
            }
        ]
        * tensor_parallel_size
    )
