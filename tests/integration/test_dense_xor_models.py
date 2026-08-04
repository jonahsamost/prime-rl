from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TEST_SUPPORT_DIR = Path(__file__).resolve().parent
if str(_TEST_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_SUPPORT_DIR))

from dense_xor_audit import (  # noqa: E402
    audit_gemma_xor,
    audit_llama3_xor,
    audit_mistral_xor,
    audit_qwen3_xor,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_MODEL_AUDITS = (
    ("qwen3_0_6b", "qwen3", "Qwen/Qwen3-0.6B-Base", audit_qwen3_xor),
    ("qwen3_8b", "qwen3", "Qwen/Qwen3-8B-Base", audit_qwen3_xor),
    ("llama3", "llama3", "meta-llama/Llama-3.2-1B-Instruct", audit_llama3_xor),
    ("gemma", "gemma", "google/gemma-3-1b-it", audit_gemma_xor),
    ("mistral", "mistral", "mistralai/Mistral-7B-v0.3", audit_mistral_xor),
)


@pytest.mark.parametrize(
    ("case", "family", "default_model", "audit"),
    _MODEL_AUDITS,
    ids=[item[0] for item in _MODEL_AUDITS],
)
def test_dense_model_source_xor_routes_and_applies_byte_exactly(
    case: str,
    family: str,
    default_model: str,
    audit,
    monkeypatch: pytest.MonkeyPatch,
):
    # vLLM's apply_model API transports a Python callable to the worker. The
    # secure serializer in vLLM 0.26 requires explicit pickle opt-in for that.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    worker_pythonpath = str(_TEST_SUPPORT_DIR)
    if existing_pythonpath:
        worker_pythonpath = f"{worker_pythonpath}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", worker_pythonpath)

    from vllm import LLM

    model = os.environ.get(
        f"DENSE_XOR_{case.upper()}_MODEL",
        os.environ.get(f"DENSE_XOR_{family.upper()}_MODEL", default_model),
    )
    tensor_parallel_size = int(os.environ.get("DENSE_XOR_TP_SIZE", "1"))
    llm = LLM(
        model=model,
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=64,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
    )

    results = llm.apply_model(audit)
    assert len(results) == tensor_parallel_size
    for result in results:
        assert result["family"] == family
        assert result["layer"] == 0
        assert result["destination_parameters_checked"] > 0
        assert result["source_tensors_checked"] > 0
        assert result["non_layer_parameters_checked"] > 0
