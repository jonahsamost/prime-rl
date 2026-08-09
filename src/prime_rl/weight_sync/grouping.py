"""Logical weight-transfer groups shared by full and delta transports."""

from __future__ import annotations

import re

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")


def weight_transfer_group_name(tensor_name: str) -> str:
    match = LAYER_RE.search(tensor_name)
    return "non_layer" if match is None else f"layer.{int(match.group(1))}"


__all__ = ["LAYER_RE", "weight_transfer_group_name"]
