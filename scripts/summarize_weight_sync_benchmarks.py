#!/usr/bin/env python3
"""Summarize paired full and XOR weight-sync benchmark logs."""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(
    r"Step (\d+) \|\s+((?:\d+m\s+)?[\d.]+s).*?Peak Mem\. ([\d.]+) GiB "
    r"\| Policy Update ([\d.]+)s \| Weight Sync ([\d.]+)s"
)
SYNC = re.compile(r"policy v(\d+) synchronized in ([\d.]+)s")
COMPRESSION = re.compile(
    r"policy v(\d+): ([\d.]+)x compression \(([\d.]+) GiB raw, ([\d.]+) GiB compressed\)"
)
INFERENCE_PEAK = re.compile(r"peak_allocated=([\d.]+) GiB")
RECEIVE_ARENAS = re.compile(r"receive_arenas=([\d.]+) MiB")
DECODE_ARENAS = re.compile(r"decode_arenas=([\d.]+) MiB")

SCENARIOS = (
    ("dense-bf16", "Dense BF16"),
    ("dense-fp8", "Dense FP8"),
    ("moe-bf16", "MoE BF16"),
    ("moe-fp8", "MoE FP8"),
)


@dataclass(frozen=True)
class RunMetrics:
    policy_update_s: tuple[float, ...]
    weight_sync_s: tuple[float, ...]
    protocol_sync_s: tuple[float, ...]
    step_s: tuple[float, ...]
    trainer_peak_gib: float
    inference_peak_gib: float
    receiver_buffers_gib: float
    compression_ratio: tuple[float, ...]
    raw_gib: tuple[float, ...]
    compressed_gib: tuple[float, ...]


def _duration_seconds(value: str) -> float:
    match = re.fullmatch(r"(?:(\d+)m\s+)?([\d.]+)s", value)
    if match is None:
        raise ValueError(f"invalid duration: {value!r}")
    return 60 * int(match.group(1) or 0) + float(match.group(2))


def _read_log(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark log: {path}")
    return ANSI_ESCAPE.sub("", path.read_text(errors="replace"))


def _parse_run(path: Path) -> RunMetrics:
    trainer = _read_log(path / "logs" / "trainer.log")
    inference = _read_log(path / "logs" / "inference.log")

    sync_by_step = {int(step): float(seconds) for step, seconds in SYNC.findall(trainer) if int(step) > 0}
    if len(sync_by_step) < 2:
        raise ValueError(f"{path} needs at least two policy updates; found {sorted(sync_by_step)}")

    # The first delta captures CUDA graphs. Exclude v1 from both sides so the
    # paired comparison measures their shared steady-state policy versions.
    measured_steps = set(sync_by_step) - {min(sync_by_step)}
    step_records = {
        int(step): (_duration_seconds(duration), float(peak), float(policy_update), float(weight_sync))
        for step, duration, peak, policy_update, weight_sync in STEP.findall(trainer)
    }
    missing_steps = measured_steps - step_records.keys()
    if missing_steps:
        raise ValueError(
            f"{path} is missing Policy Update metrics for policies {sorted(missing_steps)}; "
            "rerun with the current trainer logging"
        )

    compression_by_step = {
        int(step): (float(ratio), float(raw), float(compressed))
        for step, ratio, raw, compressed in COMPRESSION.findall(trainer)
        if int(step) in measured_steps
    }
    inference_peaks = [float(value) for value in INFERENCE_PEAK.findall(inference)]
    receive_arenas = [float(value) for value in RECEIVE_ARENAS.findall(inference)]
    decode_arenas = [float(value) for value in DECODE_ARENAS.findall(inference)]

    ordered_steps = sorted(measured_steps)
    return RunMetrics(
        policy_update_s=tuple(step_records[step][2] for step in ordered_steps),
        weight_sync_s=tuple(step_records[step][3] for step in ordered_steps),
        protocol_sync_s=tuple(sync_by_step[step] for step in ordered_steps),
        step_s=tuple(step_records[step][0] for step in ordered_steps),
        trainer_peak_gib=max(step_records[step][1] for step in ordered_steps),
        inference_peak_gib=max(inference_peaks, default=float("nan")),
        receiver_buffers_gib=(max(receive_arenas, default=0.0) + max(decode_arenas, default=0.0)) / 1024,
        compression_ratio=tuple(compression_by_step[step][0] for step in ordered_steps if step in compression_by_step),
        raw_gib=tuple(compression_by_step[step][1] for step in ordered_steps if step in compression_by_step),
        compressed_gib=tuple(
            compression_by_step[step][2] for step in ordered_steps if step in compression_by_step
        ),
    )


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values)


def _delta(value: float) -> str:
    return f"{value:+.2f}"


def _print_transfer_table(root: Path) -> None:
    print(
        "| Scenario | Full policy update | XOR policy update | XOR speedup | Full broadcast | XOR broadcast | "
        "Full protocol | XOR protocol | Compression | Payload |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for directory, label in SCENARIOS:
        full = _parse_run(root / directory / "full")
        xor = _parse_run(root / directory / "xor")
        full_update = _mean(full.policy_update_s)
        xor_update = _mean(xor.policy_update_s)
        compression = f"{_mean(xor.compression_ratio):.1f}x" if xor.compression_ratio else "n/a"
        payload = (
            f"{_mean(xor.raw_gib):.2f}→{_mean(xor.compressed_gib):.2f} GiB"
            if xor.compressed_gib
            else "n/a"
        )
        print(
            f"| {label} | {full_update:.2f}s | {xor_update:.2f}s | {full_update / xor_update:.2f}x | "
            f"{_mean(full.weight_sync_s):.2f}s | {_mean(xor.weight_sync_s):.2f}s | "
            f"{_mean(full.protocol_sync_s):.2f}s | {_mean(xor.protocol_sync_s):.2f}s | "
            f"{compression} | {payload} |"
        )


def _print_memory_table(root: Path) -> None:
    print(
        "\n| Scenario | Full trainer peak | XOR trainer peak | XOR trainer Δ | Full inference peak | "
        "XOR inference peak | XOR inference Δ | Full/XOR receiver buffers |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for directory, label in SCENARIOS:
        full = _parse_run(root / directory / "full")
        xor = _parse_run(root / directory / "xor")
        print(
            f"| {label} | {full.trainer_peak_gib:.2f} GiB | {xor.trainer_peak_gib:.2f} GiB | "
            f"{_delta(xor.trainer_peak_gib - full.trainer_peak_gib)} GiB | "
            f"{full.inference_peak_gib:.2f} GiB | {xor.inference_peak_gib:.2f} GiB | "
            f"{_delta(xor.inference_peak_gib - full.inference_peak_gib)} GiB | "
            f"{full.receiver_buffers_gib:.2f}/{xor.receiver_buffers_gib:.2f} GiB |"
        )


def _print_step_table(root: Path) -> None:
    print("\n| Scenario | Full trainer step | XOR trainer step | XOR step speedup | Samples |")
    print("|---|---:|---:|---:|---:|")
    for directory, label in SCENARIOS:
        full = _parse_run(root / directory / "full")
        xor = _parse_run(root / directory / "xor")
        full_step = _mean(full.step_s)
        xor_step = _mean(xor.step_s)
        print(f"| {label} | {full_step:.2f}s | {xor_step:.2f}s | {full_step / xor_step:.2f}x | {len(xor.step_s)} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="suite root containing <scenario>/{full,xor}")
    args = parser.parse_args()
    _print_transfer_table(args.root)
    _print_memory_table(args.root)
    _print_step_table(args.root)
    print("\nSteady-state means exclude policy v1 (CUDA-graph capture). Settings: 512 MiB buckets, depth 8; not tuned.")


if __name__ == "__main__":
    main()
