# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed.checkpoint as dcp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP

MergeMethod = Literal["mean", "linear", "sqrt", "cosine", "ema"]

STEP_DIR_RE = re.compile(r"^step-(\d+)$")
METHODS: set[str] = {"mean", "linear", "sqrt", "cosine", "ema"}


@dataclass(frozen=True, slots=True)
class StepCheckpoint:
    step: int
    path: Path


def parse_csv_ints(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{name} must contain integers, got {item!r}") from exc
        if value < 1:
            raise ValueError(f"{name} entries must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def parse_csv_methods(raw: str) -> list[MergeMethod]:
    methods: list[MergeMethod] = []
    for item in raw.split(","):
        method = item.strip().lower()
        if not method:
            continue
        if method not in METHODS:
            raise ValueError(
                f"Unknown merge method {method!r}; choose from {sorted(METHODS)}"
            )
        methods.append(method)  # type: ignore[arg-type]
    if not methods:
        raise ValueError("methods must contain at least one value")
    return methods


def parse_step_dir(path: Path) -> int | None:
    match = STEP_DIR_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def is_dcp_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / ".metadata").is_file()


def discover_checkpoints(checkpoint_root: Path) -> list[StepCheckpoint]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    checkpoints: list[StepCheckpoint] = []
    for child in checkpoint_root.iterdir():
        step = parse_step_dir(child)
        if step is None or not is_dcp_checkpoint_dir(child):
            continue
        checkpoints.append(StepCheckpoint(step=step, path=child))

    checkpoints.sort(key=lambda checkpoint: checkpoint.step)
    if not checkpoints:
        raise FileNotFoundError(
            f"No DCP step-* checkpoints found under {checkpoint_root}"
        )
    return checkpoints


def parse_end_step(raw: str) -> int | Literal["latest"]:
    raw = raw.strip().lower()
    if raw == "latest":
        return "latest"
    try:
        step = int(raw)
    except ValueError as exc:
        raise ValueError("--end-step must be 'latest' or an integer") from exc
    if step < 0:
        raise ValueError("--end-step must be non-negative")
    return step


def select_checkpoint_window(
    checkpoints: list[StepCheckpoint],
    *,
    end_step: int | Literal["latest"],
    window_size: int,
) -> list[StepCheckpoint]:
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if not checkpoints:
        raise ValueError("No checkpoints were provided")

    if end_step == "latest":
        end_index = len(checkpoints) - 1
    else:
        end_index = next(
            (
                index
                for index, checkpoint in enumerate(checkpoints)
                if checkpoint.step == end_step
            ),
            -1,
        )
        if end_index == -1:
            available = ", ".join(str(checkpoint.step) for checkpoint in checkpoints)
            raise ValueError(
                f"Requested end step {end_step} was not found. "
                f"Available steps: {available}"
            )

    start_index = end_index - window_size + 1
    if start_index < 0:
        raise ValueError(
            f"Need {window_size} checkpoints ending at step "
            f"{checkpoints[end_index].step}, but only {end_index + 1} are available"
        )
    return checkpoints[start_index : end_index + 1]


def validate_expected_interval(
    selected_checkpoints: list[StepCheckpoint],
    expected_interval: int | None,
) -> None:
    if expected_interval is None:
        return
    if expected_interval < 1:
        raise ValueError("--expected-interval must be positive when provided")
    if len(selected_checkpoints) <= 1:
        return

    steps = [checkpoint.step for checkpoint in selected_checkpoints]
    diffs = [next_step - step for step, next_step in zip(steps, steps[1:])]
    interior_diffs = diffs[:-1]
    bad_interior_diffs = [diff for diff in interior_diffs if diff != expected_interval]
    final_diff = diffs[-1]
    if bad_interior_diffs or final_diff > expected_interval:
        raise ValueError(
            f"Selected checkpoint steps {steps} are not compatible with expected "
            f"interval {expected_interval}; found intervals {diffs}. Interior "
            "intervals must match exactly, and the final interval may only be "
            "shorter to support partial final checkpoints."
        )


def _validate_checkpoint_steps(checkpoint_steps: list[int]) -> None:
    if not checkpoint_steps:
        raise ValueError("checkpoint_steps must contain at least one step")
    diffs = [
        next_step - step
        for step, next_step in zip(checkpoint_steps, checkpoint_steps[1:])
    ]
    if any(diff <= 0 for diff in diffs):
        raise ValueError(
            f"checkpoint_steps must be strictly increasing, got {checkpoint_steps}"
        )


def infer_nominal_interval(
    checkpoint_steps: list[int],
    expected_interval: int | None = None,
) -> int | None:
    if expected_interval is not None:
        if expected_interval < 1:
            raise ValueError("--expected-interval must be positive when provided")
        return expected_interval
    if len(checkpoint_steps) <= 1:
        return None

    diffs = [
        next_step - step
        for step, next_step in zip(checkpoint_steps, checkpoint_steps[1:])
    ]
    if any(diff <= 0 for diff in diffs):
        raise ValueError(
            f"checkpoint_steps must be strictly increasing, got {checkpoint_steps}"
        )

    counts = Counter(diffs)
    max_count = max(counts.values())
    candidate_intervals = [
        interval for interval, count in counts.items() if count == max_count
    ]
    return max(candidate_intervals)


def _theorem_weights_from_decay(decay_weights: list[float]) -> list[float]:
    if not decay_weights:
        return [1.0]

    checkpoint_weights = [1.0 - decay_weights[0]]
    checkpoint_weights.extend(
        decay_weights[index] - decay_weights[index + 1]
        for index in range(len(decay_weights) - 1)
    )
    checkpoint_weights.append(decay_weights[-1])

    # Tiny negative values can appear from floating point subtraction.
    checkpoint_weights = [
        0.0 if -1e-12 < weight < 0.0 else weight for weight in checkpoint_weights
    ]
    if any(weight < 0.0 for weight in checkpoint_weights):
        raise ValueError(
            f"Derived invalid negative checkpoint weights: {checkpoint_weights}"
        )
    total = sum(checkpoint_weights)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        checkpoint_weights = [weight / total for weight in checkpoint_weights]
    return checkpoint_weights


def compute_checkpoint_weights(
    method: MergeMethod,
    checkpoint_steps: list[int],
    *,
    expected_interval: int | None = None,
    min_lr_factor: float = 0.0,
    ema_decay: float = 0.9,
) -> list[float]:
    _validate_checkpoint_steps(checkpoint_steps)
    num_checkpoints = len(checkpoint_steps)
    if num_checkpoints < 1:
        raise ValueError("num_checkpoints must be positive")
    if not 0.0 <= min_lr_factor <= 1.0:
        raise ValueError("--min-lr-factor must be between 0 and 1")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    if num_checkpoints == 1:
        return [1.0]

    if method == "mean":
        return [1.0 / num_checkpoints] * num_checkpoints

    nominal_interval = infer_nominal_interval(checkpoint_steps, expected_interval)
    if nominal_interval is None:
        return [1.0]

    if method == "ema":
        latest_step = checkpoint_steps[-1]
        raw_weights = [
            (1.0 - ema_decay) * (ema_decay ** ((latest_step - step) / nominal_interval))
            for step in checkpoint_steps
        ]
        total = sum(raw_weights)
        return [weight / total for weight in raw_weights]

    start_step = checkpoint_steps[0]
    virtual_end_step = checkpoint_steps[-1] + nominal_interval
    decay_span = virtual_end_step - start_step
    if decay_span <= 0:
        raise ValueError(
            f"Invalid decay span from checkpoint steps {checkpoint_steps} "
            f"and nominal interval {nominal_interval}"
        )

    decay_weights: list[float] = []
    for step in checkpoint_steps[1:]:
        progress = (step - start_step) / decay_span
        if method == "linear":
            decay_factor = 1.0 - progress
        elif method == "sqrt":
            decay_factor = 1.0 - math.sqrt(progress)
        elif method == "cosine":
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown merge method: {method}")
        decay_weights.append(min_lr_factor + (1.0 - min_lr_factor) * decay_factor)

    return _theorem_weights_from_decay(decay_weights)


def build_empty_model_state_dict(
    model_name: str,
    model_flavor: str,
) -> dict[str, torch.Tensor]:
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model_state_dict = ModelWrapper(model)._get_state_dict()

    empty_state_dict: dict[str, torch.Tensor] = {}
    for key, value in model_state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Expected model state_dict value {key!r} to be a tensor, "
                f"got {type(value).__name__}"
            )
        empty_state_dict[key] = torch.empty_like(value, device="cpu")

    del model_state_dict
    del model
    gc.collect()
    return empty_state_dict


def clone_empty_state_dict(
    template_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty_like(value, device="cpu")
        for key, value in template_state_dict.items()
    }


def load_model_state_dict(
    checkpoint_path: Path,
    template_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state_dict = clone_empty_state_dict(template_state_dict)
    dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
    return state_dict


def _validate_loaded_state_dict(
    loaded_state_dict: dict[str, torch.Tensor],
    template_state_dict: dict[str, torch.Tensor],
    checkpoint_path: Path,
) -> None:
    loaded_keys = set(loaded_state_dict)
    template_keys = set(template_state_dict)
    if loaded_keys != template_keys:
        missing = sorted(template_keys - loaded_keys)
        unexpected = sorted(loaded_keys - template_keys)
        raise ValueError(
            f"Checkpoint {checkpoint_path} keys do not match the model template. "
            f"Missing: {missing[:10]} Unexpected: {unexpected[:10]}"
        )

    for key, template_value in template_state_dict.items():
        loaded_value = loaded_state_dict[key]
        if tuple(loaded_value.shape) != tuple(template_value.shape):
            raise ValueError(
                f"Shape mismatch for {key!r} in {checkpoint_path}: "
                f"loaded={tuple(loaded_value.shape)} "
                f"expected={tuple(template_value.shape)}"
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _metadata_for_merge(
    *,
    args: argparse.Namespace | None,
    method: MergeMethod,
    window_size: int,
    selected_checkpoints: list[StepCheckpoint],
    weights: list[float],
    output_dir: Path,
    accum_dtype: str,
    export_dtype: str,
    expected_interval: int | None,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "window_size": window_size,
        "source_steps": [checkpoint.step for checkpoint in selected_checkpoints],
        "source_intervals": [
            next_checkpoint.step - checkpoint.step
            for checkpoint, next_checkpoint in zip(
                selected_checkpoints, selected_checkpoints[1:]
            )
        ],
        "source_checkpoints": [
            str(checkpoint.path) for checkpoint in selected_checkpoints
        ],
        "weights": weights,
        "weight_progress": "step",
        "nominal_interval": infer_nominal_interval(
            [checkpoint.step for checkpoint in selected_checkpoints],
            expected_interval,
        ),
        "output_dir": str(output_dir),
        "accum_dtype": accum_dtype,
        "export_dtype": export_dtype,
        "args": _jsonable(vars(args)) if args is not None else None,
    }


def save_merge_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output_dir / "merge_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


@torch.inference_mode()
def merge_dcp_checkpoints(
    *,
    selected_checkpoints: list[StepCheckpoint],
    weights: list[float],
    output_dir: Path,
    template_state_dict: dict[str, torch.Tensor],
    accum_dtype: torch.dtype,
    export_dtype: torch.dtype,
    overwrite: bool = False,
    strict_non_floating: bool = False,
) -> None:
    if len(selected_checkpoints) != len(weights):
        raise ValueError(
            f"Expected one weight per checkpoint, got "
            f"{len(weights)} weights and {len(selected_checkpoints)} checkpoints"
        )
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    merged_state_dict: dict[str, torch.Tensor] = {}
    non_floating_reference: dict[str, torch.Tensor] = {}

    for checkpoint_index, (checkpoint, weight) in enumerate(
        zip(selected_checkpoints, weights)
    ):
        print(
            f"Loading source checkpoint {checkpoint_index + 1}/"
            f"{len(selected_checkpoints)}: step-{checkpoint.step}"
        )
        loaded_state_dict = load_model_state_dict(checkpoint.path, template_state_dict)
        _validate_loaded_state_dict(
            loaded_state_dict, template_state_dict, checkpoint.path
        )

        is_latest_source = checkpoint_index == len(selected_checkpoints) - 1
        for key, loaded_value in loaded_state_dict.items():
            if torch.is_floating_point(loaded_value):
                weighted_value = loaded_value.to(dtype=accum_dtype) * weight
                if key in merged_state_dict:
                    merged_state_dict[key].add_(weighted_value)
                else:
                    merged_state_dict[key] = weighted_value
            else:
                if strict_non_floating:
                    if key not in non_floating_reference:
                        non_floating_reference[key] = loaded_value.clone()
                    elif not torch.equal(non_floating_reference[key], loaded_value):
                        raise ValueError(
                            f"Non-floating tensor {key!r} differs across selected "
                            "checkpoints. Disable --strict-non-floating to copy the "
                            "latest value."
                        )
                if is_latest_source:
                    merged_state_dict[key] = loaded_value.clone()

        del loaded_state_dict
        gc.collect()

    for key, value in list(merged_state_dict.items()):
        if torch.is_floating_point(value) and value.dtype != export_dtype:
            merged_state_dict[key] = value.to(dtype=export_dtype)

    print(f"Saving merged DCP checkpoint to {output_dir}")
    dcp.save(merged_state_dict, checkpoint_id=str(output_dir))


def output_dir_for_merge(
    output_root: Path,
    *,
    end_step: int,
    method: MergeMethod,
    window_size: int,
) -> Path:
    return output_root / f"step-{end_step}_{method}_w{window_size}"


def _print_dry_run_plan(
    *,
    output_dir: Path,
    method: MergeMethod,
    window_size: int,
    selected_checkpoints: list[StepCheckpoint],
    weights: list[float],
    expected_interval: int | None,
) -> None:
    source_steps = [checkpoint.step for checkpoint in selected_checkpoints]
    source_intervals = [
        next_checkpoint.step - checkpoint.step
        for checkpoint, next_checkpoint in zip(
            selected_checkpoints, selected_checkpoints[1:]
        )
    ]
    print(f"{output_dir}:")
    print(f"  method: {method}")
    print(f"  window_size: {window_size}")
    print("  sources: " + ", ".join(f"step-{step}" for step in source_steps))
    print("  source_intervals: " + ", ".join(str(diff) for diff in source_intervals))
    print(
        "  nominal_interval: "
        + str(infer_nominal_interval(source_steps, expected_interval))
    )
    print("  weights: " + ", ".join(f"{weight:.10f}" for weight in weights))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge TorchTitan DCP checkpoints with WSM-style weights."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Directory containing step-* DCP checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where merged DCP checkpoints will be written.",
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        dest="model_name",
        required=True,
        help="TorchTitan model module name, e.g. rwkv_vl.",
    )
    parser.add_argument(
        "--model-flavor",
        "--model_flavor",
        dest="model_flavor",
        required=True,
        help="TorchTitan model flavor, e.g. 0.4B-v100M.",
    )
    parser.add_argument(
        "--end-step",
        default="latest",
        help="Step to end the merge window at, or 'latest'.",
    )
    parser.add_argument(
        "--window-sizes",
        default="4,8,12,16",
        help="Comma-separated merge window sizes.",
    )
    parser.add_argument(
        "--methods",
        default="mean,sqrt",
        help=(
            "Comma-separated methods: mean,linear,sqrt,cosine,ema. "
            "Decay-shaped methods use actual checkpoint step numbers."
        ),
    )
    parser.add_argument(
        "--expected-interval",
        type=int,
        default=None,
        help=(
            "Nominal spacing between selected checkpoint steps. Interior gaps "
            "must match exactly; the final gap may be shorter."
        ),
    )
    parser.add_argument(
        "--min-lr-factor",
        type=float,
        default=0.0,
        help="Minimum synthetic LR factor for theorem-derived methods.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9,
        help="EMA decay used only when --methods includes ema.",
    )
    parser.add_argument(
        "--accum-dtype",
        choices=sorted(TORCH_DTYPE_MAP),
        default="float32",
        help="Floating-point dtype used for accumulation.",
    )
    parser.add_argument(
        "--export-dtype",
        choices=sorted(TORCH_DTYPE_MAP),
        default="bfloat16",
        help="Floating-point dtype saved in merged checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected checkpoint windows and weights without loading or saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output directories.",
    )
    parser.add_argument(
        "--strict-non-floating",
        action="store_true",
        help="Require non-floating tensors to be identical across source checkpoints.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    end_step = parse_end_step(args.end_step)
    window_sizes = parse_csv_ints(args.window_sizes, name="--window-sizes")
    methods = parse_csv_methods(args.methods)
    checkpoints = discover_checkpoints(args.checkpoint_root)

    planned_merges: list[
        tuple[Path, MergeMethod, int, list[StepCheckpoint], list[float]]
    ] = []
    for window_size in window_sizes:
        selected_checkpoints = select_checkpoint_window(
            checkpoints,
            end_step=end_step,
            window_size=window_size,
        )
        validate_expected_interval(selected_checkpoints, args.expected_interval)
        concrete_end_step = selected_checkpoints[-1].step
        for method in methods:
            weights = compute_checkpoint_weights(
                method,
                [checkpoint.step for checkpoint in selected_checkpoints],
                expected_interval=args.expected_interval,
                min_lr_factor=args.min_lr_factor,
                ema_decay=args.ema_decay,
            )
            output_dir = output_dir_for_merge(
                args.output_root,
                end_step=concrete_end_step,
                method=method,
                window_size=window_size,
            )
            planned_merges.append(
                (output_dir, method, window_size, selected_checkpoints, weights)
            )

    for (
        output_dir,
        method,
        window_size,
        selected_checkpoints,
        weights,
    ) in planned_merges:
        _print_dry_run_plan(
            output_dir=output_dir,
            method=method,
            window_size=window_size,
            selected_checkpoints=selected_checkpoints,
            weights=weights,
            expected_interval=args.expected_interval,
        )

    if args.dry_run:
        return

    print(
        f"Building empty CPU state dict for {args.model_name} " f"{args.model_flavor}"
    )
    template_state_dict = build_empty_model_state_dict(
        args.model_name,
        args.model_flavor,
    )
    accum_dtype = TORCH_DTYPE_MAP[args.accum_dtype]
    export_dtype = TORCH_DTYPE_MAP[args.export_dtype]

    for (
        output_dir,
        method,
        window_size,
        selected_checkpoints,
        weights,
    ) in planned_merges:
        print(
            f"Merging {len(selected_checkpoints)} checkpoints with {method} "
            f"weights into {output_dir}"
        )
        merge_dcp_checkpoints(
            selected_checkpoints=selected_checkpoints,
            weights=weights,
            output_dir=output_dir,
            template_state_dict=template_state_dict,
            accum_dtype=accum_dtype,
            export_dtype=export_dtype,
            overwrite=args.overwrite,
            strict_non_floating=args.strict_non_floating,
        )
        metadata = _metadata_for_merge(
            args=args,
            method=method,
            window_size=window_size,
            selected_checkpoints=selected_checkpoints,
            weights=weights,
            output_dir=output_dir,
            accum_dtype=args.accum_dtype,
            export_dtype=args.export_dtype,
            expected_interval=args.expected_interval,
        )
        save_merge_metadata(output_dir, metadata)
        print(f"Wrote metadata to {output_dir / 'merge_metadata.json'}")

        gc.collect()


if __name__ == "__main__":
    main()
