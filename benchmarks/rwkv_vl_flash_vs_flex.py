from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import current_flash_attention_impl
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchtitan.models.common.attention import (
    build_varlen_metadata,
    configure_flash_attention_backend,
    flash_attention_varlen,
)

FLEX_COMPILE_OPTIONS = {
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": False,
    "assume_aligned_inputs": True,
}

VISION_KERNEL_OPTIONS = {
    "USE_TMA": True,
    "ROWS_GUARANTEED_SAFE": False,
    "IS_DIVISIBLE": True,
    "fwd_BLOCK_M": 64,
    "fwd_BLOCK_N": 64,
    "fwd_num_stages": 3,
    "fwd_num_warps": 4,
}

PROJECTOR_KERNEL_OPTIONS = {
    "USE_TMA": True,
    "ROWS_GUARANTEED_SAFE": False,
    "IS_DIVISIBLE": True,
}


@dataclass(frozen=True)
class Measurement:
    case: str
    mode: str
    implementation: str
    median_ms: float
    minimum_ms: float
    maximum_ms: float


_clock_warmup_input: torch.Tensor | None = None
_clock_warmup_output: torch.Tensor | None = None


def _warm_device_clock() -> None:
    global _clock_warmup_input, _clock_warmup_output
    if _clock_warmup_input is None:
        _clock_warmup_input = torch.randn(
            4096, 4096, device="cuda", dtype=torch.bfloat16
        )
        _clock_warmup_output = torch.empty_like(_clock_warmup_input)
    for _ in range(100):
        torch.mm(
            _clock_warmup_input,
            _clock_warmup_input,
            out=_clock_warmup_output,
        )
    torch.cuda.synchronize()


def _next_pow2_bucket(length: int) -> int:
    if length <= 128:
        return 128
    return 1 << (length - 1).bit_length()


def _measure(
    case: str,
    mode: str,
    implementation: str,
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> Measurement:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    _warm_device_clock()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return Measurement(
        case=case,
        mode=mode,
        implementation=implementation,
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
    )


def _make_vision_mask(lengths: list[int], bucket: int, device: torch.device):
    item_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32, device=device),
        torch.tensor(lengths, device=device),
    )
    item_ids = F.pad(item_ids, (0, bucket - item_ids.numel()), value=-1)

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        query_item = item_ids[query_index]
        key_item = item_ids[key_index]
        valid_query = query_item >= 0
        valid_key = key_item >= 0
        padding_self = (~valid_query) & (~valid_key) & (query_index == key_index)
        return (valid_query & valid_key & (query_item == key_item)) | padding_self

    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=bucket,
        KV_LEN=bucket,
        device=device,
    )


def _make_projector_mask(
    query_lengths: list[int],
    key_lengths: list[int],
    query_bucket: int,
    key_bucket: int,
    device: torch.device,
):
    query_ids = torch.repeat_interleave(
        torch.arange(len(query_lengths), dtype=torch.int32, device=device),
        torch.tensor(query_lengths, device=device),
    )
    query_ids = F.pad(query_ids, (0, query_bucket - query_ids.numel()), value=-1)
    key_ids = torch.repeat_interleave(
        torch.arange(len(key_lengths), dtype=torch.int32, device=device),
        torch.tensor(key_lengths, device=device),
    )
    real_key_length = key_ids.numel()
    key_ids = F.pad(key_ids, (0, key_bucket - real_key_length), value=-1)

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        query_item = query_ids[query_index]
        key_item = key_ids[key_index]
        valid_query = query_item >= 0
        valid_key = key_item >= 0
        same_image = query_item == key_item
        padding_dummy = (~valid_query) & (key_index == real_key_length)
        return (valid_query & valid_key & same_image) | padding_dummy

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=query_bucket,
        KV_LEN=key_bucket,
        device=device,
    )


def _benchmark_pair(
    case: str,
    flex_forward: Callable[[], torch.Tensor],
    flash_forward: Callable[[], torch.Tensor],
    flex_training_step: Callable[[], object],
    flash_training_step: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[Measurement]:
    return [
        _measure(
            case,
            "forward",
            "flex",
            flex_forward,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        _measure(
            case,
            "forward",
            "flash",
            flash_forward,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        _measure(
            case,
            "forward+backward",
            "flex",
            flex_training_step,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        _measure(
            case,
            "forward+backward",
            "flash",
            flash_training_step,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    ]


def benchmark_vision(
    compiled_flex: Callable,
    lengths: list[int],
    bucket: int,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[Measurement]:
    heads = 16
    head_dim = 64
    real_length = sum(lengths)
    generator = torch.Generator(device=device).manual_seed(real_length)
    query = torch.randn(
        1,
        bucket,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn_like(query, generator=generator)
    value = torch.randn_like(query, generator=generator)
    indices = torch.arange(real_length, device=device)
    metadata = build_varlen_metadata(torch.tensor(lengths, device=device))
    block_mask = _make_vision_mask(lengths, bucket, device)

    def flex_path(q, k, v):
        return compiled_flex(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            kernel_options=VISION_KERNEL_OPTIONS,
        ).transpose(1, 2)

    def flash_path(q, k, v):
        flat_query = q.flatten(0, 1)
        flat_key = k.flatten(0, 1)
        flat_value = v.flatten(0, 1)
        if real_length == bucket:
            packed_query = flat_query
            packed_key = flat_key
            packed_value = flat_value
        else:
            packed_query = flat_query[:real_length]
            packed_key = flat_key[:real_length]
            packed_value = flat_value[:real_length]
        packed = flash_attention_varlen(
            packed_query,
            packed_key,
            packed_value,
            metadata,
        )
        if real_length == bucket:
            return packed.unsqueeze(0)
        return F.pad(packed, (0, 0, 0, 0, 0, bucket - real_length)).unsqueeze(0)

    packed_query = query.flatten(0, 1).index_select(0, indices)
    packed_key = key.flatten(0, 1).index_select(0, indices)
    packed_value = value.flatten(0, 1).index_select(0, indices)

    def flash_core(q, k, v):
        return flash_attention_varlen(q, k, v, metadata)

    with torch.no_grad():
        flex_actual = flex_path(query, key, value)
        flash_actual = flash_path(query, key, value)
        torch.testing.assert_close(
            flex_actual[:, :real_length],
            flash_actual[:, :real_length],
            rtol=3e-2,
            atol=3e-2,
        )

    train_query = query.detach().requires_grad_(True)
    train_key = key.detach().requires_grad_(True)
    train_value = value.detach().requires_grad_(True)
    gradient = torch.randn_like(query, generator=generator)

    def flex_training_step():
        output = flex_path(train_query, train_key, train_value)
        return torch.autograd.grad(
            output,
            (train_query, train_key, train_value),
            gradient,
        )

    def flash_training_step():
        output = flash_path(train_query, train_key, train_value)
        return torch.autograd.grad(
            output,
            (train_query, train_key, train_value),
            gradient,
        )

    train_packed_query = packed_query.detach().requires_grad_(True)
    train_packed_key = packed_key.detach().requires_grad_(True)
    train_packed_value = packed_value.detach().requires_grad_(True)
    packed_gradient = gradient.flatten(0, 1).index_select(0, indices)

    def flash_core_training_step():
        output = flash_core(train_packed_query, train_packed_key, train_packed_value)
        return torch.autograd.grad(
            output,
            (train_packed_query, train_packed_key, train_packed_value),
            packed_gradient,
        )

    case = f"vision_{real_length}_of_{bucket}_patches"
    measurements = _benchmark_pair(
        case,
        lambda: flex_path(query, key, value),
        lambda: flash_path(query, key, value),
        flex_training_step,
        flash_training_step,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    measurements.extend(
        [
            _measure(
                case,
                "forward",
                "flash_core",
                lambda: flash_core(packed_query, packed_key, packed_value),
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            ),
            _measure(
                case,
                "forward+backward",
                "flash_core",
                flash_core_training_step,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            ),
        ]
    )
    return measurements


def benchmark_projector(
    compiled_flex: Callable,
    query_lengths: list[int],
    key_lengths: list[int],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[Measurement]:
    query_heads = 16
    key_heads = 8
    head_dim = 64
    query_length = sum(query_lengths)
    key_length = sum(key_lengths)
    query_bucket = _next_pow2_bucket(query_length)
    key_bucket = _next_pow2_bucket(key_length + 1)
    generator = torch.Generator(device=device).manual_seed(key_length)
    query = torch.randn(
        query_length,
        query_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        key_length,
        key_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn_like(key, generator=generator)
    metadata = build_varlen_metadata(
        torch.tensor(query_lengths, device=device),
        torch.tensor(key_lengths, device=device),
    )
    block_mask = _make_projector_mask(
        query_lengths,
        key_lengths,
        query_bucket,
        key_bucket,
        device,
    )

    def flex_path(q, k, v):
        padded_query = F.pad(q, (0, 0, 0, 0, 0, query_bucket - query_length))
        padded_key = F.pad(k, (0, 0, 0, 0, 0, key_bucket - key_length))
        padded_value = F.pad(v, (0, 0, 0, 0, 0, key_bucket - key_length))
        output = compiled_flex(
            padded_query.transpose(0, 1).unsqueeze(0),
            padded_key.transpose(0, 1).unsqueeze(0),
            padded_value.transpose(0, 1).unsqueeze(0),
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=PROJECTOR_KERNEL_OPTIONS,
        )
        return output[0, :, :query_length].transpose(0, 1)

    def flash_path(q, k, v):
        return flash_attention_varlen(q, k, v, metadata, enable_gqa=True)

    with torch.no_grad():
        torch.testing.assert_close(
            flex_path(query, key, value),
            flash_path(query, key, value),
            rtol=3e-2,
            atol=3e-2,
        )

    train_query = query.detach().requires_grad_(True)
    train_key = key.detach().requires_grad_(True)
    train_value = value.detach().requires_grad_(True)
    gradient = torch.randn_like(query, generator=generator)

    def flex_training_step():
        output = flex_path(train_query, train_key, train_value)
        return torch.autograd.grad(
            output,
            (train_query, train_key, train_value),
            gradient,
        )

    def flash_training_step():
        output = flash_path(train_query, train_key, train_value)
        return torch.autograd.grad(
            output,
            (train_query, train_key, train_value),
            gradient,
        )

    case = (
        f"projector_{len(query_lengths)}_images_"
        f"q{query_length}_kv{key_length}_buckets{query_bucket}x{key_bucket}"
    )
    return _benchmark_pair(
        case,
        lambda: flex_path(query, key, value),
        lambda: flash_path(query, key, value),
        flex_training_step,
        flash_training_step,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )


def _print_results(measurements: list[Measurement]) -> None:
    print("\ncase,mode,implementation,median_ms,min_ms,max_ms,speedup_vs_flex")
    flex_times = {
        (measurement.case, measurement.mode): measurement.median_ms
        for measurement in measurements
        if measurement.implementation == "flex"
    }
    for measurement in measurements:
        speedup = (
            flex_times[(measurement.case, measurement.mode)] / measurement.median_ms
        )
        print(
            f"{measurement.case},{measurement.mode},{measurement.implementation},"
            f"{measurement.median_ms:.4f},{measurement.minimum_ms:.4f},"
            f"{measurement.maximum_ms:.4f},{speedup:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    configure_flash_attention_backend()
    compiled_flex = torch.compile(
        flex_attention,
        dynamic=False,
        options=FLEX_COMPILE_OPTIONS,
    )
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"capability={torch.cuda.get_device_capability(device)}")
    print(f"torch={torch.__version__}")
    print(f"flash_impl={current_flash_attention_impl() or 'native/default'}")
    print(f"warmup={args.warmup} iterations={args.iterations} repeats={args.repeats}")

    measurements = []
    measurements.extend(
        benchmark_vision(
            compiled_flex,
            [1024] * 8,
            32768,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    )
    measurements.extend(
        benchmark_vision(
            compiled_flex,
            [2048] * 16,
            32768,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    )
    measurements.extend(
        benchmark_projector(
            compiled_flex,
            [128] * 8,
            [2048] * 8,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    )
    _print_results(measurements)


if __name__ == "__main__":
    main()
