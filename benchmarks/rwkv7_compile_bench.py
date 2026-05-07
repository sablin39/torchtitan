#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark eager vs torch.compile for the TorchTitan RWKV7 backend.

Examples:
  python benchmarks/rwkv7_compile_bench.py --device cuda --dtype bfloat16
  torchrun --nproc_per_node 2 benchmarks/rwkv7_compile_bench.py --cp
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from torchtitan.models.rwkv7.model import rwkv7_causal_lm_config


def _dist_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_dist() -> tuple[int, int]:
    if not _dist_enabled():
        return 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def _make_model(args, device: torch.device, dtype: torch.dtype):
    config = rwkv7_causal_lm_config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_heads=args.heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        a_low_rank_dim=args.a_low_rank_dim,
        decay_low_rank_dim=args.decay_low_rank_dim,
        gate_low_rank_dim=args.gate_low_rank_dim,
        v_low_rank_dim=args.v_low_rank_dim,
        chunk_size=args.chunk_size,
    )
    model = config.build().to(device=device, dtype=dtype)
    model.init_states()
    return model


def _compile_blocks(model, backend: str) -> None:
    for block in model.llm.layers.values():
        block.compile(backend=backend, fullgraph=False)


def _step(model, tokens, cu_seqlens_global=None) -> torch.Tensor:
    logits = model(tokens, cu_seqlens_global=cu_seqlens_global)
    loss = logits.float().square().mean()
    loss.backward()
    return loss.detach()


def _time_steps(model, tokens, cu_seqlens_global, iters: int) -> tuple[float, float]:
    if tokens.device.type == "cuda":
        torch.cuda.synchronize(tokens.device)
    start = time.perf_counter()
    loss = None
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        loss = _step(model, tokens, cu_seqlens_global)
    if tokens.device.type == "cuda":
        torch.cuda.synchronize(tokens.device)
    elapsed = (time.perf_counter() - start) / iters
    assert loss is not None
    return elapsed, float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--a-low-rank-dim", type=int, default=32)
    parser.add_argument("--decay-low-rank-dim", type=int, default=32)
    parser.add_argument("--gate-low-rank-dim", type=int, default=64)
    parser.add_argument("--v-low-rank-dim", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    rank, world_size = _init_dist()
    if args.cp and world_size == 1:
        raise RuntimeError("--cp requires torchrun with WORLD_SIZE > 1")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLA RWKV7 kernels")

    torch.manual_seed(1234)
    total_tokens = args.batch_size * args.seq_len
    if args.cp and total_tokens % world_size != 0:
        raise ValueError("batch_size * seq_len must be divisible by CP world size")

    global_tokens = torch.randint(
        0,
        args.vocab_size,
        (1 if args.cp else args.batch_size, total_tokens if args.cp else args.seq_len),
        device=device,
    )
    cu_seqlens_global = None
    if args.cp:
        cu_seqlens_global = torch.arange(
            0,
            total_tokens + 1,
            args.seq_len,
            dtype=torch.long,
            device=device,
        )
        part = total_tokens // world_size
        tokens = global_tokens[:, rank * part : (rank + 1) * part].contiguous()
    else:
        tokens = global_tokens

    eager = _make_model(args, device, dtype)
    compiled = _make_model(args, device, dtype)
    compiled.load_state_dict(eager.state_dict())
    if args.cp:
        eager.set_cp_process_group(dist.group.WORLD)
        compiled.set_cp_process_group(dist.group.WORLD)
    _compile_blocks(compiled, args.backend)

    for _ in range(args.warmup):
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        _step(eager, tokens, cu_seqlens_global)
        _step(compiled, tokens, cu_seqlens_global)

    eager_time, eager_loss = _time_steps(eager, tokens, cu_seqlens_global, args.iters)
    compiled_time, compiled_loss = _time_steps(
        compiled, tokens, cu_seqlens_global, args.iters
    )

    speedup = eager_time / compiled_time
    if rank == 0:
        print(f"eager:    {eager_time * 1000:.3f} ms/step loss={eager_loss:.6f}")
        print(f"compiled: {compiled_time * 1000:.3f} ms/step loss={compiled_loss:.6f}")
        print(f"speedup:  {speedup:.3f}x")
        print(f"loss_abs_diff: {abs(eager_loss - compiled_loss):.6e}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
