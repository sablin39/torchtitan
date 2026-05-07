#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward/backward parity bench for FLA CP in the TorchTitan RWKV7 backend.

Run with:
  torchrun --nproc_per_node 2 benchmarks/rwkv7_cp_parity.py
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from torchtitan.models.rwkv7.model import rwkv7_causal_lm_config


def _init_dist() -> tuple[int, int]:
    if "RANK" not in os.environ:
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


def _loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().square().mean()


def _allreduce_grads(model) -> None:
    if not dist.is_initialized():
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
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
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    rank, world_size = _init_dist()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLA RWKV7 kernels")

    total_tokens = args.batch_size * args.seq_len
    if total_tokens % world_size != 0:
        raise ValueError("batch_size * seq_len must be divisible by world size")

    torch.manual_seed(2026)
    ref = _make_model(args, device, dtype)
    cp = _make_model(args, device, dtype)
    cp.load_state_dict(ref.state_dict())
    if dist.is_initialized():
        cp.set_cp_process_group(dist.group.WORLD)

    tokens = torch.randint(0, args.vocab_size, (1, total_tokens), device=device)
    cu_seqlens_global = torch.arange(
        0,
        total_tokens + 1,
        args.seq_len,
        dtype=torch.long,
        device=device,
    )
    part = total_tokens // world_size
    local_tokens = tokens[:, rank * part : (rank + 1) * part].contiguous()

    ref_logits = ref(tokens, cu_seqlens_global=cu_seqlens_global)
    ref_local_logits = ref_logits[:, rank * part : (rank + 1) * part].detach()
    cp_logits = cp(local_tokens, cu_seqlens_global=cu_seqlens_global)

    max_forward_diff = (ref_local_logits - cp_logits.detach()).abs().max()
    if dist.is_initialized():
        dist.all_reduce(max_forward_diff, op=dist.ReduceOp.MAX)

    ref.zero_grad(set_to_none=True)
    cp.zero_grad(set_to_none=True)
    ref_loss = _loss_from_logits(ref_logits)
    cp_loss = _loss_from_logits(cp_logits) / world_size
    ref_loss.backward()
    cp_loss.backward()
    _allreduce_grads(cp)

    max_grad_diff = torch.tensor(0.0, device=device)
    for (_, ref_param), (_, cp_param) in zip(ref.named_parameters(), cp.named_parameters()):
        if ref_param.grad is None or cp_param.grad is None:
            continue
        max_grad_diff = torch.maximum(
            max_grad_diff,
            (ref_param.grad - cp_param.grad).abs().max(),
        )
    if dist.is_initialized():
        dist.all_reduce(max_grad_diff, op=dist.ReduceOp.MAX)

    ok_forward = bool(torch.allclose(ref_local_logits, cp_logits, atol=args.atol, rtol=args.rtol))
    ok_grad = bool(max_grad_diff.item() <= args.atol)
    ok = ok_forward and ok_grad
    if rank == 0:
        print(f"max_forward_diff={max_forward_diff.item():.6e}")
        print(f"max_grad_diff={max_grad_diff.item():.6e}")
        print(f"status={'PASS' if ok else 'FAIL'}")
    if dist.is_initialized():
        dist.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
