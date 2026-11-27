# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA MoE supervised-training recipes."""

from functools import partial

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import DebugConfig, PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.experiments.kda_moe import build_model_config
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


_DEFAULT_SEQ_LEN = 2048
_DEFAULT_SEQS_PER_MICROBATCH = 16


def _document_capacity(num_tokens_per_microbatch_per_dp_rank: int) -> int:
    # 64 slots fills sampled 32,768-token rows; 96 provides headroom for short docs.
    return max(16, round(num_tokens_per_microbatch_per_dp_rank * 3 / 1024))


def _kda_moe(
    flavor: str,
    *,
    seq_len: int = _DEFAULT_SEQ_LEN,
    microbatch_seq_multiplier: int = _DEFAULT_SEQS_PER_MICROBATCH,
    num_tokens_per_microbatch_per_dp_rank: int | None = None,
    max_num_documents: int | None = None,
    parallelism: ParallelismConfig | None = None,
    steps: int | None = None,
    dump_folder: str | None = None,
    loss_num_chunks: int = 8,
    _debug: bool = False,
) -> Trainer.Config:
    """Build one recipe with an independently tunable local token budget.

    Production flavors default to eight-rank FSDP without expert parallelism.
    Sixteen 2048-token segments per rank give 262144 tokens per optimizer step.
    EP4 is available through ``parallelism`` but is slower on the eight-L20D host.
    ``max_num_documents`` defaults to 96 document segments per 32768 local
    tokens; increasing it avoids early row closure for unusually short
    documents but increases varlen metadata size. It does not limit dataset
    reads. Host-specific recipes may replace the returned dataloader dataset
    and tokenizer path. The default uses C4 and ``assets/hf/Kimi-K3``.
    """
    if parallelism is None:
        parallelism = ParallelismConfig(data_parallel_shard_degree=8)
    model_config = build_model_config(
        flavor,
        seq_len=seq_len,
        parallelism=parallelism,
    )
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive when provided")
    if num_tokens_per_microbatch_per_dp_rank is None:
        if (
            isinstance(microbatch_seq_multiplier, bool)
            or not isinstance(microbatch_seq_multiplier, int)
            or microbatch_seq_multiplier <= 0
        ):
            raise ValueError("microbatch_seq_multiplier must be a positive integer")
        num_tokens_per_microbatch_per_dp_rank = (
            microbatch_seq_multiplier * model_config.max_context_length
        )
    elif num_tokens_per_microbatch_per_dp_rank <= 0:
        raise ValueError("num_tokens_per_microbatch_per_dp_rank must be greater than 0")
    if max_num_documents is None:
        max_num_documents = _document_capacity(num_tokens_per_microbatch_per_dp_rank)

    dataset = DATASETS["c4_test"] if _debug else DATASETS["c4"]
    hf_assets_path = "./tests/assets/tokenizer" if _debug else "./assets/hf/Kimi-K3"

    return Trainer.Config(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            num_chunks=loss_num_chunks,
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        precision=PrecisionConfig(dtype="bfloat16"),
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=GrainDataLoader.Config(
            max_num_documents=max_num_documents,
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            shuffle=not _debug,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=4e-4,
                        betas=(0.9, 0.95),
                        eps=1e-15,
                        weight_decay=0.1,
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=8000,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=(
                num_tokens_per_microbatch_per_dp_rank
            ),
            num_tokens_per_train_step=(
                num_tokens_per_microbatch_per_dp_rank
                * parallelism.data_parallel_replicate_degree
                * parallelism.data_parallel_shard_degree
                if parallelism.pipeline_parallel_degree == 1
                and parallelism.data_parallel_shard_degree > 0
                else -1
            ),
            steps=steps if steps is not None else (10 if _debug else 10000),
            # Token dispatch synchronizes with the host, including without EP.
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        activation_remat=FullActivationRematConfig(),
        checkpointer=None,
        debug=DebugConfig(seed=42),
        dump_folder=(
            dump_folder
            if dump_folder is not None
            else f"outputs/kda_moe_{flavor.lower()}"
        ),
    )


kda_moe_s = partial(_kda_moe, "S")
kda_moe_m = partial(_kda_moe, "M")
kda_moe_l = partial(_kda_moe, "L")


def kda_moe_debugmodel(
    seq_len: int = _DEFAULT_SEQ_LEN,
    *,
    microbatch_seq_multiplier: int = 1,
    num_tokens_per_microbatch_per_dp_rank: int | None = None,
    max_num_documents: int | None = None,
    parallelism: ParallelismConfig | None = None,
    steps: int | None = None,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig(data_parallel_shard_degree=1)
    return _kda_moe(
        "debugmodel",
        seq_len=seq_len,
        microbatch_seq_multiplier=microbatch_seq_multiplier,
        num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
        max_num_documents=max_num_documents,
        parallelism=parallelism,
        steps=steps,
        _debug=True,
    )
