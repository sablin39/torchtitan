# Pure KDA MoE

This experiment combines KDA layers from Kimi K3 with the MobileMoE geometry
from [arXiv:2605.27358](https://arxiv.org/html/2605.27358). Every block is an
ordinary pre-norm residual block containing KDA followed by MoE. The model has
no MLA, vision encoder, LatentMoE, dense FFN-only layer, or K3 attention
residuals.

## Architecture

| Flavor | Model dim | Layers | KDA heads | Vocabulary | Experts / top-k | Expert hidden | Shared hidden | Total / active parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `debugmodel` | 256 | 3 | 2 | 2,048 | 8 / 2 | 128 | 512 | - |
| `S` | 768 | 20 | 6 | 163,840 | 60 / 4 | 384 | 3,072 | 1.393B / 402M |
| `M` | 1,024 | 26 | 8 | 163,840 | 60 / 4 | 512 | 4,096 | 3.094B / 804M |
| `L` | 1,280 | 32 | 10 | 163,840 | 60 / 4 | 640 | 5,120 | 5.834B / 1.430B |

The paper's S/M/L model widths and depths are preserved. Its dense-equivalent
FFN widths are 3,072/4,096/5,120. The shared expert replaces four `g=8`
segments, so its width is eight times a fine-grained routed expert's width.
Parameter totals differ from MobileMoE because KDA has different projections
and the vocabulary follows K3 rather than Llama 3.

KDA uses head width 128, convolution width 4, and projection width equal to the
model width. K3's RMSNorm, SiTU-GLU (`beta=4`, `linear_beta=25`), initialization,
tied embeddings, and 163,840-token vocabulary are retained. Every routed MoE
uses K3's quantile-balanced sigmoid router with normalized top-k scores, scale
1, 1,000 bins, and a persistent quantile bias. There is no sign-based balance
coefficient.

Use `torchtitan.experiments.kda_moe.build_model_config(flavor,
parallelism=..., seq_len=...)`. The default document segment length is 2,048;
the supported maximum follows K3 at 262,144. TP must divide the KDA head count
and shared expert width; EP must divide the expert count. The model registers
the quantile balancing optimizer pre-hook.

Training uses K3's full-BF16 `PrecisionConfig(dtype="bfloat16")`: BF16
parameters, gradients, and optimizer states, with FP32 reductions and logits.
Optimizer defaults follow the paper: AdamW betas 0.9/0.95, epsilon 1e-15,
weight decay 0.1, learning rate 4e-4, 8,000 warmup steps, cosine decay, and
minimum LR ratio 0.1.

## cuDNN and tokenizer setup

[cuDNN frontend v1.33.0](https://github.com/NVIDIA/cudnn-frontend/tree/v1.33.0)
provides the only KDA recurrence on SM100/SM103. `CuDNNKDA` is a sibling
`Module` behind K3's `InnerKDA` convolution and metadata plumbing; its config
retains K3's gate-bound validation. Built-in Q/K L2 normalization, beta
sigmoid, safe gate, FP32 gate parameters, and query scaling are enabled. Odd
ragged document offsets and repeated empty-document entries are passed directly
to cuDNN without padding or sequential document scans. Install the pinned
frontend with:

```bash
uv pip install -r .ci/docker/requirements-kda-moe-cudnn.txt
# or install the project extra after reinstalling:
uv pip install -e '.[kda-moe-cudnn]'
```

The existing packed BF16 Attention Gym convolution is retained because
cuDNN's separate packed convolution API rejects the active-prefix and
repeated-offset metadata used here. No `nn.Conv1d` call or alternate recurrence
is used.

The official K3 assets provide a slow TikToken tokenizer rather than the
`tokenizer.json` required by TorchTitan. The converted tokenizer and source
files are already installed in `assets/hf/Kimi-K3`; no model weights were
downloaded. Conversion matched the official tokenizer on 105 strings/documents,
including 100 FineWeb documents.

To regenerate it, download only the tokenizer sources and convert them:

```bash
python scripts/download_hf_assets.py \
  --repo_id moonshotai/Kimi-K3 \
  --local_dir outputs/kda-moe-tokenizer-source \
  --additional_patterns tokenizer_config.json tiktoken.model tokenization_kimi.py encoding_k3.py

python -m torchtitan.experiments.kda_moe.prepare_tokenizer \
  --source-dir outputs/kda-moe-tokenizer-source/Kimi-K3 \
  --output-dir assets/hf/Kimi-K3
```

The converter refuses to overwrite an existing destination. It validates five
strings in a temporary directory, prints the result, and publishes the
converted files only after validation succeeds.

## Training

`torchtitan_recipes/models/kda_moe.py` provides S/M/L recipes using C4 and
`assets/hf/Kimi-K3`, plus a debug recipe using bundled C4 test data and its
tokenizer. The default is 32,768 tokens per rank. `seq_len` bounds document
segments; `num_tokens_per_microbatch_per_dp_rank` independently controls the
physical packed-row capacity. A host-specific Python recipe can replace the
returned dataset or tokenizer path without proprietary paths or KDA-specific
environment variables in this repository.

```bash
source .venv/bin/activate
NGPU=1 MODULE=torchtitan_recipes.models.kda_moe CONFIG=kda_moe_debugmodel ./run_train.sh
NGPU=8 MODULE=torchtitan_recipes.models.kda_moe CONFIG=kda_moe_s ./run_train.sh
```

`kda_moe_m` and `kda_moe_l` use the same launch form. Use `--output-dir` for
run-specific output placement. A local Python recipe can use `partial` with
`kda_moe_s`, including `steps`, `seq_len`,
`num_tokens_per_microbatch_per_dp_rank`, `max_num_documents`, and
`loss_num_chunks`, and select that factory with `MODULE` and `CONFIG`.

The CPU test exercises the quantile-balancing optimizer hook and tied weights.
The GPU test passes seven odd-length documents and four repeated terminal
offsets through cuDNN forward/backward and checks finite gradients:

```bash
.venv/bin/pytest -q torchtitan/experiments/kda_moe/tests/test_kda_moe.py
```
