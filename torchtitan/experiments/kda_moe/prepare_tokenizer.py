# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert the public Kimi K3 tokenizer for TorchTitan's tokenizer wrapper."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from tokenizers import AddedToken

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import TikTokenConverter

from torchtitan.components.tokenizer import HuggingFaceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/kda-moe-tokenizer-source/Kimi-K3"),
        help="Directory containing K3 tokenizer files, without model weights.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".tokenizer-staging-", dir=args.output_dir.parent)
    )

    reference = AutoTokenizer.from_pretrained(args.source_dir, trust_remote_code=True)
    if reference is None:
        raise ValueError(f"Could not load a tokenizer from {args.source_dir}")
    converter = TikTokenConverter(
        vocab_file=str(args.source_dir / "tiktoken.model"),
        pattern=reference.pat_str,
        add_prefix_space=False,
    )
    tokenizer = converter.converted()
    config = json.loads((args.source_dir / "tokenizer_config.json").read_text())
    special_tokens = [
        AddedToken(
            item["content"],
            single_word=item["single_word"],
            lstrip=item["lstrip"],
            rstrip=item["rstrip"],
            normalized=item["normalized"],
            special=item["special"],
        )
        for item in config["added_tokens_decoder"].values()
    ]
    tokenizer.add_tokens(special_tokens)
    tokenizer.save(str(staging_dir / "tokenizer.json"))
    for filename in (
        "tokenizer_config.json",
        "tiktoken.model",
        "tokenization_kimi.py",
        "encoding_k3.py",
    ):
        shutil.copy2(args.source_dir / filename, staging_dir / filename)

    local = HuggingFaceTokenizer(tokenizer_path=str(staging_dir))
    texts = [
        "Packing documents for language model training.",
        "[BOS][EOS] and special tokens",
        "UpperCASE lowerCase numbers 123456789 and punctuation!",
        "Chinese: \u4f60\u597d\uff0c\u4e16\u754c\u3002",
        "Whitespace:   two\n\nthree\t four",
    ]
    failures = []
    for index, text in enumerate(texts):
        expected = reference.encode(text)
        actual = local.encode(text, add_bos=False, add_eos=False)
        if expected != actual:
            failures.append(
                {
                    "index": index,
                    "characters": len(text),
                    "expected_prefix": expected[:40],
                    "actual_prefix": actual[:40],
                }
            )
    report = {
        "source_snapshot": str(args.source_dir),
        "destination": str(args.output_dir),
        "tokenizer_vocab_size": local.get_vocab_size(),
        "bos_id": local.bos_id,
        "eos_id": local.eos_id,
        "documents_compared": len(texts),
        "failures": failures,
        "generated_files": sorted(path.name for path in staging_dir.iterdir()),
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(f"Converted tokenizer mismatches: {len(failures)}")
    staging_dir.rename(args.output_dir)


if __name__ == "__main__":
    main()
