"""Utility to train a tokenizer for the MVLM project.

This script scans the ``mvlm_training_dataset_complete`` directory for text
files and trains either a BPE or Unigram tokenizer using the ``tokenizers``
library. The resulting vocabulary files are written to
``tokenization/prioritary_tokenizer``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def iter_text_files(root: Path) -> Iterable[str]:
    """Yield all `.txt` files under *root* recursively."""
    for path, _, files in os.walk(root):
        for file in files:
            if file.endswith(".txt"):
                yield str(Path(path) / file)


def train(dataset_dir: Path, output_dir: Path, model_type: str, vocab_size: int) -> None:
    files = list(iter_text_files(dataset_dir))
    if not files:
        raise RuntimeError(f"no .txt files found under {dataset_dir}")

    if model_type == "unigram":
        model = Unigram()
        trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    else:
        model = BPE(unk_token="[UNK]")
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save model-specific files (vocab.json, merges.txt, etc.)
    tokenizer.model.save(str(output_dir))
    # Also save the combined tokenizer.json for easy loading
    tokenizer.save(str(output_dir / "tokenizer.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tokenizer for MVLM")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("mvlm_training_dataset_complete"),
        help="Directory containing training text files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenization/prioritary_tokenizer"),
        help="Where to store the trained tokenizer files",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Type of tokenizer model to train",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Size of the vocabulary",
    )
    args = parser.parse_args()

    train(args.dataset_dir, args.output_dir, args.model_type, args.vocab_size)


if __name__ == "__main__":
    main()
