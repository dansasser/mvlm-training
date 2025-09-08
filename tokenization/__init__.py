"""Tokenization utilities for the MVLM project.

This package provides the :class:`PrioritaryTokenizer` which wraps a
pretrained tokenizer saved under ``tokenization/prioritary_tokenizer``.
"""

from pathlib import Path
from tokenizers import Tokenizer


class PrioritaryTokenizer:
    """Simple wrapper around a tokenizer trained for MVLM.

    Parameters
    ----------
    tokenizer_path: str or Path, optional
        Path to a ``tokenizer.json`` file. By default this uses the tokenizer
        shipped with the repository under ``tokenization/prioritary_tokenizer``.
    """

    def __init__(self, tokenizer_path: Path | str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        if tokenizer_path is None:
            tokenizer_path = base_dir / "prioritary_tokenizer" / "tokenizer.json"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        """Tokenize *text* and return a list of token ids."""
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token *ids* back into text."""
        return self._tokenizer.decode(ids)


__all__ = ["PrioritaryTokenizer"]
