"""Tokenization utilities for the MVLM project.

This package provides the :class:`PrioritaryTokenizer` which wraps a
pretrained tokenizer saved under ``tokenization/prioritary_tokenizer``.
"""

from pathlib import Path
from tokenizers import Tokenizer


class PrioritaryTokenizer:
    """Simple wrapper around a tokenizer trained for MVLM.

    This wrapper exposes a small, :class:`transformers`-like interface so it can
    be used by the training utilities without pulling in the full Transformers
    dependency.  Special token IDs follow the convention used when the
    tokenizer was trained: ``[PAD]``=0, ``[UNK]``=1, ``[CLS]``=2 and
    ``[SEP]``=3.

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

        # Special token identifiers
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def __len__(self) -> int:
        """Return the size of the tokenizer vocabulary."""
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Tokenize *text* and return a list of token ids."""
        ids = self._tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token *ids* back into text."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


__all__ = ["PrioritaryTokenizer"]
