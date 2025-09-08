"""Tokenizer wrapper for Prioritary MVLM models.

This module re-exports the :class:`PrioritaryTokenizer` defined in the
``tokenization`` package.  The tokenizer is based on a pretrained
subword/BPE vocabulary stored in ``tokenization/prioritary_tokenizer`` and
provides a small interface compatible with the training utilities.
"""

from tokenization import PrioritaryTokenizer as PrioritaryTokenizer  # re-export

__all__ = ["PrioritaryTokenizer"]

