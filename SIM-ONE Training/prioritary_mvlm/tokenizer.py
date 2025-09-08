"""Subword tokenizer for SIM-ONE models.

This module re-uses the repository's shared :class:`PrioritaryTokenizer`
implementation which is based on a pretrained BPE tokenizer stored under
``tokenization/prioritary_tokenizer``.  The tokenizer provides special token
IDs and an interface compatible with the training utilities.
"""

from tokenization import PrioritaryTokenizer as PrioritaryTokenizer  # re-export

__all__ = ["PrioritaryTokenizer"]

