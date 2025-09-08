from dataclasses import dataclass


@dataclass
class PrioritaryConfig:
    """Configuration for the Prioritary MVLM model and training.

    Attributes
    ----------
    vocab_size: int
        Size of the tokenizer vocabulary.
    n_layer: int
        Number of transformer layers.
    n_head: int
        Number of attention heads.
    n_embd: int
        Embedding dimensionality.
    batch_size: int
        Batch size used during training.
    max_length: int
        Maximum sequence length for training examples.
    stride: int
        Stride used when creating training windows from long texts.
    """

    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    batch_size: int = 8
    max_length: int = 512
    stride: int = 128
