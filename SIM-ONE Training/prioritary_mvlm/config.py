from dataclasses import dataclass


@dataclass
class PrioritaryConfig:
    """Configuration for the SIM-ONE transformer and training.

    Attributes
    ----------
    vocab_size: int
        Size of the tokenizer vocabulary.
    hidden_dim: int
        Dimension of transformer embeddings.
    num_heads: int
        Number of attention heads.
    ff_dim: int
        Dimension of feedforward layer.
    num_layers: int
        Number of transformer blocks.
    batch_size: int
        Batch size used during training.
    max_length: int
        Maximum sequence length.
    stride: int
        Stride when creating training windows.
    """

    vocab_size: int = 8000
    hidden_dim: int = 512
    num_heads: int = 8
    ff_dim: int = 2048
    num_layers: int = 6
    batch_size: int = 8
    max_length: int = 512
    stride: int = 128
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 100
    lambda_policy: float = 1.0
    lambda_memory: float = 1.0
    lambda_energy: float = 1.0
