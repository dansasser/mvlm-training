from transformers import GPT2Config, GPT2LMHeadModel

class PrioritaryMVLM(GPT2LMHeadModel):
    """Minimal MVLM model based on GPT-2."""

    def __init__(self, config: GPT2Config | None = None):
        if config is None:
            config = GPT2Config()
        super().__init__(config)
