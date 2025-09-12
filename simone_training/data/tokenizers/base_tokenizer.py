"""Base tokenizer - handles GPT-2 and other standard tokenizers."""

from transformers import GPT2Tokenizer
from typing import List, Dict, Any


class BaseTokenizer:
    """Base tokenizer wrapper for standard tokenizers."""
    
    def __init__(self, tokenizer_type: str = "gpt2", **kwargs):
        """Initialize tokenizer.
        
        Args:
            tokenizer_type: Type of tokenizer ('gpt2', etc.)
            **kwargs: Additional arguments for tokenizer
        """
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", **kwargs)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def __call__(self, text, **kwargs):
        """Make tokenizer callable."""
        return self.tokenizer(text, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def save_pretrained(self, path: str):
        """Save tokenizer."""
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load tokenizer."""
        instance = cls.__new__(cls)
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path, **kwargs)
        instance.tokenizer_type = "gpt2"  # Assume GPT-2 for loaded tokenizers
        return instance