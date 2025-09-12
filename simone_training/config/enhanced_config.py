"""Enhanced SIM-ONE configuration - bridges to existing PrioritaryConfig."""

from dataclasses import dataclass
from .base_config import BaseConfig

# Import existing config for compatibility
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "SIM-ONE Training"))

try:
    from prioritary_mvlm.config import PrioritaryConfig
except ImportError:
    # Fallback if import fails
    PrioritaryConfig = None


@dataclass  
class EnhancedConfig(BaseConfig):
    """Enhanced SIM-ONE configuration with modern features."""
    
    # Enhanced model architecture
    hidden_dim: int = 768
    num_heads: int = 12
    ff_dim: int = 3072
    num_layers: int = 12
    
    # Modern training features
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 4
    
    # Enhanced features
    use_rope: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    use_flash_attention: bool = True
    
    # Biblical training specific
    biblical_alignment_weight: float = 0.1
    coherence_weight: float = 0.05
    
    # Tokenizer
    tokenizer_type: str = "bpe"  # "bpe" or "gpt2"
    bpe_vocab_size: int = 32000
    
    def to_prioritary_config(self) -> 'PrioritaryConfig':
        """Convert to existing PrioritaryConfig for compatibility."""
        if PrioritaryConfig is None:
            raise ImportError("Could not import PrioritaryConfig")
            
        config = PrioritaryConfig()
        
        # Map fields
        config.vocab_size = self.vocab_size
        config.hidden_dim = self.hidden_dim
        config.num_heads = self.num_heads
        config.ff_dim = self.ff_dim
        config.num_layers = self.num_layers
        config.batch_size = self.batch_size
        config.max_length = self.max_length
        config.learning_rate = self.learning_rate
        config.num_epochs = self.num_epochs
        config.warmup_steps = self.warmup_steps
        config.weight_decay = self.weight_decay
        config.max_grad_norm = self.max_grad_norm
        config.gradient_accumulation_steps = self.gradient_accumulation_steps
        config.log_interval = self.log_interval
        config.eval_interval = self.eval_interval
        
        return config
    
    @classmethod
    def from_prioritary_config(cls, pconfig: 'PrioritaryConfig'):
        """Create EnhancedConfig from existing PrioritaryConfig."""
        return cls(
            vocab_size=pconfig.vocab_size,
            hidden_dim=pconfig.hidden_dim,
            num_heads=pconfig.num_heads,
            ff_dim=pconfig.ff_dim,
            num_layers=pconfig.num_layers,
            batch_size=pconfig.batch_size,
            max_length=pconfig.max_length,
            learning_rate=pconfig.learning_rate,
            num_epochs=pconfig.num_epochs,
            warmup_steps=pconfig.warmup_steps,
            weight_decay=pconfig.weight_decay,
            max_grad_norm=pconfig.max_grad_norm,
            gradient_accumulation_steps=pconfig.gradient_accumulation_steps,
            log_interval=pconfig.log_interval,
            eval_interval=pconfig.eval_interval
        )