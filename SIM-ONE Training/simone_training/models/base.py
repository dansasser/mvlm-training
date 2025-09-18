"""
Base model adapters for Enhanced SIM-ONE integration.
Provides working wrappers to replace NotImplementedError stubs.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union, Tuple
import logging

from simone_transformer import EnhancedSIMONEModel
from prioritary_mvlm.config import PrioritaryConfig, PropheticSingularityState

logger = logging.getLogger(__name__)


class MVLMAdapter(nn.Module):
    """
    Working MVLM adapter wrapper for Enhanced SIM-ONE integration.
    Provides compatibility between MVLM interfaces and Enhanced SIM-ONE architecture.
    """
    
    def __init__(self, enhanced_model: EnhancedSIMONEModel, mvlm_config: Optional[Dict] = None):
        super().__init__()
        self.enhanced_model = enhanced_model
        self.mvlm_config = mvlm_config or {}
        
        # Extract model dimensions
        self.hidden_dim = enhanced_model.hidden_dim
        self.vocab_size = enhanced_model.vocab_size
        
        # Adapter layers for MVLM compatibility (if needed)
        if mvlm_config and 'n_embd' in mvlm_config:
            mvlm_dim = mvlm_config['n_embd']
            if mvlm_dim != self.hidden_dim:
                self.input_adapter = nn.Linear(mvlm_dim, self.hidden_dim)
                self.output_adapter = nn.Linear(self.hidden_dim, mvlm_dim)
                logger.info(f"Created adapters: MVLM dim {mvlm_dim} <-> Enhanced SIM-ONE dim {self.hidden_dim}")
            else:
                self.input_adapter = None
                self.output_adapter = None
        else:
            self.input_adapter = None
            self.output_adapter = None
        
        logger.info(f"MVLMAdapter initialized with vocab_size={self.vocab_size}, hidden_dim={self.hidden_dim}")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_governance: bool = False,
        prophetic_state: Optional[PropheticSingularityState] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MVLM compatibility.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for loss computation [batch, seq_len]
            output_governance: Whether to output governance information
            prophetic_state: Prophetic singularity state for governance
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with logits, loss (if labels provided), and governance outputs
        """
        try:
            # Forward pass through Enhanced SIM-ONE
            logits, governance_outputs = self.enhanced_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_governance=output_governance or (labels is not None),
                prophetic_state=prophetic_state,
                **kwargs
            )
            
            # Prepare output dictionary
            outputs = {
                'logits': logits,
                'last_hidden_state': logits,  # MVLM compatibility
            }
            
            # Add governance outputs if requested
            if output_governance and governance_outputs:
                outputs['governance_outputs'] = governance_outputs
            
            # Compute loss if labels provided
            if labels is not None:
                # Shift labels for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                outputs['loss'] = loss
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in MVLMAdapter forward pass: {e}")
            raise
    
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        prophetic_state: Optional[PropheticSingularityState] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using Enhanced SIM-ONE generation capabilities.
        
        Args:
            input_ids: Initial token IDs [batch, initial_seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            prophetic_state: Prophetic singularity state for governance
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs [batch, total_seq_len]
        """
        try:
            return self.enhanced_model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                prophetic_state=prophetic_state,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in MVLMAdapter generation: {e}")
            raise
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return self.enhanced_model.get_num_params()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return self.enhanced_model.get_memory_usage()
    
    def enable_cache(self):
        """Enable KV cache for efficient generation."""
        self.enhanced_model.enable_cache()
    
    def disable_cache(self):
        """Disable KV cache."""
        self.enhanced_model.disable_cache()
    
    def clear_cache(self):
        """Clear KV cache."""
        self.enhanced_model.clear_cache()


class EnhancedSIMONEWrapper(nn.Module):
    """
    Wrapper for Enhanced SIM-ONE model with additional utilities.
    Provides a clean interface for model instantiation and configuration.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ff_dim: int = 3072,
        num_layers: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        tie_embeddings: bool = True,
        config: Optional[PrioritaryConfig] = None
    ):
        super().__init__()
        
        # Use config if provided, otherwise use individual parameters
        if config:
            model_config = {
                'vocab_size': config.vocab_size,
                'hidden_dim': config.hidden_dim,
                'num_heads': config.num_heads,
                'ff_dim': config.ff_dim,
                'num_layers': config.num_layers,
                'max_seq_len': config.max_length,
                'dropout': dropout,
                'use_moe': use_moe,
                'num_experts': num_experts,
                'tie_embeddings': tie_embeddings
            }
        else:
            model_config = {
                'vocab_size': vocab_size,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim,
                'num_layers': num_layers,
                'max_seq_len': max_seq_len,
                'dropout': dropout,
                'use_moe': use_moe,
                'num_experts': num_experts,
                'tie_embeddings': tie_embeddings
            }
        
        # Create Enhanced SIM-ONE model
        self.model = EnhancedSIMONEModel(**model_config)
        
        # Store configuration
        self.config = model_config
        
        logger.info(f"EnhancedSIMONEWrapper initialized with config: {model_config}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through Enhanced SIM-ONE model."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using Enhanced SIM-ONE model."""
        return self.model.generate(*args, **kwargs)
    
    def get_mvlm_adapter(self, mvlm_config: Optional[Dict] = None) -> MVLMAdapter:
        """
        Get MVLM adapter for this model.
        
        Args:
            mvlm_config: Optional MVLM configuration for compatibility
            
        Returns:
            MVLMAdapter instance wrapping this model
        """
        return MVLMAdapter(self.model, mvlm_config)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return self.model.get_num_params()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return self.model.get_memory_usage()
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'EnhancedSIMONEWrapper':
        """
        Load model from pretrained checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            **kwargs: Additional arguments for model initialization
            
        Returns:
            EnhancedSIMONEWrapper instance with loaded weights
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model config from checkpoint
            model_config = checkpoint.get('model_config', {})
            model_config.update(kwargs)  # Override with provided kwargs
            
            # Create wrapper
            wrapper = cls(**model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                wrapper.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                wrapper.model.load_state_dict(checkpoint)
            
            logger.info(f"Model loaded from {model_path}")
            return wrapper
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def save_pretrained(self, save_path: str):
        """
        Save model to checkpoint.
        
        Args:
            save_path: Path to save model checkpoint
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': self.config
            }
            
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}")
            raise


def create_mvlm_adapter(
    vocab_size: int = 32000,
    hidden_dim: int = 768,
    num_heads: int = 12,
    ff_dim: int = 3072,
    num_layers: int = 12,
    max_seq_len: int = 2048,
    mvlm_config: Optional[Dict] = None,
    **kwargs
) -> MVLMAdapter:
    """
    Convenience function to create MVLM adapter with Enhanced SIM-ONE model.
    
    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        ff_dim: Feedforward dimension
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        mvlm_config: Optional MVLM configuration for compatibility
        **kwargs: Additional model arguments
        
    Returns:
        MVLMAdapter instance
    """
    # Create Enhanced SIM-ONE model
    model_config = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len,
        **kwargs
    }
    
    enhanced_model = EnhancedSIMONEModel(**model_config)
    
    # Create and return adapter
    return MVLMAdapter(enhanced_model, mvlm_config)


# Test function to verify adapter functionality
def test_mvlm_adapter():
    """Test function to verify MVLM adapter works correctly."""
    try:
        # Create test adapter
        adapter = create_mvlm_adapter(
            vocab_size=1000,  # Small vocab for testing
            hidden_dim=256,
            num_heads=4,
            ff_dim=1024,
            num_layers=2,
            max_seq_len=128
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        outputs = adapter(input_ids=input_ids, labels=labels)
        
        # Check outputs
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, 1000)
        
        # Test generation
        generated = adapter.generate(input_ids[:, :10], max_length=20)
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 20
        
        logger.info("✓ MVLM adapter test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MVLM adapter test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    test_mvlm_adapter()