"""Enhanced SIM-ONE base wrappers and MVLM adapter implementations."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from prioritary_mvlm.config import PrioritaryConfig, PropheticSingularityState
from simone_transformer.shared_governance import SharedGovernanceBackbone

logger = logging.getLogger(__name__)


class _TransformerBlock(nn.Module):
    """Lightweight transformer block used by the enhanced wrapper."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = None
        if attention_mask is not None:
            # Expect attention mask as 1 for tokens to keep, 0 for masked
            key_padding_mask = attention_mask == 0

        attn_out, attn_weights = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights


class MVLMAdapter(nn.Module):
    """
    Working MVLM adapter wrapper for Enhanced SIM-ONE integration.
    Provides compatibility between MVLM interfaces and Enhanced SIM-ONE architecture.
    """
    
    def __init__(self, enhanced_model: 'SIMONEModel', mvlm_config: Optional[Dict] = None):
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
            logits, governance_outputs, _ = self.enhanced_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_governance=output_governance or (labels is not None),
                prophetic_state=prophetic_state,
                **kwargs
            )
            
            # Prepare output dictionary
            hidden_state = kwargs.get('hidden_state')
            if hidden_state is None and isinstance(governance_outputs, dict):
                hidden_state = governance_outputs.get('shared_governance_features')

            outputs = {
                'logits': logits,
                'last_hidden_state': hidden_state if hidden_state is not None else logits,
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
        
        self.vocab_size = model_config['vocab_size']
        self.hidden_dim = model_config['hidden_dim']
        self.num_heads = model_config['num_heads']
        self.ff_dim = model_config['ff_dim']
        self.num_layers = model_config['num_layers']
        self.max_seq_len = model_config['max_seq_len']
        self.dropout = model_config['dropout']
        self.use_moe = model_config['use_moe']
        self.num_experts = model_config['num_experts']
        self.tie_embeddings = model_config['tie_embeddings']

        # Core transformer components
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList(
            _TransformerBlock(self.hidden_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Governance backbone for optional trace generation
        self.governance = SharedGovernanceBackbone(self.hidden_dim, num_heads=self.num_heads)

        # Store configuration
        self.config = model_config

        logger.info(f"EnhancedSIMONEWrapper initialized with config: {model_config}")

    def _forward_internal(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.dropout_layer(hidden_states)

        last_attn_weights = None
        for block in self.blocks:
            hidden_states, attn_weights = block(hidden_states, attention_mask)
            last_attn_weights = attn_weights

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, hidden_states, last_attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_governance: bool = False,
        prophetic_state: Optional[PropheticSingularityState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Any]]:
        """Forward pass through Enhanced SIM-ONE model."""

        logits, hidden_states, attn_weights = self._forward_internal(input_ids, attention_mask)

        governance_outputs: Dict[str, torch.Tensor] = {}
        if output_governance:
            governance_outputs = self.governance(
                hidden_states,
                attention_weights=attn_weights,
                prophetic_state=prophetic_state,
            )
            governance_outputs['trace'] = governance_outputs.get('trace', {})
        return logits, governance_outputs, None

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
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using a simple autoregressive loop."""

        self.eval()
        generated = input_ids
        with torch.no_grad():
            for _ in range(max(0, max_length - input_ids.size(1))):
                attention_mask = torch.ones_like(generated, device=generated.device)
                logits, _, _ = self._forward_internal(generated, attention_mask)
                next_token_logits = logits[:, -1, :]

                if temperature != 1.0:
                    next_token_logits = next_token_logits / max(temperature, 1e-5)

                probs = torch.softmax(next_token_logits, dim=-1)

                if do_sample:
                    if top_k is not None and top_k > 0:
                        values, indices = torch.topk(probs, top_k, dim=-1)
                        probs = torch.zeros_like(probs).scatter(-1, indices, values)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = probs.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break

        if pad_token_id is not None and generated.size(1) < max_length:
            pad_size = max_length - generated.size(1)
            padding = torch.full((generated.size(0), pad_size), pad_token_id, device=generated.device)
            generated = torch.cat([generated, padding], dim=1)

        return generated
    
    def get_mvlm_adapter(self, mvlm_config: Optional[Dict] = None) -> MVLMAdapter:
        """
        Get MVLM adapter for this model.
        
        Args:
            mvlm_config: Optional MVLM configuration for compatibility
            
        Returns:
            MVLMAdapter instance wrapping this model
        """
        return MVLMAdapter(self, mvlm_config)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return {
            'parameters': param_bytes,
            'buffers': buffer_bytes,
            'total': param_bytes + buffer_bytes,
        }
    
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

            model_config = checkpoint.get('model_config', {})
            model_config.update(kwargs)

            wrapper = cls(**model_config)

            state_dict = checkpoint.get('model_state_dict', checkpoint)
            wrapper.load_state_dict(state_dict)
            
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
                'model_state_dict': self.state_dict(),
                'model_config': self.config,
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
    
    enhanced_model = EnhancedSIMONEWrapper(**model_config)

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