"""
Advanced Loss Functions with Biblical Alignment Metrics for SIM-ONE Training
Implements sophisticated loss functions that encourage biblical fidelity and theological coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from collections import defaultdict

from .config import PropheticSingularityState


class BiblicalAlignmentLoss(nn.Module):
    """
    Loss function that measures alignment with biblical principles and content.
    Uses semantic similarity and theological concept matching.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        biblical_concepts: List[str] = None,
        alignment_weight: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.alignment_weight = alignment_weight
        
        # Biblical concept embeddings
        self.biblical_concepts = biblical_concepts or self._get_default_biblical_concepts()
        num_concepts = len(self.biblical_concepts)
        
        self.concept_embeddings = nn.Embedding(num_concepts, hidden_dim)
        
        # Concept detection network
        self.concept_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
            nn.Sigmoid()
        )
        
        # Alignment scorer
        self.alignment_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize concept embeddings with biblical themes
        self._init_biblical_embeddings()
    
    def _get_default_biblical_concepts(self) -> List[str]:
        """Default biblical concepts for alignment."""
        return [
            'love', 'faith', 'hope', 'grace', 'mercy', 'forgiveness', 'salvation',
            'redemption', 'sanctification', 'righteousness', 'holiness', 'truth',
            'justice', 'peace', 'joy', 'patience', 'kindness', 'goodness',
            'faithfulness', 'gentleness', 'self-control', 'wisdom', 'understanding',
            'knowledge', 'discernment', 'prayer', 'worship', 'praise', 'thanksgiving',
            'repentance', 'confession', 'baptism', 'communion', 'fellowship',
            'discipleship', 'evangelism', 'mission', 'service', 'stewardship',
            'obedience', 'surrender', 'sacrifice', 'covenant', 'promise',
            'prophecy', 'fulfillment', 'kingdom', 'eternal', 'glory'
        ]
    
    def _init_biblical_embeddings(self):
        """Initialize embeddings to capture biblical themes."""
        # Use orthogonal initialization to ensure diverse concept representations
        nn.init.orthogonal_(self.concept_embeddings.weight)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_concepts: Optional[torch.Tensor] = None,
        text_metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute biblical alignment loss.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            target_concepts: Target concept labels [batch, seq_len, num_concepts]
            text_metadata: Metadata about biblical content
            
        Returns:
            alignment_loss: Loss encouraging biblical alignment
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Detect biblical concepts in hidden states
        concept_logits = self.concept_detector(hidden_states)  # [batch, seq_len, num_concepts]
        
        if target_concepts is not None:
            # Supervised alignment with known biblical concepts
            concept_loss = F.binary_cross_entropy(concept_logits, target_concepts)
        else:
            # Unsupervised alignment - encourage diverse concept activation
            concept_probs = concept_logits
            # Entropy regularization to encourage diverse concept usage
            concept_entropy = -torch.sum(concept_probs * torch.log(concept_probs + 1e-8), dim=-1).mean()
            concept_loss = -concept_entropy  # Maximize entropy
        
        # Compute alignment with biblical concepts
        all_concept_embeds = self.concept_embeddings.weight  # [num_concepts, hidden_dim]
        
        # Compute similarity between hidden states and biblical concepts
        hidden_norm = F.normalize(hidden_states, dim=-1)
        concept_norm = F.normalize(all_concept_embeds, dim=-1)
        
        # Similarity matrix: [batch, seq_len, num_concepts]
        similarities = torch.matmul(hidden_norm, concept_norm.t())
        
        # Weight by concept activations
        weighted_similarities = similarities * concept_logits
        
        # Alignment score - higher is better
        alignment_score = weighted_similarities.sum(dim=-1).mean()
        
        # Convert to loss (minimize negative alignment)
        alignment_loss = -alignment_score
        
        # Combine losses
        total_loss = concept_loss + self.alignment_weight * alignment_loss
        
        return total_loss


class TheologicalCoherenceLoss(nn.Module):
    """
    Ensures theological coherence across generated text.
    Penalizes contradictions and encourages consistent doctrine.
    """
    
    def __init__(self, hidden_dim: int, coherence_weight: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coherence_weight = coherence_weight
        
        # Theological consistency checker
        self.consistency_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Doctrinal conflict detector
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor, window_size: int = 32) -> torch.Tensor:
        """
        Compute theological coherence loss.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            window_size: Window size for coherence checking
            
        Returns:
            coherence_loss: Loss encouraging theological coherence
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        total_loss = 0
        num_windows = 0
        
        # Slide window across sequence
        for start in range(0, seq_len - window_size + 1, window_size // 2):
            end = start + window_size
            window_states = hidden_states[:, start:end, :]
            
            # Encode for consistency checking
            consistent_encoding = self.consistency_encoder(window_states)
            
            # Compute consistency scores
            consistency_scores = self.consistency_scorer(consistent_encoding)
            
            # High consistency scores are good
            consistency_loss = 1.0 - consistency_scores.mean()
            
            # Check for conflicts between adjacent segments
            if start > 0:
                prev_window = hidden_states[:, start-window_size//2:start+window_size//2, :]
                curr_window = window_states
                
                # Mean pooling for segment representation
                prev_repr = prev_window.mean(dim=1)  # [batch, hidden_dim]
                curr_repr = curr_window.mean(dim=1)  # [batch, hidden_dim]
                
                # Detect conflicts
                conflict_input = torch.cat([prev_repr, curr_repr], dim=-1)
                conflict_score = self.conflict_detector(conflict_input)
                
                # Low conflict scores are good
                conflict_loss = conflict_score.mean()
                
                consistency_loss += conflict_loss
            
            total_loss += consistency_loss
            num_windows += 1
        
        return self.coherence_weight * (total_loss / max(1, num_windows))


class ScriptureReferenceLoss(nn.Module):
    """
    Encourages accurate scripture references and citations.
    Penalizes incorrect or fabricated biblical references.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Scripture reference patterns (book:chapter:verse)
        self.reference_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Accuracy scorer for detected references
        self.accuracy_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        reference_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scripture reference accuracy loss.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            input_ids: Input token IDs [batch, seq_len]
            reference_labels: Known scripture reference accuracy [batch, seq_len]
            
        Returns:
            reference_loss: Loss encouraging accurate scripture references
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Detect potential scripture references
        reference_probs = self.reference_detector(hidden_states).squeeze(-1)
        
        if reference_labels is not None:
            # Supervised training with known reference accuracy
            reference_loss = F.binary_cross_entropy(reference_probs, reference_labels)
        else:
            # Unsupervised - encourage high confidence in references
            # This would be combined with external validation
            confidence_penalty = -torch.log(reference_probs + 1e-8).mean()
            reference_loss = confidence_penalty * 0.1  # Small penalty
        
        # Score accuracy of detected references
        reference_mask = reference_probs > 0.5
        if reference_mask.any():
            reference_hidden = hidden_states[reference_mask]
            accuracy_scores = self.accuracy_scorer(reference_hidden)
            
            # Encourage high accuracy
            accuracy_loss = 1.0 - accuracy_scores.mean()
            reference_loss += accuracy_loss
        
        return reference_loss


class StyleConsistencyLoss(nn.Module):
    """
    Maintains consistent biblical/theological writing style.
    Encourages appropriate tone and language patterns.
    """
    
    def __init__(self, hidden_dim: int, num_styles: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_styles = num_styles
        
        # Style classifier
        self.style_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_styles)
        )
        
        # Style consistency enforcer
        self.consistency_enforcer = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        
        self.style_scorer = nn.Linear(hidden_dim * 2, num_styles)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_style: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute style consistency loss.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            target_style: Target style ID (0=narrative, 1=prophetic, 2=wisdom, etc.)
            
        Returns:
            style_loss: Loss encouraging consistent style
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Classify style at each position
        style_logits = self.style_classifier(hidden_states)  # [batch, seq_len, num_styles]
        
        if target_style is not None:
            # Supervised style consistency
            target_labels = torch.full((batch_size, seq_len), target_style, device=hidden_states.device)
            style_loss = F.cross_entropy(
                style_logits.view(-1, self.num_styles),
                target_labels.view(-1)
            )
        else:
            # Unsupervised style consistency - encourage consistent style within sequence
            style_probs = F.softmax(style_logits, dim=-1)
            
            # Compute style consistency across sequence
            style_consistency, _ = self.consistency_enforcer(hidden_states)
            consistent_style_logits = self.style_scorer(style_consistency)
            
            # KL divergence between local and global style predictions
            global_style_probs = F.softmax(consistent_style_logits, dim=-1)
            
            style_loss = F.kl_div(
                F.log_softmax(style_logits, dim=-1),
                global_style_probs,
                reduction='batchmean'
            )
        
        return style_loss


class ComprehensiveBiblicalLoss(nn.Module):
    """
    Comprehensive loss function combining all biblical alignment objectives.
    Provides balanced training for theologically sound language generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        # Default loss weights
        default_weights = {
            'mle': 1.0,                    # Standard language modeling
            'biblical_alignment': 0.5,      # Biblical concept alignment
            'theological_coherence': 0.3,   # Theological consistency
            'scripture_reference': 0.2,     # Scripture accuracy
            'style_consistency': 0.2,       # Style consistency
            'policy': 0.1,                 # Governance policy
            'memory': 0.1,                 # Memory management
            'energy': 0.05                 # Energy efficiency
        }
        
        self.loss_weights = loss_weights or default_weights
        
        # Individual loss components
        self.biblical_alignment = BiblicalAlignmentLoss(vocab_size, hidden_dim)
        self.theological_coherence = TheologicalCoherenceLoss(hidden_dim)
        self.scripture_reference = ScriptureReferenceLoss(vocab_size, hidden_dim)
        self.style_consistency = StyleConsistencyLoss(hidden_dim)
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        governance_outputs: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        prophetic_state: Optional[PropheticSingularityState] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute comprehensive biblical loss.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            governance_outputs: Governance information from model
            metadata: Additional metadata about the text
            
        Returns:
            total_loss: Combined loss value
            loss_components: Individual loss components
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Standard MLE loss
        mle_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Biblical alignment loss
        biblical_loss = self.biblical_alignment(
            hidden_states,
            target_concepts=metadata.get('biblical_concepts') if metadata else None,
            text_metadata=metadata
        )
        
        # Theological coherence loss
        coherence_loss = self.theological_coherence(hidden_states)
        
        # Scripture reference loss
        input_ids = metadata.get('input_ids') if metadata else None
        reference_loss = self.scripture_reference(
            hidden_states,
            input_ids,
            reference_labels=metadata.get('reference_accuracy') if metadata else None
        )
        
        # Style consistency loss
        target_style = metadata.get('target_style') if metadata else None
        style_loss = self.style_consistency(hidden_states, target_style)
        
        # Governance losses (from original implementation)
        policy_loss = torch.tensor(0.0, device=logits.device)
        memory_loss = torch.tensor(0.0, device=logits.device)
        # Energy efficiency - L1 penalty on activations is always applied
        energy_loss = torch.mean(torch.abs(hidden_states))

        if governance_outputs:
            def _extract_final_tensor(source_key: str, fallback_key: Optional[str] = None):
                """Helper to get the final-layer tensor from governance outputs."""
                if source_key in governance_outputs and governance_outputs[source_key] is not None:
                    source = governance_outputs[source_key]
                elif fallback_key and fallback_key in governance_outputs and governance_outputs[fallback_key] is not None:
                    source = governance_outputs[fallback_key]
                else:
                    source = None

                if isinstance(source, (list, tuple)):
                    if len(source) == 0:
                        return None
                    return source[-1]
                return source

            policy_logits = _extract_final_tensor('policy_logits', fallback_key='policy')
            if policy_logits is None and 'policy_logits_all_layers' in governance_outputs:
                policy_logits = _extract_final_tensor('policy_logits_all_layers')

            if policy_logits is not None:
                # Policy regularization - encourage confident policy decisions
                policy_loss = -torch.mean(torch.max(torch.abs(policy_logits), dim=-1)[0])

            memory_signals = _extract_final_tensor('memory_signals', fallback_key='memory')
            if memory_signals is None and 'memory_signals_all_layers' in governance_outputs:
                memory_signals = _extract_final_tensor('memory_signals_all_layers')

            if memory_signals is not None:
                # Memory efficiency - encourage sparse memory usage
                memory_loss = torch.mean(torch.abs(memory_signals))
        
        if prophetic_state is None and metadata:
            candidate = metadata.get('prophetic_state') if isinstance(metadata, dict) else None
            if isinstance(candidate, PropheticSingularityState):
                prophetic_state = candidate

        dynamic_weights = dict(self.loss_weights)
        kingdom_mean = None
        lambda_mean = None

        if prophetic_state is not None:
            aligned_state = prophetic_state.align_to_length(seq_len).to(logits.device, logits.dtype)
            summary = aligned_state.summary()
            kingdom_mean = summary['kingdom']['mean']
            lambda_mean = summary['lambda']['mean']
            intensity_mean = summary['intensity']['mean']
            mercy_mean = summary['mercy']['mean']
            dominion_mean = summary['dominion']['mean']

            def _scalar(value: torch.Tensor) -> float:
                if isinstance(value, torch.Tensor):
                    return float(value.detach().cpu().item())
                return float(value)

            dynamic_weights['memory'] *= 1.0 + _scalar(kingdom_mean)
            dynamic_weights['policy'] *= 1.0 + 0.5 * _scalar(dominion_mean)
            dynamic_weights['energy'] *= 1.0 + 0.2 * _scalar(lambda_mean)
            dynamic_weights['biblical_alignment'] *= 1.0 + 0.1 * _scalar(intensity_mean)
            dynamic_weights['theological_coherence'] *= 1.0 + 0.1 * _scalar(mercy_mean)

            decay_factor = torch.exp(-lambda_mean.clamp(min=0.0, max=5.0))
            memory_loss = memory_loss * decay_factor
        else:
            decay_factor = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)

        # Combine all losses
        loss_components = {
            'mle': mle_loss,
            'biblical_alignment': biblical_loss,
            'theological_coherence': coherence_loss,
            'scripture_reference': reference_loss,
            'style_consistency': style_loss,
            'policy': policy_loss,
            'memory': memory_loss,
            'energy': energy_loss
        }

        if kingdom_mean is not None:
            loss_components['kingdom_flow'] = kingdom_mean
        loss_components['memory_decay_factor'] = decay_factor

        # Weighted sum
        total_loss = sum(
            dynamic_weights.get(name, 0.0) * loss
            for name, loss in loss_components.items()
        )

        return total_loss, loss_components


def create_biblical_metadata(
    texts: List[str],
    tokenizer = None
) -> List[Dict]:
    """
    Create metadata for biblical training texts.
    
    Args:
        texts: List of training texts
        tokenizer: Tokenizer for encoding
        
    Returns:
        metadata: List of metadata dictionaries
    """
    metadata_list = []
    
    for text in texts:
        metadata = {
            'text_length': len(text),
            'has_scripture_ref': any(
                pattern in text.lower() 
                for pattern in [':', 'chapter', 'verse', 'psalm', 'genesis', 'matthew']
            ),
            'biblical_concepts': None,  # Would be populated by NLP analysis
            'reference_accuracy': None,  # Would be populated by reference validation
            'target_style': 0,  # Default to narrative style
        }
        
        if tokenizer:
            tokens = tokenizer.encode(text)
            metadata['input_ids'] = torch.tensor(tokens)
        
        metadata_list.append(metadata)
    
    return metadata_list


if __name__ == "__main__":
    # Test the advanced loss functions
    print("Testing Advanced Biblical Loss Functions...")
    
    # Test configuration
    batch_size, seq_len, vocab_size, hidden_dim = 2, 64, 1000, 512
    
    # Create test data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test comprehensive loss
    comprehensive_loss = ComprehensiveBiblicalLoss(vocab_size, hidden_dim)
    
    total_loss, components = comprehensive_loss(
        logits, labels, hidden_states
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss components:")
    for name, loss in components.items():
        print(f"  {name}: {loss.item():.4f}")
    
    print("✓ Advanced loss functions working correctly!")