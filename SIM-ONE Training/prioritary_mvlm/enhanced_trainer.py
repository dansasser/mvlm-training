"""
Enhanced trainer for SIM-ONE transformer with all modern improvements.
Integrates BPE tokenizer, advanced losses, and enhanced model architecture.
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from .config import PrioritaryConfig, PropheticSingularityState
from .dataset import WeightedTextDataset
from .advanced_tokenizer import BiblicalBPETokenizer, train_biblical_tokenizer
from .advanced_losses import ComprehensiveBiblicalLoss, create_biblical_metadata
from simone_transformer import EnhancedSIMONEModel


class EnhancedPrioritaryTrainer:
    """
    Enhanced trainer for SIM-ONE models with comprehensive biblical training.
    
    Features:
    - Advanced BPE tokenization with biblical vocabulary
    - Modern transformer architecture with RoPE and SwiGLU
    - Comprehensive biblical loss functions
    - Advanced optimization with cosine annealing
    - Mixed precision training
    - Gradient clipping and accumulation
    - Detailed logging and visualization
    - Model checkpointing and resuming
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: Optional[PrioritaryConfig] = None,
        tokenizer_path: Optional[str] = None,
        log_file: str = None,
        use_mixed_precision: bool = True,
        compile_model: bool = True
    ):
        self.config = config or PrioritaryConfig()
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mixed_precision = use_mixed_precision
        
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        # Initialize tokenizer
        self.logger.info("Initializing enhanced biblical tokenizer...")
        self.tokenizer = self._setup_tokenizer(tokenizer_path)
        
        # Initialize model
        self.logger.info("Initializing enhanced SIM-ONE model...")
        self.model = self._setup_model()
        
        # Compile model for efficiency (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Setup device and mixed precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if self.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using mixed precision training")
        else:
            self.scaler = None
        
        # Initialize dataset and dataloader
        self.logger.info("Loading enhanced training dataset...")
        self.dataset, self.dataloader = self._setup_dataset()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimization()
        
        # Initialize loss function
        self.loss_function = ComprehensiveBiblicalLoss(
            vocab_size=len(self.tokenizer),
            hidden_dim=self.config.hidden_dim,
            loss_weights={
                'mle': 1.0,
                'biblical_alignment': 0.5,
                'theological_coherence': 0.3,
                'scripture_reference': 0.2,
                'style_consistency': 0.2,
                'policy': 0.1,
                'memory': 0.1,
                'energy': 0.05
            }
        ).to(self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Log model info
        num_params = self.model.get_num_params()
        memory_usage = self.model.get_memory_usage()
        self.logger.info(f"Model parameters: {num_params:,}")
        self.logger.info(f"Model memory: {memory_usage['total'] / 1e6:.2f} MB")

    def _setup_logging(self, log_file: str = None) -> logging.Logger:
        """Setup enhanced logging."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger

    def _setup_tokenizer(self, tokenizer_path: Optional[str] = None) -> BiblicalBPETokenizer:
        """Setup the biblical BPE tokenizer."""
        if tokenizer_path and Path(tokenizer_path).exists():
            # Load existing tokenizer
            tokenizer = BiblicalBPETokenizer()
            tokenizer.load(tokenizer_path)
            self.logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            # Train new tokenizer on dataset
            self.logger.info("Training new biblical BPE tokenizer...")
            
            # Collect training texts
            texts = []
            data_path = Path(self.data_dir)
            
            for txt_file in data_path.rglob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            texts.append(text)
                except Exception as e:
                    self.logger.warning(f"Error reading {txt_file}: {e}")
            
            if not texts:
                self.logger.warning("No training texts found, using default tokenizer")
                tokenizer = BiblicalBPETokenizer(vocab_size=self.config.vocab_size)
                # Initialize with basic vocabulary
                tokenizer.train(["Hello world", "Biblical text sample"], 
                              save_path=str(self.output_dir / "tokenizer.pkl"))
            else:
                self.logger.info(f"Training tokenizer on {len(texts)} texts")
                tokenizer = train_biblical_tokenizer(
                    texts,
                    vocab_size=self.config.vocab_size,
                    save_path=str(self.output_dir / "tokenizer.pkl")
                )
        
        self.logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
        return tokenizer

    def _setup_model(self) -> EnhancedSIMONEModel:
        """Setup the enhanced SIM-ONE model."""
        model_config = {
            'vocab_size': len(self.tokenizer),
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads,
            'ff_dim': self.config.ff_dim,
            'num_layers': self.config.num_layers,
            'max_seq_len': self.config.max_length,
            'dropout': 0.1,
            'use_moe': False,  # Can be enabled for larger models
            'tie_embeddings': True
        }
        
        model = EnhancedSIMONEModel(**model_config)
        
        self.logger.info(f"Model configuration: {model_config}")
        return model

    def _setup_dataset(self) -> Tuple[WeightedTextDataset, DataLoader]:
        """Setup the enhanced dataset and dataloader."""
        # Use enhanced config values
        dataset = WeightedTextDataset(
            self.data_dir,
            self.tokenizer,
            self.config
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset.collate_fn
        )
        
        self.logger.info(f"Dataset size: {len(dataset)} examples")
        self.logger.info(f"Batches per epoch: {len(dataloader)}")
        
        return dataset, dataloader

    def _setup_optimization(self) -> Tuple[AdamW, torch.optim.lr_scheduler.LRScheduler]:
        """Setup optimizer and learning rate scheduler."""
        # AdamW optimizer with weight decay
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),  # Modern LLM settings
            eps=1e-8
        )
        
        # Combined scheduler: Linear warmup + Cosine annealing
        total_steps = len(self.dataloader) * self.config.num_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )
        
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Warmup steps: {self.config.warmup_steps}")
        
        return optimizer, scheduler

    def compute_enhanced_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss using the comprehensive biblical loss function."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        prophetic_state = batch.get('prophetic_state')
        if isinstance(prophetic_state, PropheticSingularityState):
            prophetic_state = prophetic_state.to(self.device, dtype=torch.float32)

        # Forward pass through model
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits, governance_outputs = self.model(
                    input_ids,
                    output_governance=True,
                    prophetic_state=prophetic_state
                )

                # Create metadata for loss computation
                metadata = {
                    'input_ids': input_ids,
                    'batch_metadata': batch.get('metadata', {}),
                    'prophetic_state': prophetic_state
                }
                if isinstance(governance_outputs, dict) and 'trace_metadata' in governance_outputs:
                    metadata['trace_metadata'] = governance_outputs['trace_metadata']

                trace_tensor = None
                if isinstance(governance_outputs, dict):
                    trace_tensor = governance_outputs.get('trace')
                if not isinstance(trace_tensor, torch.Tensor):
                    raise ValueError(
                        "Governance outputs must include a trace tensor for enhanced loss computation."
                    )

                # Compute comprehensive loss
                total_loss, loss_components = self.loss_function(
                    logits=logits,
                    labels=labels,
                    hidden_states=trace_tensor,
                    governance_outputs=governance_outputs,
                    metadata=metadata,
                    prophetic_state=prophetic_state
                )
        else:
            logits, governance_outputs = self.model(
                input_ids,
                output_governance=True,
                prophetic_state=prophetic_state
            )

            metadata = {
                'input_ids': input_ids,
                'batch_metadata': batch.get('metadata', {}),
                'prophetic_state': prophetic_state
            }
            if isinstance(governance_outputs, dict) and 'trace_metadata' in governance_outputs:
                metadata['trace_metadata'] = governance_outputs['trace_metadata']

            trace_tensor = None
            if isinstance(governance_outputs, dict):
                trace_tensor = governance_outputs.get('trace')
            if not isinstance(trace_tensor, torch.Tensor):
                raise ValueError(
                    "Governance outputs must include a trace tensor for enhanced loss computation."
                )

            total_loss, loss_components = self.loss_function(
                logits=logits,
                labels=labels,
                hidden_states=trace_tensor,
                governance_outputs=governance_outputs,
                metadata=metadata,
                prophetic_state=prophetic_state
            )
        
        return total_loss, loss_components

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with enhanced features."""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'mle_loss': 0.0,
            'biblical_loss': 0.0,
            'coherence_loss': 0.0,
            'governance_loss': 0.0
        }
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Compute loss
            loss, loss_components = self.compute_enhanced_loss(batch)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Update metrics
            epoch_metrics['total_loss'] += loss.item()
            for name, component_loss in loss_components.items():
                if name in ['mle', 'biblical_alignment', 'theological_coherence']:
                    key = name.replace('mle', 'mle_loss').replace('biblical_alignment', 'biblical_loss').replace('theological_coherence', 'coherence_loss')
                    if key in epoch_metrics:
                        epoch_metrics[key] += component_loss.item()
                elif name in ['policy', 'memory', 'energy']:
                    epoch_metrics['governance_loss'] += component_loss.item()
            
            # Optimizer step
            if ((batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or 
                (batch_idx + 1) == len(self.dataloader)):
                
                # Gradient clipping
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f"Epoch {self.epoch}, Step {self.global_step}, "
                        f"Loss: {loss.item():.4f}, LR: {lr:.2e}"
                    )
                    
                    # Log detailed loss components
                    if self.global_step % (self.config.log_interval * 5) == 0:
                        self.logger.info("Loss components:")
                        for name, component_loss in loss_components.items():
                            self.logger.info(f"  {name}: {component_loss.item():.4f}")
                
                # Save checkpoint
                if self.global_step % self.config.eval_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")
                    
                    # Evaluation
                    eval_metrics = self.evaluate()
                    self.logger.info(f"Eval loss: {eval_metrics['loss']:.4f}, PPL: {eval_metrics['perplexity']:.2f}")
                    
                    # Save best model
                    if eval_metrics['loss'] < self.best_loss:
                        self.best_loss = eval_metrics['loss']
                        self.save_checkpoint("best_model")
                        self.logger.info(f"New best model saved (loss: {self.best_loss:.4f})")
            
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_batches)
        
        return epoch_metrics

    def evaluate(self, max_batches: int = 100) -> Dict[str, float]:
        """Evaluate model with enhanced metrics."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= max_batches:
                    break
                
                loss, _ = self.compute_enhanced_loss(batch)
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 10))  # Cap for numerical stability
        
        self.model.train()
        return {'loss': avg_loss, 'perplexity': perplexity}

    def generate_sample(self, prompt: str = "In the beginning", max_length: int = 100) -> str:
        """Generate sample text using the enhanced model."""
        self.model.eval()
        
        with torch.no_grad():
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            
            # Generate
            generated_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        self.model.train()
        return generated_text

    def save_checkpoint(self, filename: str):
        """Save enhanced checkpoint."""
        checkpoint_path = self.output_dir / f"{filename}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'tokenizer_config': {
                'vocab_size': len(self.tokenizer),
                'vocab': self.tokenizer.get_vocab(),
            },
            'training_state': {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
            },
            'training_history': self.training_history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load enhanced checkpoint."""
        checkpoint_path = self.output_dir / f"{filename}.pt"
        
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        training_state = checkpoint.get('training_state', {})
        self.epoch = training_state.get('epoch', 0)
        self.global_step = training_state.get('global_step', 0)
        self.best_loss = training_state.get('best_loss', float('inf'))
        
        self.training_history = checkpoint.get('training_history', [])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")

    def create_training_plots(self):
        """Create comprehensive training visualization."""
        if not self.training_history:
            return
        
        # Extract metrics
        steps = [h['global_step'] for h in self.training_history]
        losses = [h.get('total_loss', h.get('loss', 0)) for h in self.training_history]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(steps, losses, label='Total Loss', color='blue', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Enhanced SIM-ONE Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rate' in self.training_history[0]:
            lrs = [h['learning_rate'] for h in self.training_history]
            ax2.plot(steps, lrs, color='green', linewidth=2)
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
        
        # Loss components (if available)
        if 'biblical_loss' in self.training_history[0]:
            biblical_losses = [h['biblical_loss'] for h in self.training_history]
            coherence_losses = [h.get('coherence_loss', 0) for h in self.training_history]
            
            ax3.plot(steps, biblical_losses, label='Biblical Alignment', color='purple')
            ax3.plot(steps, coherence_losses, label='Theological Coherence', color='orange')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Loss')
            ax3.set_title('Biblical Training Components')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Smoothed loss
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            
            ax4.plot(smoothed_steps, smoothed_losses, color='red', linewidth=2)
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Smoothed Loss')
            ax4.set_title(f'Smoothed Training Loss (window={window})')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "enhanced_training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved: {plot_path}")

    def train(self, num_epochs: Optional[int] = None) -> Path:
        """Run the complete enhanced training pipeline."""
        epochs = num_epochs or self.config.num_epochs
        
        self.logger.info("="*60)
        self.logger.info("STARTING ENHANCED SIM-ONE TRAINING")
        self.logger.info("="*60)
        self.logger.info(f"Dataset: {len(self.dataset)} examples")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
        start_time = time.time()
        
        for epoch_idx in range(epochs):
            self.epoch = epoch_idx
            self.logger.info(f"\n--- Epoch {epoch_idx + 1}/{epochs} ---")
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch_idx + 1} completed:")
            for metric, value in epoch_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
            
            # Record history
            epoch_record = {
                'epoch': epoch_idx + 1,
                'global_step': self.global_step,
                **epoch_metrics
            }
            self.training_history.append(epoch_record)
            
            # Generate sample
            sample = self.generate_sample()
            self.logger.info(f"Sample generation: {sample[:200]}...")
        
        # Training completed
        training_time = time.time() - start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info("ENHANCED SIM-ONE TRAINING COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Training time: {training_time:.2f}s ({training_time/60:.2f}m)")
        self.logger.info(f"Total steps: {self.global_step}")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        
        # Save final model
        final_model_path = self.save_final_model()
        
        # Create plots
        self.create_training_plots()
        
        # Final samples
        self.logger.info("\n--- Final Sample Generations ---")
        biblical_prompts = [
            "In the beginning God",
            "For God so loved",
            "The LORD is my shepherd",
            "Faith is the substance",
            "Trust in the LORD"
        ]
        
        for prompt in biblical_prompts:
            sample = self.generate_sample(prompt, max_length=150)
            self.logger.info(f"Prompt: '{prompt}' -> {sample}")
        
        return final_model_path

    def save_final_model(self) -> Path:
        """Save the final trained model."""
        model_path = self.output_dir / "enhanced_simone_final"
        
        # Save model components
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': len(self.tokenizer),
                'hidden_dim': self.config.hidden_dim,
                'num_heads': self.config.num_heads,
                'ff_dim': self.config.ff_dim,
                'num_layers': self.config.num_layers,
                'max_seq_len': self.config.max_length,
            },
            'tokenizer_path': str(self.output_dir / "tokenizer.pkl"),
            'training_config': self.config.__dict__,
            'training_stats': {
                'final_loss': self.best_loss,
                'total_steps': self.global_step,
                'total_epochs': self.epoch,
            }
        }
        
        torch.save(final_checkpoint, f"{model_path}.pt")
        
        # Also save training history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Final model saved: {model_path}.pt")
        return model_path