#!/usr/bin/env python3
"""
MVLM (Minimum Viable Language Model) Training Script
Complete training pipeline for biblically-grounded language model
Designed for Digital Ocean GPU instances
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mvlm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MVLMConfig:
    """Configuration for MVLM training"""
    
    def __init__(self):
        # Model architecture
        self.vocab_size = 50257  # GPT-2 tokenizer vocab size
        self.n_positions = 1024  # Context length
        self.n_embd = 768       # Embedding dimension
        self.n_layer = 12       # Number of transformer layers
        self.n_head = 12        # Number of attention heads
        self.n_inner = 3072     # Feed-forward dimension

        # Training parameters
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 500
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 1
        self.emb_dropout = 0.1
        self.resid_dropout = 0.1
        self.attn_dropout = 0.1
        self.label_smoothing = 0.0
        
        # Data parameters
        self.max_length = 512
        self.stride = 256
        
        # Logging and saving
        self.log_interval = 100
        self.save_interval = 1000
        self.eval_interval = 500
        
        # Biblical worldview optimization
        self.biblical_weight = 1.2  # Slight emphasis on biblical content
        self.quality_weight = 1.1   # Emphasis on high-quality content

class BiblicalTextDataset(Dataset):
    """Dataset class for biblical worldview training data"""
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        self.metadata = []
        
        logger.info(f"Loading dataset from {data_dir}")
        self.load_data(data_dir)
        logger.info(f"Loaded {len(self.examples)} training examples")
    
    def load_data(self, data_dir: str):
        """Load and tokenize all text files from the dataset"""
        data_path = Path(data_dir)
        
        for txt_file in data_path.rglob("*.txt"):
            # Load corresponding metadata
            json_file = txt_file.with_suffix('.json')
            if json_file.exists():
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'quality_score': 8.0, 'biblical_alignment': 7.0}
            
            # Read text content
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize and create examples
                self.process_text(text, metadata)
                
            except Exception as e:
                logger.warning(f"Error processing {txt_file}: {e}")
    
    def process_text(self, text: str, metadata: Dict):
        """Process text into training examples"""
        # Tokenize the full text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Create overlapping windows
        for i in range(0, len(tokens) - self.max_length + 1, self.stride):
            example_tokens = tokens[i:i + self.max_length]
            
            # Pad if necessary
            if len(example_tokens) < self.max_length:
                example_tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(example_tokens)))
            
            self.examples.append(torch.tensor(example_tokens, dtype=torch.long))
            self.metadata.append(metadata)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx],
            'labels': self.examples[idx].clone(),
            'metadata': self.metadata[idx]
        }

class MVLMTrainer:
    """Complete MVLM training pipeline"""
    
    def __init__(self, config: MVLMConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        logger.info("Initializing MVLM model...")
        self.model = self.create_model()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize dataset and dataloader
        logger.info("Loading training dataset...")
        self.dataset = BiblicalTextDataset(
            data_dir, 
            self.tokenizer, 
            self.config.max_length, 
            self.config.stride
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(self.dataloader) * self.config.num_epochs
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
    def create_model(self) -> GPT2LMHeadModel:
        """Create MVLM model with biblical worldview optimization"""
        config = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.n_positions,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=self.config.n_inner,
            activation_function="gelu_new",
            resid_pdrop=self.config.resid_dropout,
            embd_pdrop=self.config.emb_dropout,
            attn_pdrop=self.config.attn_dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=self.config.emb_dropout,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50256
        )
        
        model = GPT2LMHeadModel(config)
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def compute_loss(self, batch):
        """Compute loss with biblical worldview weighting"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=self.config.label_smoothing,
        )
        base_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Apply biblical worldview weighting
        biblical_weights = []
        quality_weights = []
        
        for metadata in batch['metadata']:
            biblical_score = metadata.get('biblical_alignment', 7.0)
            quality_score = metadata.get('quality_score', 8.0)
            
            # Higher scores get slightly higher weight
            biblical_weight = 1.0 + (biblical_score - 7.0) * 0.1
            quality_weight = 1.0 + (quality_score - 8.0) * 0.05
            
            biblical_weights.append(biblical_weight)
            quality_weights.append(quality_weight)
        
        # Apply weights (simplified for batch processing)
        avg_biblical_weight = sum(biblical_weights) / len(biblical_weights)
        avg_quality_weight = sum(quality_weights) / len(quality_weights)
        
        weighted_loss = base_loss * avg_biblical_weight * avg_quality_weight
        
        return weighted_loss, base_loss
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_base_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.dataloader):
            # Compute loss
            loss, base_loss = self.compute_loss(batch)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_base_loss += base_loss.item()

            if (
                (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.dataloader)
            ):
                # Gradient clipping
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.global_step < self.config.warmup_steps:
                    self.scheduler.step()

                num_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = total_loss / num_batches
                    avg_base_loss = total_base_loss / num_batches
                    lr = self.optimizer.param_groups[0]["lr"]

                    logger.info(
                        f"Epoch {self.epoch}, Step {self.global_step}, "
                        f"Loss: {avg_loss:.4f}, Base Loss: {avg_base_loss:.4f}, "
                        f"LR: {lr:.2e}"
                    )

                    # Record training history
                    self.training_history.append(
                        {
                            "epoch": self.epoch,
                            "batch": batch_idx,
                            "loss": avg_loss,
                            "base_loss": avg_base_loss,
                            "learning_rate": lr,
                            "global_step": self.global_step,
                        }
                    )

                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")

        return total_loss / num_batches
    
    def evaluate(self):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            # Use a subset of data for evaluation
            eval_batches = min(50, len(self.dataloader))
            
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= eval_batches:
                    break
                
                loss, _ = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return avg_loss, perplexity
    
    def generate_sample(self, prompt: str = "In the beginning", max_length: int = 100):
        """Generate sample text to test model"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
        logger.info(f"Sample generation:\nPrompt: {prompt}\nGenerated: {generated_text}")
        return generated_text
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"{checkpoint_name}.pt"
        
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        model_path = self.output_dir / "mvlm_final"
        
        # Save model and tokenizer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save training configuration and history
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'training_history': self.training_history,
                'final_step': self.global_step,
                'final_epoch': self.epoch
            }, f, indent=2)
        
        logger.info(f"Final model saved: {model_path}")
        return model_path
    
    def create_training_plots(self):
        """Create training visualization plots"""
        if not self.training_history:
            return
        
        # Extract data
        steps = [h['global_step'] for h in self.training_history]
        losses = [h['loss'] for h in self.training_history]
        base_losses = [h['base_loss'] for h in self.training_history]
        learning_rates = [h['learning_rate'] for h in self.training_history]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(steps, losses, label='Weighted Loss', color='blue')
        ax1.plot(steps, base_losses, label='Base Loss', color='red', alpha=0.7)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(steps, learning_rates, color='green')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        # Loss smoothed
        if len(losses) > 10:
            window = min(10, len(losses) // 10)
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            ax3.plot(smoothed_steps, smoothed_losses, color='purple')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Smoothed Loss')
            ax3.set_title(f'Smoothed Training Loss (window={window})')
            ax3.grid(True, alpha=0.3)
        
        # Training progress
        epochs = [h['epoch'] for h in self.training_history]
        epoch_losses = {}
        for h in self.training_history:
            epoch = h['epoch']
            if epoch not in epoch_losses:
                epoch_losses[epoch] = []
            epoch_losses[epoch].append(h['loss'])
        
        epoch_avg_losses = [np.mean(losses) for losses in epoch_losses.values()]
        ax4.bar(range(len(epoch_avg_losses)), epoch_avg_losses, color='orange', alpha=0.7)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Average Loss')
        ax4.set_title('Average Loss per Epoch')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved: {plot_path}")
    
    def train(self):
        """Complete training pipeline"""
        logger.info("Starting MVLM training...")
        logger.info(f"Dataset size: {len(self.dataset)} examples")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # Train epoch
            epoch_loss = self.train_epoch()
            
            # Evaluate
            eval_loss, perplexity = self.evaluate()
            
            # Generate sample
            self.generate_sample()
            
            # Save best model
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.save_checkpoint("best_model")
                logger.info(f"New best model saved with loss: {eval_loss:.4f}")
            
            logger.info(f"Epoch {epoch + 1} completed - Train Loss: {epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        
        # Training completed
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save final model
        final_model_path = self.save_final_model()
        
        # Create training plots
        self.create_training_plots()
        
        # Final evaluation
        logger.info("\n=== Final Evaluation ===")
        final_loss, final_perplexity = self.evaluate()
        
        # Generate final samples
        logger.info("\n=== Final Sample Generations ===")
        biblical_prompts = [
            "In the beginning",
            "The Lord is my shepherd",
            "Faith is the substance",
            "Love is patient and kind",
            "Trust in the Lord"
        ]
        
        for prompt in biblical_prompts:
            self.generate_sample(prompt, max_length=150)
        
        # Training summary
        logger.info("\n" + "="*60)
        logger.info("MVLM TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        logger.info(f"Total training steps: {self.global_step}")
        logger.info(f"Final training loss: {epoch_loss:.4f}")
        logger.info(f"Final evaluation loss: {final_loss:.4f}")
        logger.info(f"Final perplexity: {final_perplexity:.2f}")
        logger.info(f"Best loss achieved: {self.best_loss:.4f}")
        logger.info(f"Model saved to: {final_model_path}")
        logger.info("="*60)
        
        return final_model_path

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MVLM on biblical dataset')
    parser.add_argument('--data_dir', type=str, default='mvlm_comprehensive_dataset',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='mvlm_output',
                       help='Directory to save trained model')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients before optimizing')
    parser.add_argument('--emb_dropout', type=float, default=0.1,
                       help='Embedding dropout rate')
    parser.add_argument('--resid_dropout', type=float, default=0.1,
                       help='Residual dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                       help='Attention dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping (set to 0 to disable)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = MVLMConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.max_length = args.max_length
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.emb_dropout = args.emb_dropout
    config.resid_dropout = args.resid_dropout
    config.attn_dropout = args.attn_dropout
    config.label_smoothing = args.label_smoothing
    config.max_grad_norm = None if args.max_grad_norm == 0 else args.max_grad_norm
    
    # Check for GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        logger.warning("Please use a GPU instance for reasonable training times.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB memory)")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Please ensure the training dataset is available.")
        return
    
    # Create trainer and start training
    trainer = MVLMTrainer(config, args.data_dir, args.output_dir)
    final_model_path = trainer.train()
    
    logger.info(f"\nMVLM training completed successfully!")
    logger.info(f"Trained model available at: {final_model_path}")
    logger.info(f"Training logs saved to: mvlm_training.log")
    logger.info(f"Training plots saved to: {args.output_dir}/training_plots.png")

if __name__ == "__main__":
    main()

