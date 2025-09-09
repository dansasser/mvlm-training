#!/usr/bin/env python3
"""
Enhanced training entry-point for the SIM-ONE transformer.
Uses all modern improvements: BPE tokenizer, RoPE attention, SwiGLU, advanced losses, etc.
"""

import argparse
import sys
from pathlib import Path

from prioritary_mvlm import PrioritaryConfig
from prioritary_mvlm.enhanced_trainer import EnhancedPrioritaryTrainer


def create_enhanced_config(args) -> PrioritaryConfig:
    """Create enhanced configuration with modern defaults."""
    config = PrioritaryConfig()
    
    # Update with command line arguments
    if args.vocab_size:
        config.vocab_size = args.vocab_size
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.ff_dim:
        config.ff_dim = args.ff_dim
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_length:
        config.max_length = args.max_length
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    
    # Enhanced defaults for modern training
    config.weight_decay = 0.1  # Higher weight decay
    config.max_grad_norm = 1.0
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.log_interval = 10
    config.eval_interval = 100
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced SIM-ONE Transformer Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="../mvlm_training_dataset_complete",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="enhanced_checkpoints",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to existing tokenizer (will train new one if not provided)"
    )
    
    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=3072, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    
    # Enhanced features
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    
    # Logging
    parser.add_argument("--log_file", type=str, help="Log file path")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please ensure the training dataset is available.")
        sys.exit(1)
    
    # Create enhanced configuration
    config = create_enhanced_config(args)
    
    print("Enhanced SIM-ONE Transformer Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Model size: {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A")
    print(f"Sequence length: {config.max_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Training epochs: {config.num_epochs}")
    print(f"Mixed precision: {not args.no_mixed_precision}")
    print(f"Model compilation: {not args.no_compile}")
    print("=" * 50)
    
    try:
        # Create enhanced trainer
        trainer = EnhancedPrioritaryTrainer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            tokenizer_path=args.tokenizer_path,
            log_file=args.log_file,
            use_mixed_precision=not args.no_mixed_precision,
            compile_model=not args.no_compile
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Resuming training from: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        
        # Run training
        final_model_path = trainer.train()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final model saved to: {final_model_path}")
        print(f"Tokenizer saved to: {args.output_dir}/tokenizer.pkl")
        print(f"Training plots: {args.output_dir}/enhanced_training_plots.png")
        print(f"Training history: {args.output_dir}/training_history.json")
        
        # Model statistics
        num_params = trainer.model.get_num_params()
        memory_usage = trainer.model.get_memory_usage()
        print(f"\nModel Statistics:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Memory usage: {memory_usage['total'] / 1e6:.2f} MB")
        print(f"  Best validation loss: {trainer.best_loss:.4f}")
        print(f"  Total training steps: {trainer.global_step:,}")
        
        print("\nEnhanced SIM-ONE training completed! 🎉")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()