#!/usr/bin/env python3
"""Entry-point for training the Enhanced SIM-ONE transformer."""

from prioritary_mvlm import PrioritaryConfig, EnhancedPrioritaryTrainer


def main() -> None:
    """Train the Enhanced SIM-ONE model."""
    cfg = PrioritaryConfig()
    
    # Use enhanced trainer with modern architecture
    trainer = EnhancedPrioritaryTrainer(
        data_dir="../mvlm_training_dataset_complete", 
        output_dir="simone_checkpoints", 
        config=cfg,
        use_mixed_precision=True,
        compile_model=True
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
