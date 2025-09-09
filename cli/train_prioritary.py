"""CLI for training Prioritary MVLM models."""

import argparse

from prioritary_mvlm.config import PrioritaryConfig
from prioritary_mvlm.trainer import PrioritaryTrainer



def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a model using PrioritaryTrainer"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints and final model will be saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride for creating training windows",
    )
    parser.add_argument(
        "--resume-from",
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a text sample after training",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In the beginning",
        help="Prompt for text generation",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()

    config = PrioritaryConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        stride=args.stride,
    )

    trainer = PrioritaryTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()

    if args.eval:
        eval_loss, perplexity = trainer.evaluate()
        trainer.logger.info(
            "Final evaluation loss=%.4f perplexity=%.2f", eval_loss, perplexity
        )

    if args.generate:
        sample = trainer.generate_sample(prompt=args.prompt)
        print(sample)


if __name__ == "__main__":
    main()
