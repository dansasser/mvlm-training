"""CLI for training Prioritary models"""

import argparse
import inspect
from prioritary.trainer import PrioritaryTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a model using PrioritaryTrainer"
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to the training data set",
    )
    parser.add_argument(
        "--val-data",
        help="Optional path to the validation data set",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints and final model will be saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--resume-from",
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trainer = PrioritaryTrainer(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )

    # If resuming, try to use a dedicated method if available
    if args.resume_from:
        if hasattr(trainer, "load_checkpoint"):
            trainer.load_checkpoint(args.resume_from)
        elif hasattr(trainer, "resume_from_checkpoint"):
            trainer.resume_from_checkpoint(args.resume_from)

    train_kwargs = {}
    if args.resume_from:
        signature = inspect.signature(trainer.train)
        for key in (
            "resume_from",
            "resume_from_checkpoint",
            "checkpoint_path",
        ):
            if key in signature.parameters:
                train_kwargs[key] = args.resume_from
                break

    trainer.train(**train_kwargs)


if __name__ == "__main__":
    main()
