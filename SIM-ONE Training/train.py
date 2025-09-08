#!/usr/bin/env python3
"""Entry-point for training the SIM-ONE transformer."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from prioritary_mvlm import PrioritaryConfig, PrioritaryTrainer


def parse_args() -> Namespace:
    """Parse command-line arguments for training."""
    parser = ArgumentParser(description="Train the SIM-ONE transformer")
    default_data_dir = (
        Path(__file__).resolve().parent
        / "../mvlm_training_dataset_complete/mvlm_comprehensive_dataset"
    ).resolve()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to store model checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PrioritaryConfig()
    trainer = PrioritaryTrainer(
        data_dir=args.data_dir, output_dir=args.output_dir, config=cfg
    )
    trainer.train()


if __name__ == "__main__":
    main()
