#!/usr/bin/env python3
"""Entry-point for training the SIM-ONE transformer."""

from prioritary_mvlm import PrioritaryConfig, PrioritaryTrainer


def main() -> None:
    cfg = PrioritaryConfig()
    trainer = PrioritaryTrainer(data_dir="data", output_dir="checkpoints", config=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
