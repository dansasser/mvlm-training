"""Training utilities for the Prioritary MVLM.

This module provides a lightweight trainer that mirrors the behaviour of
``MVLMTrainer`` in the repository.  It supports multi‑epoch training with
gradient accumulation, gradient clipping and an AdamW optimizer with a simple
linear warmup schedule.  Training examples can carry metadata scores that
indicate their priority; these scores are used to scale the loss and emphasise
important samples.

The trainer also includes helpers for evaluation, text generation and
checkpoint management.  Logging is performed to the console and, optionally, a
file so that training progress can be monitored easily.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import GPT2Config

from .config import PrioritaryConfig
from .dataset import WeightedTextDataset
from .model import PrioritaryMVLM
from .tokenizer import PrioritaryTokenizer


class PrioritaryTrainer:
    """Trainer for ``PrioritaryMVLM`` models."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: Optional[PrioritaryConfig] = None,
        log_file: str | None = None,
    ) -> None:
        """Initialise the trainer and all required components.

        Parameters
        ----------
        data_dir:
            Directory containing ``.txt`` and corresponding ``.json`` metadata
            files.
        output_dir:
            Directory where checkpoints will be written.
        config:
            Optional training configuration.  When omitted a default
            :class:`PrioritaryConfig` instance is created.
        log_file:
            Optional path to a log file for persisting metrics.
        """

        self.config = config or PrioritaryConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        for h in handlers:
            h.setFormatter(fmt)
            self.logger.addHandler(h)

        # ------------------------------------------------------------------
        # Tokenizer and model
        self.tokenizer = PrioritaryTokenizer()
        # Ensure the configuration matches the tokenizer vocabulary
        self.config.vocab_size = len(self.tokenizer)

        model_cfg = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            n_positions=self.config.max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.model = PrioritaryMVLM(model_cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ------------------------------------------------------------------
        # Dataset and dataloader
        self.dataset = WeightedTextDataset(data_dir, self.tokenizer, self.config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # ------------------------------------------------------------------
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=max(1, self.config.warmup_steps),
        )

        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------
    def compute_weighted_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted and base loss for a batch.

        ``WeightedTextDataset`` supplies a list of metadata dictionaries for
        each batch under the ``"metadata"`` key.  Each dictionary may contain a
        ``"priority"`` or ``"score"`` field.  Higher numbers indicate that the
        sample should contribute more strongly to the training signal.
        """

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.model(input_ids=input_ids, labels=labels)
        base_loss = outputs.loss

        priorities = [m.get("priority", m.get("score", 1.0)) for m in batch["metadata"]]
        weight = sum(priorities) / len(priorities)
        weighted_loss = base_loss * weight
        return weighted_loss, base_loss

    # ------------------------------------------------------------------
    def train(self, num_epochs: Optional[int] = None) -> None:
        """Execute the training loop."""

        epochs = num_epochs or self.config.num_epochs
        for _ in range(epochs):
            self.logger.info(
                f"Starting epoch {self.epoch + 1}/{self.epoch + epochs}"
            )
            epoch_loss = 0.0
            step_in_epoch = 0
            self.model.train()

            for batch in self.dataloader:
                loss, base_loss = self.compute_weighted_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (step_in_epoch + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.log_interval == 0:
                        self.logger.info(
                            "step=%d loss=%.4f base_loss=%.4f",  # pylint: disable=logging-format-interpolation
                            self.global_step,
                            loss.item() * self.config.gradient_accumulation_steps,
                            base_loss.item(),
                        )

                    if self.global_step % self.config.eval_interval == 0:
                        eval_loss, ppl = self.evaluate()
                        self.logger.info(
                            "eval_loss=%.4f perplexity=%.2f", eval_loss, ppl
                        )
                        self.save_checkpoint(
                            f"checkpoint_step{self.global_step}.pt"
                        )

                epoch_loss += loss.item()
                step_in_epoch += 1

            avg_epoch_loss = epoch_loss / max(1, step_in_epoch)
            self.logger.info("Epoch %d complete. avg_loss=%.4f", self.epoch + 1, avg_epoch_loss)
            self.epoch += 1

    # ------------------------------------------------------------------
    def evaluate(self, max_batches: int = 100) -> Tuple[float, float]:
        """Evaluate the model on a subset of the training data."""

        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= max_batches:
                    break
                loss, _ = self.compute_weighted_loss(batch)
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(1, batches)
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    # ------------------------------------------------------------------
    def generate_sample(
        self, prompt: str = "In the beginning", max_length: int = 50
    ) -> str:
        """Generate a sample of text from the model."""

        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(output[0].tolist())

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str = "checkpoint.pt") -> Path:
        """Persist training state to ``output_dir``."""

        path = self.output_dir / name
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        torch.save(checkpoint, path)
        self.logger.info("Saved checkpoint to %s", path)
        return path

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str | Path) -> None:
        """Load a previously saved checkpoint."""

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.logger.info("Loaded checkpoint from %s", path)

