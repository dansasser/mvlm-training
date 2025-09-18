"""Training utilities for the SIM-ONE transformer."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from .config import PrioritaryConfig
from .dataset import WeightedTextDataset
from .tokenizer import PrioritaryTokenizer
from simone_transformer import SIMONEModel
from .losses import compute_policy_loss, compute_memory_loss, compute_energy_loss


class PrioritaryTrainer:
    """Trainer for SIM-ONE models with priority-weighted loss."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: Optional[PrioritaryConfig] = None,
        log_file: str | None = None,
    ) -> None:
        self.config = config or PrioritaryConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        for h in handlers:
            h.setFormatter(fmt)
            self.logger.addHandler(h)

        # Tokenizer and model
        self.tokenizer = PrioritaryTokenizer()
        self.model = SIMONEModel(
            vocab_size=len(self.tokenizer),
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_length,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Dataset and dataloader
        self.dataset = WeightedTextDataset(data_dir, self.tokenizer, self.config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )

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
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        logits, aux = self.model(input_ids)
        vocab_size = len(self.tokenizer)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, :-1].contiguous()

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None or pad_token_id == eos_token_id:
            ignore_index = -100
        else:
            ignore_index = pad_token_id

        if shifted_logits.size(1) == 0:
            mle_loss = logits.sum() * 0.0
        else:
            mle_loss = F.cross_entropy(
                shifted_logits.view(-1, vocab_size),
                shifted_labels.view(-1),
                ignore_index=ignore_index,
            )
        policy_loss = compute_policy_loss(aux["policy_logits"][-1])
        memory_loss = compute_memory_loss(aux["memory_signals"][-1])
        energy_loss = compute_energy_loss(logits)

        total = (
            mle_loss
            + self.config.lambda_policy * policy_loss
            + self.config.lambda_memory * memory_loss
            + self.config.lambda_energy * energy_loss
        )


        metadata = batch["metadata"]
        if isinstance(metadata, dict):
            if isinstance(metadata.get("priority"), torch.Tensor):
                weight = metadata.get("priority").float().mean().item()
            else:
                weight = float(metadata.get("priority", metadata.get("score", 1.0)))
        else:
            priorities = [m.get("priority", m.get("score", 1.0)) for m in metadata]
            weight = sum(priorities) / len(priorities)
        weighted_loss = total * weight
        return weighted_loss, mle_loss

    # ------------------------------------------------------------------
    def train(self, num_epochs: Optional[int] = None) -> None:
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
                            "step=%d loss=%.4f mle=%.4f",
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
    def save_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
            "step": self.global_step,
            "epoch": self.epoch,
        }, path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.scheduler.load_state_dict(data["scheduler"])
        self.global_step = data["step"]
        self.epoch = data["epoch"]