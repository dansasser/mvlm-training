#!/usr/bin/env python3
"""Prioritary MVLM training utilities."""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

try:
    from prioritary_mvlm import PrioritaryMVLM
except ImportError:  # pragma: no cover - fallback for absent dependency
    class PrioritaryMVLM(GPT2LMHeadModel):
        """Fallback PrioritaryMVLM model using GPT-2 architecture."""
        def __init__(self, config: GPT2Config):
            super().__init__(config)


class PrioritaryTrainer:
    """Trainer specialized for PrioritaryMVLM models."""

    def __init__(self, config, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.create_model()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Dataset and loader
        from mvlm_trainer import BiblicalTextDataset  # lazy import to avoid circular deps
        self.dataset = BiblicalTextDataset(
            data_dir,
            self.tokenizer,
            max_length=self.config.max_length,
            stride=self.config.stride,
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=self.config.warmup_steps
        )

        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------
    def create_model(self) -> PrioritaryMVLM:
        """Instantiate a PrioritaryMVLM model using the trainer configuration."""
        cfg = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.n_positions,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=self.config.n_inner,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50256,
        )
        model = PrioritaryMVLM(cfg)
        return model

    # ------------------------------------------------------------------
    def compute_weighted_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss emphasizing high priority samples.

        Each batch item may provide a ``priority`` score in its metadata.  The
        score defaults to ``1.0`` when missing.  Higher priority examples
        contribute more to the training signal.
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(input_ids=input_ids, labels=labels)
        base_loss = outputs.loss

        priorities = [m.get("priority", 1.0) for m in batch["metadata"]]
        avg_priority = sum(priorities) / len(priorities)
        weighted_loss = base_loss * avg_priority
        return weighted_loss, base_loss

    # ------------------------------------------------------------------
    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_base = 0.0
        batches = 0

        for batch in self.dataloader:
            loss, base_loss = self.compute_weighted_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            if self.global_step < self.config.warmup_steps:
                self.scheduler.step()

            total_loss += loss.item()
            total_base += base_loss.item()
            batches += 1
            self.global_step += 1

        return total_loss / max(1, batches)

    # ------------------------------------------------------------------
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on a subset of the training data."""
        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= min(50, len(self.dataloader)):
                    break
                loss, _ = self.compute_weighted_loss(batch)
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(1, batches)
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    # ------------------------------------------------------------------
    def generate_sample(self, prompt: str = "In the beginning", max_length: int = 50) -> str:
        """Generate sample text using the trained model."""
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

