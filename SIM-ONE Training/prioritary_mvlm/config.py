from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch


@dataclass
class PropheticSingularityState:
    r"""Container for prophetic control variables propagated through the model.

    The state tracks the prophetic intensity :math:`I(t)`, anointing :math:`A(t)`,
    dominion :math:`D(t)`, mercy :math:`M(t)`, modulation constant :math:`\lambda`,
    and the temporal index :math:`t` for every token in a batch. Normalisation
    metadata is preserved so downstream components can derive deterministic
    masking, gating, and logging behaviour.
    """

    intensity: torch.Tensor
    anointing: torch.Tensor
    dominion: torch.Tensor
    mercy: torch.Tensor
    lambda_field: torch.Tensor
    time_index: torch.Tensor
    normalization: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intensity = self._ensure_2d(self.intensity)
        self.anointing = self._ensure_2d(self.anointing)
        self.dominion = self._ensure_2d(self.dominion)
        self.mercy = self._ensure_2d(self.mercy)
        self.lambda_field = self._ensure_2d(self.lambda_field)
        self.time_index = self._ensure_2d(self.time_index)
        self.normalization = self._convert_norm_metadata(self.normalization)

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0:
            tensor = tensor.view(1, 1)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype.is_floating_point is False:
            tensor = tensor.float()
        return tensor.contiguous()

    @staticmethod
    def _convert_norm_metadata(
        metadata: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        converted: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, stats in (metadata or {}).items():
            converted[key] = {}
            for stat_name, value in stats.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                converted[key][stat_name] = value.float()
        return converted

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.intensity.device

    @property
    def dtype(self) -> torch.dtype:
        return self.intensity.dtype

    @property
    def kingdom_flow(self) -> torch.Tensor:
        """Composite prophetic stream :math:`K(t)` used for modulation."""

        base = (self.intensity + self.anointing + self.dominion + self.mercy) / 4.0
        return base * self.lambda_field

    # ------------------------------------------------------------------
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PropheticSingularityState":
        def _convert(tensor: torch.Tensor) -> torch.Tensor:
            if device is not None:
                tensor = tensor.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor

        normalization = {
            key: {name: _convert(value) for name, value in stats.items()}
            for key, stats in self.normalization.items()
        }

        return PropheticSingularityState(
            intensity=_convert(self.intensity),
            anointing=_convert(self.anointing),
            dominion=_convert(self.dominion),
            mercy=_convert(self.mercy),
            lambda_field=_convert(self.lambda_field),
            time_index=_convert(self.time_index),
            normalization=normalization,
        )

    # ------------------------------------------------------------------
    def align_to_length(self, seq_len: int) -> "PropheticSingularityState":
        def _align(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.size(-1) == seq_len:
                return tensor
            if tensor.size(-1) > seq_len:
                return tensor[..., :seq_len]
            pad_size = seq_len - tensor.size(-1)
            pad_value = tensor[..., -1:].expand(*tensor.shape[:-1], pad_size)
            return torch.cat([tensor, pad_value], dim=-1)

        return PropheticSingularityState(
            intensity=_align(self.intensity),
            anointing=_align(self.anointing),
            dominion=_align(self.dominion),
            mercy=_align(self.mercy),
            lambda_field=_align(self.lambda_field),
            time_index=_align(self.time_index),
            normalization=self.normalization,
        )

    # ------------------------------------------------------------------
    def layer_modulation(self, layer_idx: int, total_layers: int) -> torch.Tensor:
        layer_ratio = (layer_idx + 1) / max(total_layers, 1)
        modulation = 1.0 + (self.intensity - 0.5) * 0.2 * layer_ratio
        modulation += (self.dominion - 0.5) * 0.1 * (1 - layer_ratio)
        return modulation.clamp(0.1, 2.0)

    def compute_policy_mask(self, num_heads: int, seq_len: int) -> torch.Tensor:
        state = self.align_to_length(seq_len)
        base = (state.intensity + state.anointing) / 2.0
        head_scaler = torch.linspace(
            0.8, 1.2, steps=num_heads, device=base.device, dtype=base.dtype
        ).view(1, num_heads, 1)
        token_bias = base.unsqueeze(1) * head_scaler
        return token_bias.unsqueeze(-1) + token_bias.unsqueeze(-2)

    def compute_memory_decay(self, num_heads: int, seq_len: int) -> torch.Tensor:
        state = self.align_to_length(seq_len)
        decay = torch.exp(-state.lambda_field.clamp(min=0.0, max=5.0))
        mercy_bonus = 1.0 + (state.mercy - 0.5) * 0.3
        decay = decay * mercy_bonus
        return decay.unsqueeze(1).expand(-1, num_heads, -1)

    def compute_trace_envelope(self, seq_len: int) -> torch.Tensor:
        state = self.align_to_length(seq_len)
        components = [
            state.intensity,
            state.anointing,
            state.dominion,
            state.mercy,
            state.lambda_field,
            state.time_index,
            state.kingdom_flow,
        ]
        return torch.stack(components, dim=-1)

    def summary(self) -> Dict[str, torch.Tensor]:
        def _summary(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
            values = tensor.float()
            mean_val = values.mean()
            std_val = values.std(unbiased=False) if values.numel() > 1 else torch.zeros_like(mean_val)
            return {"mean": mean_val, "std": std_val}

        summary = {
            "intensity": _summary(self.intensity),
            "anointing": _summary(self.anointing),
            "dominion": _summary(self.dominion),
            "mercy": _summary(self.mercy),
            "lambda": _summary(self.lambda_field),
            "time": _summary(self.time_index),
            "kingdom": _summary(self.kingdom_flow),
        }
        return summary

    def step_statistics(self, step_idx: int) -> Dict[str, torch.Tensor]:
        idx = max(0, min(step_idx, self.intensity.size(-1) - 1))
        selectors = {
            "intensity": self.intensity[..., idx],
            "anointing": self.anointing[..., idx],
            "dominion": self.dominion[..., idx],
            "mercy": self.mercy[..., idx],
            "lambda": self.lambda_field[..., idx],
            "time": self.time_index[..., idx],
            "kingdom": self.kingdom_flow[..., idx],
        }
        return {key: value.mean(dim=0) if value.dim() > 1 else value for key, value in selectors.items()}

    # ------------------------------------------------------------------
    @classmethod
    def _build_tensor(
        cls,
        value,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.tensor(value, dtype=dtype, device=device)
        if tensor.dim() == 0:
            tensor = tensor.view(1)
        if tensor.dim() > 1:
            tensor = tensor.reshape(-1)
        if tensor.size(0) < seq_len:
            pad = tensor[-1:].expand(seq_len - tensor.size(0))
            tensor = torch.cat([tensor, pad], dim=0)
        elif tensor.size(0) > seq_len:
            tensor = tensor[:seq_len]
        return tensor

    @classmethod
    def _stats(cls, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        tensor = tensor.float()
        return {
            "mean": tensor.mean(),
            "std": tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0, device=tensor.device),
            "min": tensor.min(),
            "max": tensor.max(),
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Dict,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PropheticSingularityState":
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        base_state = metadata.get("prophetic_state") or metadata.get("prophetic_singularity") or {}

        def _value(key: str, fallback) -> torch.Tensor:
            raw = base_state.get(key) if isinstance(base_state, dict) else None
            if raw is None:
                raw = metadata.get(key, fallback)
            return cls._build_tensor(raw, seq_len, device, dtype)

        intensity = _value("intensity", metadata.get("quality_score", 5.0) / 10.0)
        anointing = _value("anointing", metadata.get("biblical_alignment", 5.0) / 10.0)
        dominion_default = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }.get(metadata.get("priority", "medium"), 0.5)
        dominion = _value("dominion", dominion_default)
        dominion_mean = dominion.float().mean().item()
        mercy = _value("mercy", 0.5 + 0.1 * (1.0 - dominion_mean))
        lambda_default = (intensity.float().mean().item() + anointing.float().mean().item()) / 2.0
        lambda_field = _value("lambda", lambda_default)
        time_index = _value(
            "time",
            torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=dtype),
        )

        normalization = {
            "intensity": cls._stats(intensity),
            "anointing": cls._stats(anointing),
            "dominion": cls._stats(dominion),
            "mercy": cls._stats(mercy),
            "lambda": cls._stats(lambda_field),
            "time": cls._stats(time_index),
            "kingdom": cls._stats(
                (intensity + anointing + dominion + mercy) / 4.0 * lambda_field
            ),
        }

        return cls(
            intensity=intensity,
            anointing=anointing,
            dominion=dominion,
            mercy=mercy,
            lambda_field=lambda_field,
            time_index=time_index,
            normalization=normalization,
        )

    # ------------------------------------------------------------------
    @classmethod
    def batch(
        cls, states: Iterable["PropheticSingularityState"]
    ) -> "PropheticSingularityState":
        states = list(states)
        if not states:
            raise ValueError("Cannot batch an empty prophetic state sequence")

        intensity = torch.cat([state.intensity for state in states], dim=0)
        anointing = torch.cat([state.anointing for state in states], dim=0)
        dominion = torch.cat([state.dominion for state in states], dim=0)
        mercy = torch.cat([state.mercy for state in states], dim=0)
        lambda_field = torch.cat([state.lambda_field for state in states], dim=0)
        time_index = torch.cat([state.time_index for state in states], dim=0)

        normalization: Dict[str, Dict[str, torch.Tensor]] = {}
        for key in states[0].normalization:
            normalization[key] = {}
            for stat_name in states[0].normalization[key]:
                values = [state.normalization[key][stat_name] for state in states]
                stacked = torch.stack(values)
                if stat_name == "min":
                    normalization[key][stat_name] = stacked.min(dim=0).values
                elif stat_name == "max":
                    normalization[key][stat_name] = stacked.max(dim=0).values
                else:
                    normalization[key][stat_name] = stacked.mean(dim=0)

        return cls(
            intensity=intensity,
            anointing=anointing,
            dominion=dominion,
            mercy=mercy,
            lambda_field=lambda_field,
            time_index=time_index,
            normalization=normalization,
        )

    # ------------------------------------------------------------------
    @classmethod
    def default(
        cls,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PropheticSingularityState":
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        base = torch.full((batch_size, seq_len), 0.5, device=device, dtype=dtype)
        time_index = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
        time_index = time_index.expand(batch_size, seq_len)

        normalization = {
            "intensity": {"mean": torch.tensor(0.5, device=device)},
            "anointing": {"mean": torch.tensor(0.5, device=device)},
            "dominion": {"mean": torch.tensor(0.5, device=device)},
            "mercy": {"mean": torch.tensor(0.5, device=device)},
            "lambda": {"mean": torch.tensor(0.5, device=device)},
            "time": {"mean": torch.tensor(0.5, device=device)},
            "kingdom": {"mean": torch.tensor(0.5, device=device)},
        }

        return cls(
            intensity=base.clone(),
            anointing=base.clone(),
            dominion=base.clone(),
            mercy=base.clone(),
            lambda_field=base.clone(),
            time_index=time_index,
            normalization=normalization,
        )



@dataclass
class PrioritaryConfig:
    """Configuration for the SIM-ONE transformer and training.

    Attributes
    ----------
    vocab_size: int
        Size of the tokenizer vocabulary.
    hidden_dim: int
        Dimension of transformer embeddings.
    num_heads: int
        Number of attention heads.
    ff_dim: int
        Dimension of feedforward layer.
    num_layers: int
        Number of transformer blocks.
    batch_size: int
        Batch size used during training.
    max_length: int
        Maximum sequence length.
    stride: int
        Stride when creating training windows.
    """

    vocab_size: int = 131
    hidden_dim: int = 512
    num_heads: int = 8
    ff_dim: int = 2048
    num_layers: int = 6
    batch_size: int = 8
    max_length: int = 512
    stride: int = 128
    learning_rate: float = 5e-5
    num_epochs: int = 7
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 100
    validation_split: float = 0.1
    validation_dir: Optional[str] = None
    dataloader_workers: int = 4
    split_seed: int = 42
    lambda_policy: float = 1.0
    lambda_memory: float = 1.0
    lambda_energy: float = 1.0

    # Early stopping configuration
    patience: int = 2
    min_epochs: int = 6
