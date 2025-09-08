import torch
import torch.nn.functional as F


def compute_policy_loss(policy_logits, policy_targets=None):
    if policy_targets is None:
        return torch.tensor(0.0, requires_grad=True)
    return F.mse_loss(policy_logits, policy_targets)


def compute_memory_loss(memory_signals, memory_targets=None):
    if memory_targets is None:
        return torch.tensor(0.0, requires_grad=True)
    return F.mse_loss(memory_signals, memory_targets)


def compute_energy_loss(hidden_states, target_energy=None):
    l1_penalty = torch.mean(torch.abs(hidden_states))
    if target_energy is not None:
        energy_penalty = torch.mean((hidden_states.pow(2).mean() - target_energy).abs())
        return l1_penalty + energy_penalty
    return l1_penalty
