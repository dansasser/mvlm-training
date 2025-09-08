# SIM-ONE Transformer

## Overview

The SIM-ONE Transformer is a **proprietary language model architecture** designed around the principles of *Governed Cognition*. Unlike legacy GPT-style models, SIM-ONE is not a clone of GPT-2 or any other open model. It is a **custom transformer engine** built to encode governance, modularity, transparency, and efficiency into the very math of the system.

This repository provides the reference implementation of the SIM-ONE Transformer and its training package.

---

## Key Principles

* **Governed Cognition**: Every layer integrates governance hooks to ensure outputs comply with defined rules.
* **Modularity**: The transformer acts only as a *pattern recognizer*, leaving higher-order reasoning to external governance modules.
* **Transparency**: The model exposes traceable intermediate states through auxiliary heads.
* **Efficiency**: FLOPs regularization and activation sparsity reduce energy use per token.
* **Memory Integration**: Native read/write signals support external memory rather than burying all knowledge in dense weights.

---

## Architecture

* **Core block**: Decoder-style transformer with governance-gated self-attention and shaped logits.
* **Auxiliary heads**:

  * **Policy head** → enforces rule compliance.
  * **Memory head** → interfaces with external memory.
  * **Trace head** → generates interpretable internal states.
* **Multi-objective loss**:

  $$
  L = L_{\text{MLE}} + \lambda_1 L_{\text{policy}} + \lambda_2 L_{\text{memory}} + \lambda_3 L_{\text{energy}}
  $$
* **Tokenizer**: Custom SIM-ONE vocabulary, including reserved governance tokens.

---

## Files

* `simone_model.py` → Core SIM-ONE transformer implementation.
* `losses.py` → Multi-objective loss functions (policy, memory, energy).
* `train.py` → Training loop with toy and production support.
* `README.md` → This document.

---

## Training

The training loop integrates governance directly into optimization. Example:

```python
logits, aux = model(input_ids, mask=attn_mask, policy_mask=policy_mask)

mle_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
policy_loss = compute_policy_loss(aux["policy_logits"][-1])
memory_loss = compute_memory_loss(aux["memory_signals"][-1])
energy_loss = compute_energy_loss(logits)

loss = mle_loss + \lambda_1*policy_loss + \lambda_2*memory_loss + \lambda_3*energy_loss
```

---

## Benchmarks

Evaluation compares SIM-ONE to vanilla GPT-2:

* **Perplexity**: Baseline language modeling.
* **Policy compliance**: Rule violation rate.
* **Energy efficiency**: FLOPs per token, activation sparsity.

---

## Intellectual Property

This codebase is **not derived from GPT-2**. It is a fresh implementation based on the SIM-ONE philosophy, with custom governance hooks and training objectives. All rights reserved to the authors and affiliates of the SIM-ONE Framework.
