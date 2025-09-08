# SIM-ONE Transformer

This module contains the proprietary **SIM-ONE** transformer architecture.
It implements governance-aware attention blocks with auxiliary heads for
policy, memory, and trace outputs. The model is designed for experimentation
and training within this repository and serves as the foundation for the
prioritary training package.

## Files

- `simone_model.py` – core transformer implementation.

The model can be imported via `from simone_transformer.simone_model import SIMONEModel`.
