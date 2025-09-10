# SIM-ONE MVLM Training Repository

**Mission:** Deliver energy-efficient language models through a dual-model pipeline trained on a hand-curated, Christian-authored corpus—selected for internal consistency and a shared pursuit of non-subjective truth (not a Bible-only dataset).

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset Overview](#dataset-overview)
- [Models](#models)
- [Training Pipeline](#training-pipeline)
- [Validation & Monitoring](#validation--monitoring)
- [Contribution](#contribution)
- [License](#license)
- [Support & Documentation](#support--documentation)

## Overview
SIM-ONE trains two complementary language models on a carefully curated Christian-authored dataset. The pipeline targets H200 GPUs and produces models for reasoning, governance, and general-purpose text generation.

## Repository Structure
- `/` – root utilities including [`train_all_models.py`](./train_all_models.py) and [`validate_models.py`](./validate_models.py)
- `SIM-ONE Training/` – enhanced components such as [`enhanced_train.py`](./SIM-ONE%20Training/enhanced_train.py), tokenizer, and loss utilities
  - [`prioritary_mvlm/enhanced_trainer.py`](./SIM-ONE%20Training/prioritary_mvlm/enhanced_trainer.py)
  - [`simone_transformer/enhanced_model.py`](./SIM-ONE%20Training/simone_transformer/enhanced_model.py)
- `mvlm_training_dataset_complete/` – dataset files and reports
- `models/` – output directories for trained weights

## Dataset Overview
Version 1.0 – July 29, 2025. See [full report](./mvlm_training_dataset_complete/MVLM_TRAINING_DATASET_FINAL_REPORT.md).

- 158 documents
- 17.55 M words
- 9.9/10 average quality
- 8.0/10 biblical alignment
- 89 % training readiness

| Category | Docs | Percent | Impact |
| --- | --- | --- | --- |
| Biblical & Classical Literature | 88 | 55.7 % | Moral and philosophical backbone |
| Technical Documentation | 28 | 17.7 % | Accurate technical communication |
| Educational Content | 24 | 15.2 % | Pedagogical patterns and historical context |
| Philosophical Works | 16 | 10.1 % | Logical reasoning grounded in truth |
| Historical & Scientific | 2 | 1.3 % | Factual grounding and perspective |

Broad coverage across technical, educational, philosophical, historical, and literary content supports general-purpose modeling. Because all authors pursue non-subjective truth, internal contradictions are minimal, yielding a low-noise corpus. The dataset is not a Bible AI; it is a high-coherence dataset suitable for wide-ranging tasks.

### Strategic Impact
This truth-aligned, low-noise corpus provides a foundation for energy-efficient AGI research and advances SIM-ONE’s cognitive governance goals.

## Models
Both models share the curated dataset.

### MVLM-GPT2
Baseline GPT-2 architecture trained via [`mvlm_trainer.py`](./mvlm_trainer.py). Output path: [`models/mvlm_gpt2/`](./models/mvlm_gpt2/).

| Metric | Value |
| --- | --- |
| Training Time | 2–3 h |
| GPU Memory | ~20–30 GB |
| Tokens/sec | ~1000 |

### Enhanced SIM-ONE
Modern transformer with RoPE, SwiGLU, RMSNorm, and governance heads, trained via [`SIM-ONE Training/enhanced_train.py`](./SIM-ONE%20Training/enhanced_train.py). Output path: [`models/simone_enhanced/`](./models/simone_enhanced/).

| Metric | Value |
| --- | --- |
| Training Time | 3–4 h |
| GPU Memory | ~30–40 GB |
| Tokens/sec | ~600 |

## Training Pipeline
1. Environment setup
   ```bash
   ./setup_environment.sh
   ```
2. Train all models
   ```bash
   python3 train_all_models.py
   ```
3. Train individual models
   ```bash
   # MVLM-GPT2
   python3 mvlm_trainer.py --data mvlm_training_dataset_complete/

   # Enhanced SIM-ONE
   cd "SIM-ONE Training" && python3 enhanced_train.py --data ../mvlm_training_dataset_complete/
   ```

## Validation & Monitoring
```bash
python3 validate_models.py
nvidia-smi
tail -f logs/h200_training_*.log
```
Logs are stored in [`logs/`](./logs/) and downloadable model archives appear in [`models_for_download/`](./models_for_download/).

## Contribution
Please read [`agents.md`](./agents.md) and [`claude.md`](./claude.md) before contributing.

Key practices:
- Maintain the dual-model structure
- Use the BPE tokenizer
- Enable H200 optimizations (mixed precision, Flash Attention)

## License
This project is licensed under the terms of the [MIT License](./LICENSE).

## Support & Documentation
- [MVLM_TRAINING_COMPLETE_GUIDE.md](./MVLM_TRAINING_COMPLETE_GUIDE.md)
- [MVLM_DEPLOYMENT_CHECKLIST.md](./MVLM_DEPLOYMENT_CHECKLIST.md)
- Additional SIM-ONE resources available through external documentation.
