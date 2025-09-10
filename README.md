# SIM-ONE MVLM Training Repository

**Mission:** Dual-model training pipeline leveraging a curated corpus of Christian-authored works (not a Bible-only dataset) to build energy-efficient MVLMs for the SIM-ONE framework.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
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
- `SIM-ONE Training/` – enhanced components such as [`enhanced_train.py`](./SIM-ONE%20Training/enhanced_train.py) and tokenizer/loss utilities
  - `prioritary_mvlm/enhanced_trainer.py`
  - `simone_transformer/enhanced_model.py`
- `mvlm_training_dataset_complete/` – dataset files and reports
- `models/` – output directories for trained weights

## Dataset
Key metrics from [`mvlm_training_dataset_complete/MVLM_TRAINING_DATASET_FINAL_REPORT.md`](./mvlm_training_dataset_complete/MVLM_TRAINING_DATASET_FINAL_REPORT.md):
- 158 documents
- 17.55 M words
- 9.9/10 average quality
- 8.0/10 biblical alignment
- 89 % training readiness
- Content spans five categories with major contributors such as John MacArthur, Charles Stanley, C.S. Lewis, and G.K. Chesterton

The dataset is curated from Christian authors with low internal contradiction and is explicitly not a Bible-only corpus.

### Strategic Impact
This corpus functions as a "truth filter" for energy-efficient AGI research. By grounding training data in consistent Christian authorship, SIM-ONE pursues cognitive governance while reducing computational overhead.

## Models
Both models share the same curated dataset.

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
Please read [`agents.md`](./agents.md) and [`claude.md`](./claude.md) before contributing. Key practices:
- Maintain the dual-model structure
- Use the BPE tokenizer
- Enable H200 optimizations (mixed precision, Flash Attention)

## License
This project is licensed under the terms of the [MIT License](./LICENSE).

## Support & Documentation
- [MVLM_TRAINING_COMPLETE_GUIDE.md](./MVLM_TRAINING_COMPLETE_GUIDE.md)
- [MVLM_DEPLOYMENT_CHECKLIST.md](./MVLM_DEPLOYMENT_CHECKLIST.md)
- Additional SIM-ONE resources available through external documentation.

