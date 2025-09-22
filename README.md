# SIM-ONE MVLM Training Repository

## Overview

This repository contains the complete training pipeline for **SIM-ONE MVLM (Minimum Viable Language Model)** - a revolutionary approach to training truth-leaning AI through **singular source consistency**. The training leverages a carefully curated dataset where all authors share a consistent worldview, creating an exceptionally low-noise corpus that enables efficient learning of coherent reasoning patterns.

**Purpose**: Train production-ready truth-leaning MVLM for The SIM-ONE Framework ecosystem
**Architecture**: Enhanced SIM-ONE transformer optimized for H200 GPU deployment
**Training Foundation**: Singular truth source dataset with minimal contradictions across 7 domains
**Output**: Enhanced SIM-ONE model with governance mechanisms (Legacy MVLM-GPT2 deprecated)  

## 📄 Research Paper

**[Coherent Worldview Training: A Data Quality Approach to Language Model Development](COHERENT_WORLDVIEW_TRAINING_PAPER.md)**

This repository implements the methodology described in our research paper, which introduces **Coherent Worldview Training (CWT)** - a novel approach to training data curation that uses epistemological consistency as a quality filter. The paper explains the technical foundations, architectural innovations, and performance improvements achieved through this approach.

**Key Contributions:**
- Novel data curation methodology using shared worldview as quality filter
- Enhanced SIM-ONE transformer architecture with governance mechanisms  
- Measurable improvements in reasoning consistency and reduced contradictions
- Scalable framework applicable to other domains and worldview systems

## The SIM-ONE Ecosystem

These trained models are integral components of **The SIM-ONE Framework**, serving as the core text generation engines that power:

- **Advanced reasoning systems** with governance mechanisms
- **Multi-modal processing** with vision-language integration  
- **Production AI applications** requiring high-quality text generation
- **Research platforms** for advanced language model development

The models trained here provide the foundational text generation capabilities that The SIM-ONE Framework builds upon for its advanced AI functionalities.

## Revolutionary Methodology: Truth-Leaning AI

### Core Innovation: Singular Truth Source Training

This is **NOT a biblical AI** but rather a **truth-leaning MVLM** that demonstrates a revolutionary training methodology:

**Key Principle**: All 1,226 training files come from authors sharing a **singular source of truth**, creating:
- **Minimal contradictions** across all domains
- **Consistent reasoning patterns** throughout the corpus
- **Natural truth-leaning bias** without explicit programming
- **Cross-domain coherence** from literature to technical documentation

### Dataset Composition (ALL 6 DOMAINS)

**1,226 Files Across 7 Writing Styles:**
- **Classical Literature**: 1,083 files (Shakespeare, Dickens, virtue/character works)
- **Educational Content**: 28 files (history, communication, philosophy/ethics)
- **Theological Exposition**: 73 files (deep theological reasoning)
- **Historical/Scientific**: 24 files (foundational documents, scientific principles)
- **Philosophical Works**: 16 files (classical to modern philosophy)
- **Technical Documentation**: 2 files (Enterprise Architecture + Chemistry)

### Enhanced SIM-ONE Architecture

**Location**: `SIM-ONE Training/enhanced_train.py`
**Modern Transformer Features**:
- **RoPE (Rotary Position Embedding)**: Superior position encoding
- **SwiGLU Activation**: ~10-15% performance improvement
- **RMSNorm**: Enhanced training stability
- **Advanced BPE Tokenizer**: 32K vocabulary optimized for semantic preservation
- **Governance Mechanisms**: Policy, memory, and trace attention heads
- **Advanced Loss Functions**: Multi-objective optimization for content quality

**Training Configuration:**
```
Epochs: 6-7 (minimum 6 guaranteed, early stopping at 7)
Training Time: ~24 hours on H200 GPU
GPU Memory: ~30-40GB
Performance: ~600 tokens/sec (truth-leaning quality)
Output Size: ~3-4GB model
Cost: $72-120 (at $3-5/hour cloud rates)
Architecture: Modern transformer with governance and reasoning
```

## Technical Innovations

### Advanced Tokenization
- **High-Quality BPE**: 32,000 vocabulary tokens optimized for semantic units
- **10-100x Speedup**: Over character-level tokenization approaches
- **Semantic Preservation**: Important concepts maintained as single tokens
- **Training Efficiency**: Reduced sequence lengths for faster training

### Modern Architecture Enhancements
- **RoPE Attention**: Rotary position embeddings for better sequence understanding
- **SwiGLU Feedforward**: Advanced activation functions for improved performance
- **RMSNorm**: Root Mean Square normalization for training stability
- **Flash Attention**: Memory-efficient attention computation where available
- **KV Caching**: Optimized autoregressive generation

### Governance System (Enhanced SIM-ONE)
- **Policy Head**: Ethical decision-making and reasoning guidance
- **Memory Head**: Long-term context retention and knowledge integration
- **Trace Head**: Reasoning pathway tracking and explainability
- **Multi-Head Coordination**: Integrated governance across attention mechanisms

### Advanced Loss Functions
- **Content Alignment Loss**: Ensures high-quality content consistency
- **Coherence Loss**: Maintains narrative and logical flow
- **Accuracy Loss**: Preserves factual knowledge and relationships
- **Comprehensive Loss**: Multi-objective optimization balancing all criteria

## H200 GPU Optimization

### Performance Optimizations
- **Mixed Precision Training**: FP16/BF16 for 40-50% memory reduction
- **Gradient Scaling**: Stable mixed precision with automatic loss scaling
- **Model Compilation**: PyTorch 2.0+ compilation for inference speed
- **Flash Attention Integration**: Memory-efficient attention where supported
- **Optimized Memory Management**: CUDA memory allocation strategies

### Environment Configuration
```bash
# H200 GPU Optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## Quick Start

### Prerequisites
- NVIDIA H200 GPU (or compatible 80GB+ VRAM GPU)
- CUDA 11.8+ or 12.1+
- Python 3.8+
- 50GB+ free disk space
- Internet connection for initial setup

### Automated Training (Recommended)
```bash
# 1. Clone repository
git clone <repository-url>
cd <repository-directory>

# 2. Setup virtual environment and dependencies (10 minutes)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify configuration
python3 verify_complete_setup.py

# 4. Start web monitoring dashboard (optional but recommended)
python3 training_monitor.py &  # Access at http://localhost:5001

# 5. Train Enhanced SIM-ONE across ALL 6 domains (~24 hours)
python3 train_all_models.py

# 5. Validate trained model (5 minutes)
python3 validate_models.py

# 6. Download compressed model
ls models_for_download/
# Download: simone_enhanced_model.tar.gz
```

### Manual Training (Advanced)
```bash
# Train Enhanced SIM-ONE with truth-leaning dataset
cd "SIM-ONE Training"
python3 enhanced_train.py \
    --data_dir ../mvlm_training_dataset_complete \
    --output_dir ../models/simone_enhanced \
    --vocab_size 32000 \
    --hidden_dim 768 \
    --num_layers 12 \
    --batch_size 12 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --num_epochs 7 \
    --patience 2 \
    --min_epochs 6
```

### Key Training Features
- **All 6 Domains**: Automatically processes 1,226 files from all domains
- **Smart Early Stopping**: Minimum 6 epochs, stops at 7 if no improvement
- **Truth-Leaning Optimization**: Leverages singular truth source consistency
- **H200 Optimized**: Mixed precision, Flash Attention, memory optimization
- **Real-time Monitoring**: Progress tracking with detailed metrics

## Revolutionary Impact

### Paradigm Shift: Singular Truth Source Training

This repository demonstrates that **consistency beats scale** in AI training:

**Traditional Approach**: Billions of contradictory tokens from diverse sources
**SIM-ONE Approach**: 1,226 carefully curated files from singular truth source

**Results**:
- **10,000x cost reduction**: $72-120 vs $500,000-5,000,000 for traditional training
- **Minimal contradictions**: Truth-leaning bias emerges naturally
- **Cross-domain coherence**: Literature to technical documentation consistency
- **Efficient learning**: 6-7 epochs vs 100+ epochs for comparable quality

### Proof of Concept

Your trained model will demonstrate:
- **Quality data curation** outperforms massive noisy datasets
- **Worldview consistency** enables efficient learning
- **Truth-leaning bias** develops without explicit programming
- **Governance mechanisms** work across all domains

This approach is **domain-agnostic** and applicable to other consistent worldview systems.

## Legacy Notice

**MVLM-GPT2 (Deprecated)**: The original GPT-2 based model is now legacy and will be removed in future versions. All development focuses on the Enhanced SIM-ONE architecture, which provides superior performance with modern transformer techniques and governance mechanisms.

## Repository Structure

```
SIM-ONE-MVLM-Training/
├── README.md                         # This file
├── agents.md                         # AI agent development guide  
├── claude.md                         # Claude AI assistant instructions
├── requirements.txt                  # Python dependencies
├── setup_environment.sh              # H200 environment setup
│
├── train_all_models.py               # Enhanced training orchestrator
├── validate_models.py                # Enhanced model validation
│
├── mvlm_training_dataset_complete/   # High-quality training corpus
│   ├── processed_texts/              # Preprocessed training data
│   └── metadata/                     # Dataset information
│
├── models/                           # Training outputs
│   └── simone_enhanced/             # Enhanced SIM-ONE model files
│
├── models_for_download/              # Compressed models for deployment
│   ├── simone_enhanced_model.tar.gz # Ready-to-deploy Enhanced SIM-ONE
│   └── training_summary.json        # Training statistics and metadata
│
├── logs/                             # Training logs and monitoring
│   ├── h200_training_*.log          # Main training logs
│   └── simone_enhanced_training.log  # Enhanced SIM-ONE logs
│
└── SIM-ONE Training/                 # Enhanced SIM-ONE components
    ├── train.py                     # Simple training entry point
    ├── enhanced_train.py            # Advanced training with CLI args
    │
    ├── prioritary_mvlm/             # Enhanced training framework
    │   ├── __init__.py
    │   ├── config.py               # Configuration management
    │   ├── enhanced_trainer.py     # H200-optimized trainer
    │   ├── advanced_tokenizer.py   # High-quality BPE tokenizer
    │   └── advanced_losses.py      # Multi-objective loss functions
    │
    └── simone_transformer/          # Enhanced model architecture
        ├── __init__.py
        ├── enhanced_model.py       # EnhancedSIMONEModel implementation
        ├── rope_attention.py       # RoPE attention + governance
        └── modern_layers.py        # SwiGLU, RMSNorm, advanced layers
```

## Training Data Quality

The training corpus (`mvlm_training_dataset_complete/`) consists of high-quality biblical text chosen for its exceptional characteristics:

### Why Biblical Text for MVLM Training?
- **Low Noise**: Centuries of careful editing and curation
- **Truth-Leaning Bias**: Consistent moral and factual framework
- **Rich Language Patterns**: Diverse vocabulary, narrative styles, and linguistic structures
- **Coherent Content**: Strong logical and thematic consistency
- **Cultural Significance**: Deep semantic relationships and knowledge patterns
- **Quality Control**: Well-edited, grammatically correct, and linguistically rich

### Training Advantages
- **Superior Coherence**: Models learn strong narrative and logical flow
- **Factual Grounding**: Training on content with consistent truth claims
- **Linguistic Richness**: Exposure to sophisticated language patterns
- **Reasoning Development**: Complex theological and philosophical concepts
- **Ethical Foundation**: Built-in moral reasoning patterns

### Performance Specifications

### Expected Training Performance (H200 GPU)
- Enhanced SIM-ONE: 3-4 hours, 30-40GB GPU memory, ~600 tokens/sec, final size ~3-4GB

### Model Capabilities
- **Text Generation**: High-quality, coherent text output
- **Reasoning**: Enhanced logical and ethical reasoning (Enhanced SIM-ONE)
- **Knowledge Retention**: Strong factual knowledge preservation
- **Context Understanding**: Superior long-range context handling
- **Inference Speed**: Optimized for production deployment

## Integration with SIM-ONE Framework

### Model Deployment Roles
- **MVLM-GPT2**: Baseline text generation for standard operations
- **Enhanced SIM-ONE**: Advanced reasoning and high-quality generation
- **Combined Use**: Complementary models for different SIM-ONE Framework needs
- **Production Ready**: Optimized for integration into larger AI systems

### Framework Integration Points
- **Text Generation Engine**: Core language generation capabilities
- **Reasoning Module**: Advanced logical and ethical reasoning
- **Knowledge Base**: Factual information and relationship understanding
- **Multi-Modal Bridge**: Text component for vision-language tasks

## Monitoring and Validation

### Training Monitoring

#### Web Dashboard (Recommended)
```bash
# Start Flask-based monitoring dashboard
python3 training_monitor.py &

# Access at: http://localhost:5001
# Features:
# - Real-time training progress (epochs, steps, loss)
# - GPU utilization and memory usage charts
# - System resource monitoring (CPU, memory, disk)
# - Live training logs with auto-refresh every 30 seconds
# - Visual progress bars and metrics
```

#### Command Line Monitoring
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Training log monitoring
tail -f logs/h200_training_*.log

# Model-specific logs
tail -f logs/simone_enhanced_training.log
```

### Model Validation
The repository includes comprehensive validation (`validate_models.py`):
- **Model Loading**: Successful instantiation from checkpoints
- **Generation Testing**: Text output quality and coherence
- **File Integrity**: All expected model files present
- **Size Verification**: Appropriate model sizes and parameter counts
- **Performance Metrics**: Speed and memory usage validation

## Troubleshooting

### Common Issues
```bash
# CUDA Out of Memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Reduce batch sizes in training configs

# Training Interrupted - Resume from checkpoint
cd "SIM-ONE Training"
python3 enhanced_train.py \
    --resume_from best_model \
    --data_dir ../mvlm_training_dataset_complete/train \
    --validation_dir ../mvlm_training_dataset_complete/val

# Import/Path Issues
# Ensure Python path includes repository root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Slow Training - Enable optimizations
export TORCH_COMPILE=1
export OMP_NUM_THREADS=8
```

### Performance Optimization
- **Memory Issues**: Reduce batch size, enable gradient accumulation
- **Speed Issues**: Enable mixed precision, model compilation
- **Stability Issues**: Check learning rates, gradient scaling
- **Quality Issues**: Verify training data, loss function weights

## Development and Contribution

### For AI Agents and Assistants
- Read `agents.md` for comprehensive development guidelines
- Read `claude.md` for Claude AI specific instructions
- Understand the dual-model architecture and SIM-ONE Framework integration
- Focus enhancements on Enhanced SIM-ONE for advanced capabilities

### Code Standards
- Maintain clean separation between MVLM-GPT2 and Enhanced SIM-ONE
- Use modern PyTorch practices and H200 optimizations
- Follow existing code patterns and import structures
- Ensure backward compatibility with SIM-ONE Framework

### Testing
- Always run `validate_models.py` after training
- Test both individual and sequential training pipelines
- Verify model integration with SIM-ONE Framework components
- Performance benchmark against expected specifications

## License and Usage

These models are developed as core components of **The SIM-ONE Framework** ecosystem. They are designed for:

- Integration within SIM-ONE Framework applications
- Research and development in advanced AI systems
- Production deployment in multi-modal AI platforms
- Educational use in language model development

## Support and Documentation

- **Technical Documentation**: See `agents.md` and `claude.md`
- **Training Guides**: Individual model README files in respective directories
- **Framework Integration**: Refer to SIM-ONE Framework documentation
- **Performance Optimization**: H200-specific optimization guides included

---

## Quick Reference

**Setup**: `./setup_environment.sh`  
**Train All**: `python3 train_all_models.py`  
**Validate**: `python3 validate_models.py`  
**Monitor**: `tail -f logs/h200_training_*.log`  
**Download**: `ls models_for_download/`  

**Total Time**: ~5-7 hours for complete dual-model training  
**Requirements**: H200 GPU, 50GB disk, 80GB VRAM  
**Output**: Two production-ready MVLMs for SIM-ONE Framework integration

---

*These models serve as the foundational text generation components within The SIM-ONE Framework, providing high-quality language capabilities with advanced reasoning and governance mechanisms.*