# SIM-ONE MVLM Training Repository

## Overview

This repository contains the complete training pipeline for **SIM-ONE MVLM (Multimodal Vision Language Model)** components that serve as text generators within **The SIM-ONE Framework**. The training leverages high-quality, low-noise biblical text corpus to develop robust, general-purpose language models with superior reasoning capabilities and truth-leaning bias.

**Purpose**: Train production-ready MVLMs for The SIM-ONE Framework ecosystem  
**Architecture**: Dual-model training pipeline optimized for H200 GPU deployment  
**Training Foundation**: Biblical corpus chosen for exceptional text quality and coherence  
**Output**: Two complementary language models serving different roles in SIM-ONE Framework  

## The SIM-ONE Ecosystem

These trained models are integral components of **The SIM-ONE Framework**, serving as the core text generation engines that power:

- **Advanced reasoning systems** with governance mechanisms
- **Multi-modal processing** with vision-language integration  
- **Production AI applications** requiring high-quality text generation
- **Research platforms** for advanced language model development

The models trained here provide the foundational text generation capabilities that The SIM-ONE Framework builds upon for its advanced AI functionalities.

## Two-Model Architecture

### Model 1: MVLM-GPT2 (Baseline Foundation)
**Location**: Root directory (`mvlm_trainer.py`)  
**Architecture**: Enhanced GPT-2 with high-quality training optimization  
**Role in SIM-ONE**: Stable, proven text generation for baseline operations  
**Features**: 
- Traditional transformer architecture with proven stability
- Optimized for consistent, reliable text generation
- Fast inference suitable for real-time applications
- Compatible with existing GPT-2 ecosystem tools

**Training Specifications:**
```
Training Time: 2-3 hours on H200
GPU Memory: ~20-30GB
Performance: ~1000 tokens/sec
Output Size: ~2-3GB model
Architecture: Standard GPT-2 with quality enhancements
```

### Model 2: Enhanced SIM-ONE (Advanced Foundation)
**Location**: `SIM-ONE Training/` directory (`enhanced_train.py`)  
**Architecture**: State-of-the-art transformer with modern enhancements  
**Role in SIM-ONE**: Advanced text generation with reasoning capabilities  
**Features**:
- **RoPE (Rotary Position Embedding)**: Superior position encoding
- **SwiGLU Activation**: ~10-15% performance improvement over standard activations
- **RMSNorm**: Enhanced training stability
- **Advanced BPE Tokenizer**: 32K vocabulary optimized for semantic preservation
- **Governance Mechanisms**: Policy, memory, and trace attention heads
- **Advanced Loss Functions**: Multi-objective optimization for content quality

**Training Specifications:**
```
Training Time: 3-4 hours on H200
GPU Memory: ~30-40GB
Performance: ~600 tokens/sec (higher quality)
Output Size: ~3-4GB model
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

# 2. Setup environment (5 minutes)
chmod +x setup_environment.sh
./setup_environment.sh

# 3. Train both models sequentially (5-7 hours)
python3 train_all_models.py

# 4. Validate trained models (5 minutes)
python3 validate_models.py

# 5. Download compressed models
ls models_for_download/
# Download: mvlm_gpt2_model.tar.gz, simone_enhanced_model.tar.gz
```

### Individual Model Training
```bash
# Train MVLM-GPT2 only
python3 mvlm_trainer.py \
    --data_dir mvlm_training_dataset_complete \
    --output_dir models/mvlm_gpt2 \
    --batch_size 16 \
    --num_epochs 3

# Train Enhanced SIM-ONE only
cd "SIM-ONE Training"
python3 enhanced_train.py \
    --data_dir ../mvlm_training_dataset_complete \
    --output_dir ../models/simone_enhanced \
    --vocab_size 32000 \
    --batch_size 12 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3
```

## Repository Structure

```
SIM-ONE-MVLM-Training/
├── README.md                         # This file
├── agents.md                         # AI agent development guide  
├── claude.md                         # Claude AI assistant instructions
├── requirements.txt                  # Python dependencies
├── setup_environment.sh              # H200 environment setup
│
├── mvlm_trainer.py                   # MVLM-GPT2 training script
├── train_all_models.py               # Automated sequential training
├── validate_models.py                # Model validation suite
│
├── mvlm_training_dataset_complete/   # High-quality training corpus
│   ├── processed_texts/              # Preprocessed training data
│   └── metadata/                     # Dataset information
│
├── models/                           # Training outputs
│   ├── mvlm_gpt2/                   # MVLM-GPT2 model files
│   └── simone_enhanced/             # Enhanced SIM-ONE model files
│
├── models_for_download/              # Compressed models for deployment
│   ├── mvlm_gpt2_model.tar.gz       # Ready-to-deploy MVLM-GPT2
│   ├── simone_enhanced_model.tar.gz # Ready-to-deploy Enhanced SIM-ONE
│   └── training_summary.json        # Training statistics and metadata
│
├── logs/                             # Training logs and monitoring
│   ├── h200_training_*.log          # Main training logs
│   ├── mvlm_gpt2_training.log       # MVLM-GPT2 specific logs
│   └── simone_enhanced_training.log  # Enhanced SIM-ONE specific logs
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

## Performance Specifications

### Expected Training Performance (H200 GPU)
| Model | Training Time | GPU Memory | Tokens/sec | Final Size | 
|-------|---------------|------------|------------|------------|
| MVLM-GPT2 | 2-3 hours | 20-30GB | ~1000 | 2-3GB |
| Enhanced SIM-ONE | 3-4 hours | 30-40GB | ~600 | 3-4GB |
| **Total Pipeline** | **5-7 hours** | **40GB peak** | **Variable** | **5-7GB** |

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
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Training log monitoring
tail -f logs/h200_training_*.log

# Model-specific logs
tail -f logs/mvlm_gpt2_training.log
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
python3 enhanced_train.py --resume_from best_model

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