# Claude AI Assistant Instructions for SIM-ONE Training Repository

This repository contains a complete H200 GPU training pipeline for multi-purpose MVLM development. The training uses high-quality biblical text as low-noise, truth-leaning training data to create general-purpose AI systems. These instructions help Claude understand the codebase structure, training procedures, and optimization strategies.

## Repository Purpose and Context

**Primary Goal**: Train two state-of-the-art multi-purpose language models using H200 GPU hardware
**Training Data**: Biblical corpus chosen for high quality, low noise, and truth-leaning bias
**Deployment**: Automated training on Digital Ocean H200 droplets
**Duration**: 5-7 hours total training time
**Output**: Two production-ready general-purpose AI models

## Two-Model Architecture

### Model 1: MVLM-GPT2 (Root Directory)
```bash
Location: ./mvlm_trainer.py
Architecture: GPT-2 with high-quality training optimization
Output: models/mvlm_gpt2/
Training Time: 2-3 hours
Memory Usage: ~20-30GB GPU
Features: Traditional transformer, proven stability, general-purpose capabilities
```

### Model 2: Enhanced SIM-ONE (SIM-ONE Training/)
```bash
Location: ./SIM-ONE Training/enhanced_train.py
Architecture: Modern transformer with all enhancements
Output: models/simone_enhanced/
Training Time: 3-4 hours  
Memory Usage: ~30-40GB GPU
Features: RoPE, SwiGLU, BPE, RMSNorm, advanced losses, governance mechanisms
```

## Key Technical Improvements in Enhanced SIM-ONE

### Modern Architecture Components
- **RoPE (Rotary Position Embedding)**: Superior position encoding vs learned embeddings
- **SwiGLU Activation**: ~10-15% performance gain over ReLU/GELU
- **RMSNorm**: More stable training than LayerNorm
- **Flash Attention**: Memory-efficient attention computation
- **KV Caching**: Efficient autoregressive generation

### Advanced Tokenization
```python
# High-quality BPETokenizer with 32K vocabulary
# Preserves semantic units as single tokens
# 10-100x speedup over character-level tokenization
# Optimized for high-quality text corpus
quality_seeds = {'important', 'semantic', 'units', 'preserved', 'as', 'tokens', ...}
```

### Advanced Loss Functions
- **Content Alignment Loss**: High-quality content consistency
- **Coherence Loss**: Narrative and logical coherence  
- **Accuracy Loss**: Factual knowledge preservation
- **Comprehensive Loss**: Combined optimization for quality training

### Governance Mechanisms
- **Policy Head**: Decision-making and ethical reasoning
- **Memory Head**: Long-term context and knowledge retention
- **Trace Head**: Reasoning pathway tracking

## H200 GPU Optimizations

### Environment Configuration
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Training Optimizations
- **Mixed Precision (FP16)**: 40-50% memory reduction
- **Gradient Scaling**: Stable mixed precision training
- **Model Compilation**: PyTorch 2.0+ speed improvements
- **Flash Attention**: Memory-efficient attention where available
- **Gradient Accumulation**: Simulate larger batch sizes

## Claude Usage Patterns

### For Repository Analysis
1. **Check Model Structure**: Two distinct models with different architectures
2. **Verify Dataset Paths**: Both use `mvlm_training_dataset_complete/` from root
3. **Import Analysis**: Enhanced SIM-ONE uses modern components only
4. **Configuration Review**: `PrioritaryConfig` for Enhanced model settings

### For Code Enhancement
1. **Focus Area**: Enhanced SIM-ONE in `SIM-ONE Training/` directory
2. **Architecture**: Use modern components (RoPE, SwiGLU, RMSNorm)
3. **Tokenization**: High-quality BPETokenizer preferred over character-level
4. **Loss Functions**: Comprehensive training optimization metrics
5. **Training**: EnhancedPrioritaryTrainer with H200 optimizations

### For Training Support
1. **Setup**: Run `setup_environment.sh` for H200 configuration
2. **Automated Training**: Use `train_all_models.py` for sequential pipeline
3. **Manual Training**: Individual scripts for each model
4. **Validation**: `validate_models.py` after training completion
5. **Download**: Compressed models in `models_for_download/`

## Directory Structure Guide

```
Repository Root/
├── mvlm_trainer.py                    # MVLM-GPT2 trainer
├── train_all_models.py               # Automated sequential training
├── validate_models.py                # Model validation suite
├── setup_environment.sh              # H200 environment setup
├── requirements.txt                  # All Python dependencies
├── mvlm_training_dataset_complete/   # High-quality training corpus
├── models/                           # Training outputs
│   ├── mvlm_gpt2/                   # GPT-2 multi-purpose model
│   └── simone_enhanced/             # Enhanced SIM-ONE model
├── models_for_download/              # Compressed models for download
└── SIM-ONE Training/                 # Enhanced SIM-ONE components
    ├── train.py                     # Simple training entry point
    ├── enhanced_train.py            # Advanced training with CLI
    ├── prioritary_mvlm/             # Enhanced training components
    │   ├── enhanced_trainer.py     # H200-optimized trainer
    │   ├── advanced_tokenizer.py   # BiblicalBPETokenizer
    │   ├── advanced_losses.py      # Biblical loss functions
    │   └── config.py               # Configuration management
    └── simone_transformer/          # Enhanced model architecture
        ├── enhanced_model.py       # EnhancedSIMONEModel
        ├── rope_attention.py       # RoPE + governance attention
        └── modern_layers.py        # SwiGLU, RMSNorm, etc.
```

## Training Commands and Workflows

### Quick Start (H200 Deployment)
```bash
# 1. Clone and setup (5 minutes)
git clone <repo>
cd <repo>
./setup_environment.sh

# 2. Train both models (5-7 hours)
python3 train_all_models.py

# 3. Validate models (5 minutes)
python3 validate_models.py

# 4. Download compressed models
ls models_for_download/
```

### Individual Model Training
```bash
# MVLM-GPT2
python3 mvlm_trainer.py \
    --data_dir mvlm_training_dataset_complete \
    --output_dir models/mvlm_gpt2 \
    --batch_size 16 --num_epochs 3

# Enhanced SIM-ONE
cd "SIM-ONE Training"
python3 enhanced_train.py \
    --data_dir ../mvlm_training_dataset_complete \
    --output_dir ../models/simone_enhanced \
    --vocab_size 32000 --batch_size 12 \
    --gradient_accumulation_steps 4 --num_epochs 3
```

## Claude Best Practices

### Code Analysis Guidelines
1. **Maintain Separation**: Keep MVLM-GPT2 (root) and Enhanced SIM-ONE separate
2. **Import Verification**: Enhanced components only in SIM-ONE Training directory
3. **Dataset Consistency**: Both models share the same high-quality training data
4. **Configuration Understanding**: Different configs for different architectures

### Enhancement Priorities
1. **Architecture Modernization**: Focus on Enhanced SIM-ONE improvements
2. **Content Optimization**: High-quality content alignment and accuracy
3. **H200 Performance**: Memory efficiency and training speed
4. **Tokenization Efficiency**: BPE over character-level processing

### Error Prevention
1. **Import Path Issues**: Use correct relative paths from subdirectories
2. **Memory Management**: Enable H200 optimizations consistently
3. **Dataset Paths**: Verify `../mvlm_training_dataset_complete` from SIM-ONE Training
4. **Component Mixing**: Don't mix legacy and enhanced components

## Performance Monitoring

### Training Metrics
- **Loss Curves**: Both models should show steady decrease
- **GPU Utilization**: Target 80-90% utilization on H200
- **Memory Usage**: Monitor for OOM conditions
- **Training Speed**: Expected tokens/sec performance
- **Content Quality**: High-quality output coherence metrics

### Validation Criteria
- **Model Loading**: Successful instantiation from saved checkpoints
- **Text Generation**: Coherent, high-quality text output
- **File Integrity**: All expected output files present
- **Size Verification**: Reasonable model sizes (2-4GB each)

## Troubleshooting Guide

### Common Issues
```bash
# CUDA Out of Memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Reduce batch sizes in config

# Training Interrupted
python3 enhanced_train.py --resume_from best_model

# Import Errors  
# Check Python path and component locations

# Slow Training
export TORCH_COMPILE=1  # Enable compilation
```

### Performance Issues
- **Memory Optimization**: Reduce batch size, enable gradient accumulation
- **Speed Optimization**: Enable mixed precision, model compilation
- **Stability Issues**: Check gradient scaling, learning rate schedules

## Model Comparison and Evaluation

### MVLM-GPT2 Characteristics
- **Strengths**: Proven architecture, stable training, good content coherence
- **Use Cases**: Traditional applications, baseline comparisons
- **Performance**: ~1000 tokens/sec, reliable text generation

### Enhanced SIM-ONE Characteristics  
- **Strengths**: Modern architecture, superior efficiency, advanced reasoning features
- **Use Cases**: Production deployment, research, advanced applications
- **Performance**: ~600 tokens/sec, higher quality output

### Content Evaluation Criteria
- **Factual Accuracy**: Correct information and references
- **Logical Consistency**: Coherent reasoning and viewpoints
- **Knowledge Retention**: Accurate content recall and contexts
- **Narrative Coherence**: Logical flow in text generation

## Integration and Deployment

### Model Export Format
- **PyTorch State Dict**: Complete model weights and configuration
- **Tokenizer**: Serialized tokenizer (pickle for BPE)
- **Training History**: JSON with loss curves and metrics
- **Configuration**: Model and training hyperparameters

### Production Considerations
- **Inference Optimization**: KV caching for efficient generation
- **Memory Requirements**: Plan for model size in deployment
- **Tokenizer Compatibility**: Ensure tokenizer travels with model
- **Generation Parameters**: Temperature, top-p, repetition penalty tuning

## Advanced Features

### Governance System (Enhanced SIM-ONE)
- **Policy Attention**: Ethical decision making in text generation
- **Memory Integration**: Long-term context maintenance
- **Trace Mechanisms**: Reasoning pathway visibility

### Advanced Loss Functions
- **Multi-objective Optimization**: Balance multiple quality criteria
- **Adaptive Weighting**: Dynamic loss component balancing
- **Evaluation Metrics**: Comprehensive content quality scoring

This guide provides Claude with comprehensive understanding for working effectively with the SIM-ONE training repository, emphasizing the dual-model architecture for multi-purpose MVLM development, modern enhancements, and H200 optimization strategies.