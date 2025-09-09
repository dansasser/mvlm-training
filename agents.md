# SIM-ONE Training Repository Agents Guide

This repository contains specialized training pipelines for multi-purpose MVLM (Minimum Viable Language Model) development, designed for automated deployment on H200 GPU instances. The training uses high-quality biblical text as a low-noise, truth-leaning foundation for general-purpose AI systems.

## Repository Overview

**Purpose**: Train two distinct multi-purpose language models using biblical corpus for low-noise training
**Architecture**: Two-model pipeline with automated deployment and validation
**Target Hardware**: NVIDIA H200 (80GB VRAM) on Digital Ocean droplets
**Training Data**: Biblical text chosen for high quality, low noise, and truth-leaning bias

## Model Architecture Summary

### 1. MVLM-GPT2 (Root Directory)
- **Script**: `mvlm_trainer.py`
- **Architecture**: Standard GPT-2 with low-noise training optimization
- **Output**: `models/mvlm_gpt2/`
- **Training Time**: 2-3 hours
- **Memory**: ~20-30GB GPU
- **Features**: Traditional transformer, proven architecture, general-purpose capabilities

### 2. Enhanced SIM-ONE (SIM-ONE Training Directory)
- **Script**: `enhanced_train.py` (main) or `train.py` (simplified)
- **Architecture**: Modern transformer with all enhancements
- **Output**: `models/simone_enhanced/`
- **Training Time**: 3-4 hours
- **Memory**: ~30-40GB GPU
- **Features**: RoPE, SwiGLU, BPE tokenizer, RMSNorm, advanced training losses, governance mechanisms

## Key Technical Components

### Enhanced SIM-ONE Architecture
```
SIM-ONE Training/
├── prioritary_mvlm/
│   ├── enhanced_trainer.py      # H200-optimized trainer
│   ├── advanced_tokenizer.py    # High-quality BPETokenizer (32K vocab)
│   ├── advanced_losses.py       # Advanced training loss functions
│   └── config.py               # Configuration
├── simone_transformer/
│   ├── enhanced_model.py       # EnhancedSIMONEModel
│   ├── rope_attention.py       # RoPE + governance
│   └── modern_layers.py        # SwiGLU, RMSNorm, etc.
├── train.py                    # Simple trainer entry point
└── enhanced_train.py           # Advanced trainer with CLI
```

### Modern Improvements Applied
- **RoPE (Rotary Position Embedding)**: Better position encoding than learned embeddings
- **SwiGLU Activation**: ~10-15% performance improvement over ReLU
- **RMSNorm**: More stable than LayerNorm
- **BPE Tokenization**: 10-100x speedup over character-level, preserves semantic units
- **Advanced Loss Functions**: Content alignment, coherence, accuracy optimization
- **Governance Mechanisms**: Policy, memory, and trace attention heads for advanced reasoning
- **H200 Optimizations**: Mixed precision, Flash Attention, model compilation

## Agent Workflow Patterns

### For Code Analysis Tasks
1. **Repository Structure**: Start with understanding the two-model setup
2. **Import Dependencies**: Check imports carefully - Enhanced SIM-ONE uses modern components
3. **Dataset Paths**: Both models use `mvlm_training_dataset_complete/` from root
4. **Configuration**: Enhanced model uses `PrioritaryConfig` with modern defaults

### For Enhancement Tasks
1. **Architecture Focus**: Enhanced SIM-ONE is where modern improvements go
2. **Tokenization**: Use `BiblicalBPETokenizer` (32K vocab) over character-level
3. **Attention**: `EnhancedGovernanceAttention` with RoPE encoding
4. **Feedforward**: `SwiGLU` layers instead of standard MLP
5. **Normalization**: `RMSNorm` preferred over LayerNorm

### For Training Tasks
1. **Automated Pipeline**: Use `train_all_models.py` for sequential training
2. **Individual Training**: Use respective trainer scripts directly
3. **H200 Setup**: Run `setup_environment.sh` first
4. **Validation**: Always run `validate_models.py` after training
5. **Download Prep**: Models auto-compressed to `models_for_download/`

## Common Agent Patterns

### When Working with SIM-ONE Training Directory
```python
# Correct imports for Enhanced SIM-ONE
from simone_transformer import EnhancedSIMONEModel
from prioritary_mvlm import EnhancedPrioritaryTrainer, AdvancedBPETokenizer
from prioritary_mvlm.advanced_losses import ComprehensiveTrainingLoss

# Dataset path from SIM-ONE Training directory
data_dir = "../mvlm_training_dataset_complete"
```

### When Analyzing Model Performance
- **MVLM-GPT2**: Focus on text coherence, traditional language modeling metrics
- **Enhanced SIM-ONE**: Emphasize modern architectural advantages, efficiency gains, advanced reasoning
- **Comparison**: Enhanced model should show superior performance/efficiency for general tasks

### When Optimizing for H200
- **Memory Management**: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- **Mixed Precision**: Always enabled for 40-50% memory savings
- **Flash Attention**: Available where supported
- **Model Compilation**: PyTorch 2.0+ compilation for speed

## File Patterns and Conventions

### Configuration Files
- `PrioritaryConfig`: Modern configuration with sensible defaults
- CLI arguments override config defaults in `enhanced_train.py`
- High-quality training parameters for alignment and coherence

### Model Outputs
- **Checkpoints**: Saved during training for recovery
- **Final Models**: Complete model state with tokenizer
- **Training Plots**: Loss curves and metrics visualization
- **Training History**: JSON with detailed training statistics

### Logging and Monitoring
- **Main Log**: `logs/h200_training_*.log`
- **Individual Logs**: Per-model training logs
- **GPU Monitoring**: Built-in nvidia-smi integration
- **Progress Reports**: Real-time statistics

## H200 Deployment Context

### Setup Sequence
1. Clone repository to H200 droplet
2. Run `setup_environment.sh` (installs all dependencies)
3. Execute `train_all_models.py` (automated pipeline)
4. Run `validate_models.py` (verify models)
5. Download compressed models from `models_for_download/`

### Performance Expectations
- **Total Training**: 5-7 hours for both models
- **MVLM-GPT2**: ~1000 tokens/sec processing speed
- **Enhanced SIM-ONE**: ~600 tokens/sec (more complex architecture)
- **Memory Usage**: Up to 40GB GPU for largest model

## Agent Best Practices

### Code Modifications
1. **Preserve Architecture**: Don't break the two-model structure
2. **Dataset Consistency**: Both models use the same biblical dataset
3. **Import Cleanliness**: Avoid mixing legacy and enhanced components
4. **H200 Optimization**: Maintain performance optimizations

### Analysis Focus Areas
1. **Content Quality**: How well models capture high-quality language patterns
2. **Architectural Efficiency**: Modern vs traditional transformer comparisons
3. **Training Stability**: Loss curves, convergence patterns
4. **Memory Efficiency**: GPU utilization and optimization effectiveness

### Error Handling
1. **CUDA OOM**: Reduce batch sizes, enable memory optimization
2. **Import Errors**: Check component paths, ensure clean separation
3. **Dataset Issues**: Verify `mvlm_training_dataset_complete/` exists
4. **Training Failures**: Check logs, resume from checkpoints

## Integration Points

### External Dependencies
- **PyTorch**: 2.0+ with CUDA 12.1 support
- **Transformers**: HuggingFace transformers library
- **Flash Attention**: Optional but recommended for H200
- **xFormers**: Memory optimization (optional)

### Data Pipeline
- **Input**: High-quality texts in `mvlm_training_dataset_complete/`
- **Preprocessing**: Handled by respective tokenizers
- **Output**: Multi-purpose models ready for general inference

### Validation Pipeline
- **Model Loading**: Test model instantiation and parameter loading
- **Generation Testing**: Basic text generation capabilities
- **Size Reporting**: Model file sizes and parameter counts
- **Integrity Checking**: Verify all expected files present

## Troubleshooting Guide for Agents

### Common Issues
1. **Import Path Confusion**: Enhanced SIM-ONE components are in `SIM-ONE Training/`
2. **Dataset Path Errors**: Use `../mvlm_training_dataset_complete` from subdirectories
3. **Memory Limitations**: H200 optimizations must be enabled
4. **Training Interruptions**: Use checkpoint resume capabilities

### Performance Optimization
1. **Batch Size Tuning**: Adjust based on available memory
2. **Gradient Accumulation**: Simulate larger batches efficiently
3. **Learning Rate Scheduling**: Warmup + cosine decay pattern
4. **Mixed Precision**: Essential for H200 efficiency

This guide provides agents with comprehensive understanding of the repository's structure, training pipelines, and optimization strategies for effective multi-purpose MVLM development using high-quality training data.