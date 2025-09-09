# H200 GPU Training Setup Guide 🚀

## Quick Start on Digital Ocean H200 Droplet

### 1. Initial Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# Make setup script executable
chmod +x setup_environment.sh

# Run automated setup
./setup_environment.sh
```

### 2. Start Training All Models
```bash
# Train all three models sequentially
python3 train_all_models.py
```

### 3. Monitor Training
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Follow training logs
tail -f logs/h200_training_*.log

# Monitor individual model logs
tail -f logs/mvlm_gpt2_training.log
tail -f logs/sim_one_legacy_training.log  
tail -f logs/sim_one_enhanced_training.log
```

### 4. Validate Models
```bash
# Validate all trained models
python3 validate_models.py
```

### 5. Download Models
```bash
# Models are automatically prepared in models_for_download/
ls -la models_for_download/

# Download files:
# - mvlm_gpt2_model.tar.gz
# - sim_one_legacy_model.tar.gz  
# - sim_one_enhanced_model.tar.gz
# - training_summary.json
# - DOWNLOAD_INSTRUCTIONS.md
```

## Training Sequence

The automated trainer will run these models in sequence:

### 1. MVLM-GPT2 (Biblical GPT-2)
- **Script**: `mvlm_trainer.py`
- **Output**: `models/mvlm_gpt2/`
- **Features**: Biblical training with GPT-2 architecture
- **Time**: ~2-3 hours

### 2. SIM-ONE Enhanced (Modern Architecture)
- **Script**: `SIM-ONE Training/enhanced_train.py`
- **Output**: `models/simone_enhanced/`
- **Features**: RoPE, SwiGLU, BPE tokenizer, advanced losses, governance
- **Time**: ~3-4 hours

**Total Training Time**: ~5-7 hours

## H200 Optimizations Applied

- **CUDA Memory Management**: `max_split_size_mb:512,expandable_segments:True`
- **Mixed Precision**: FP16 training for memory efficiency
- **Flash Attention**: Memory-optimized attention (where available)
- **Model Compilation**: PyTorch 2.0+ compilation for speed
- **Gradient Scaling**: Automatic loss scaling for stability
- **Optimal Threading**: OMP/MKL thread configuration

## System Requirements

- **GPU**: NVIDIA H200 (or compatible CUDA GPU)
- **Memory**: 80GB+ GPU memory recommended
- **Storage**: 50GB+ free space for models and logs
- **Python**: 3.8+
- **CUDA**: 11.8+ or 12.1+

## Directory Structure After Training

```
├── models/
│   ├── mvlm_gpt2/                    # GPT-2 biblical model
│   ├── simone_enhanced/              # Enhanced SIM-ONE
│   └── training_summary.json         # Training statistics
├── models_for_download/              # Compressed models
│   ├── mvlm_gpt2_model.tar.gz
│   ├── simone_enhanced_model.tar.gz
│   └── DOWNLOAD_INSTRUCTIONS.md
├── logs/                             # Training logs
│   ├── h200_training_*.log           # Main training log
│   ├── mvlm_gpt2_training.log        # GPT-2 model log
│   └── simone_enhanced_training.log  # Enhanced model log
└── checkpoints/                      # Training checkpoints
```

## Manual Training (Alternative)

If you prefer to train models individually:

### MVLM-GPT2
```bash
python3 mvlm_trainer.py \
    --data_dir mvlm_training_dataset_complete \
    --output_dir models/mvlm_gpt2 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 3
```

### SIM-ONE Enhanced
```bash
cd "SIM-ONE Training"
python3 enhanced_train.py \
    --data_dir ../mvlm_training_dataset_complete \
    --output_dir ../models/mvlm_simone_enhanced \
    --vocab_size 32000 \
    --batch_size 12 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --num_epochs 3
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch sizes
export BATCH_SIZE=4  # Instead of 8-16

# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### Training Interrupted
```bash
# Resume enhanced training from checkpoint
cd "SIM-ONE Training"
python3 enhanced_train.py --resume_from best_model
```

### Package Installation Issues
```bash
# Install with pip instead of conda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash attention manually
pip3 install flash-attn --no-build-isolation
```

### Slow Training
```bash
# Enable model compilation (PyTorch 2.0+)
export TORCH_COMPILE=1

# Use optimal thread counts
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## Monitoring Tools

### GPU Usage
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Detailed GPU stats
gpustat -i 1

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Training Progress
```bash
# Live training logs
tail -f logs/h200_training_*.log

# GPU utilization graph
watch -n 5 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits'

# Training curves (if available)
tensorboard --logdir logs/
```

### System Resources
```bash
# CPU and memory usage
htop

# Disk usage
df -h
du -sh models/

# Network usage (if applicable)  
iotop
```

## Performance Expectations

### H200 (80GB) Expected Performance:
- **MVLM-GPT2**: ~1000 tokens/sec, 2-3 hours total
- **SIM-ONE Enhanced**: ~600 tokens/sec, 3-4 hours total

### Memory Usage:
- **MVLM-GPT2**: ~20-30GB GPU memory
- **SIM-ONE Enhanced**: ~30-40GB GPU memory

## Success Indicators

✅ **Training Successful When**:
- All logs show decreasing loss values
- Models save without errors
- Validation tests pass
- Generated text is coherent and biblical
- Model files are created in output directories

❌ **Training Failed When**:
- CUDA out of memory errors
- Loss becomes NaN or doesn't decrease
- Models fail to save
- Generated text is incoherent
- Missing output files

## Post-Training Cleanup

```bash
# Remove large temporary files
rm -rf checkpoints/
rm -rf __pycache__/
rm -rf logs/*.log

# Keep only compressed models
ls -la models_for_download/

# Optional: Remove uncompressed models to save space  
# rm -rf models/mvlm_*
```

## Final Checklist Before Droplet Destruction

- [ ] Both models trained successfully
- [ ] Models validated with `validate_models.py`
- [ ] Compressed models created in `models_for_download/`
- [ ] Downloaded all `.tar.gz` files
- [ ] Downloaded `training_summary.json`
- [ ] Downloaded log files (optional)
- [ ] Verified model file integrity

---

🎉 **Ready to train two state-of-the-art biblical language models on H200 GPU!**