#!/bin/bash
set -e

echo "🚀 Setting up SIM-ONE Training Environment for H200 GPU"
echo "======================================================="

# Check if we're on a GPU instance
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: No NVIDIA GPU detected. This script requires GPU support."
    exit 1
fi

# Display GPU information
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo ""

# Check CUDA version
echo "🔍 CUDA Version:"
nvcc --version || echo "CUDA compiler not found (may still work with runtime)"
echo ""

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    htop \
    tree \
    unzip \
    screen \
    tmux

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (H200 compatible)
echo "🔥 Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA installation
echo "✅ Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available!')
    exit(1)
"

# Install remaining requirements
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Try to install flash-attention for H200 optimization
echo "⚡ Installing Flash Attention for H200 optimization..."
pip3 install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed (optional)"

# Try to install xformers for memory efficiency
echo "⚡ Installing xFormers for memory optimization..."
pip3 install xformers || echo "⚠️  xFormers installation failed (optional)"

# Set up environment variables for optimal H200 performance
echo "⚙️  Setting up H200 optimization environment variables..."
cat << 'EOF' >> ~/.bashrc

# H200 GPU Optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Performance optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

EOF

# Source the new environment
source ~/.bashrc

# Create training directories
echo "📁 Creating training directories..."
mkdir -p models/mvlm_gpt2
mkdir -p models/simone_enhanced
mkdir -p logs
mkdir -p checkpoints

# Set permissions
chmod +x *.py
chmod +x *.sh

# Display final status
echo ""
echo "✅ Environment setup complete!"
echo "🎯 Ready for SIM-ONE training on H200 GPU"
echo ""
echo "📋 Training directories created:"
echo "   - models/mvlm_gpt2/"
echo "   - models/simone_enhanced/"
echo ""
echo "🚀 To start training, run:"
echo "   python3 train_all_models.py"
echo ""
echo "💡 Monitor training with:"
echo "   tail -f logs/training_*.log"
echo "   nvidia-smi -l 1"
echo ""

# Final system check
echo "🔍 Final System Check:"
python3 -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('✅ System ready for training!')
"

echo ""
echo "🎉 Setup complete! Ready to train SIM-ONE models on H200 GPU!"