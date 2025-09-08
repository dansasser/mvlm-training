#!/bin/bash

# Digital Ocean GPU Instance Setup Script for MVLM Training
# This script sets up everything needed to train the MVLM on a fresh Ubuntu GPU instance

set -e  # Exit on any error

echo "=========================================="
echo "MVLM Digital Ocean Setup Script"
echo "Setting up GPU instance for MVLM training"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user with sudo privileges."
   exit 1
fi

# Update system
print_header "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
print_header "Installing essential packages..."
sudo apt install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    unzip \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python 3.11 and pip
print_header "Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev

# Create symbolic links for python3 and pip3
sudo ln -sf /usr/bin/python3.11 /usr/bin/python3
sudo ln -sf /usr/bin/pip3.11 /usr/bin/pip3

# Verify Python installation
python3 --version
pip3 --version

# Install NVIDIA drivers and CUDA (if not already installed)
print_header "Checking NVIDIA GPU and drivers..."

if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA drivers already installed:"
    nvidia-smi
else
    print_warning "NVIDIA drivers not found. Installing..."
    
    # Install NVIDIA drivers
    sudo apt install -y nvidia-driver-535
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-2
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    print_warning "NVIDIA drivers installed. Please reboot the system and run this script again."
    print_warning "After reboot, run: sudo reboot"
    exit 0
fi

# Create project directory
print_header "Creating project directory..."
PROJECT_DIR="$HOME/mvlm_training"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

print_status "Project directory created: $PROJECT_DIR"

# Create Python virtual environment
print_header "Creating Python virtual environment..."
python3 -m venv mvlm_env
source mvlm_env/bin/activate

print_status "Virtual environment created and activated"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
print_header "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other required packages
print_header "Installing required Python packages..."
pip install \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    matplotlib \
    seaborn \
    numpy \
    pandas \
    tqdm \
    wandb \
    tensorboard \
    scikit-learn \
    requests \
    beautifulsoup4 \
    nltk \
    textstat

# Verify PyTorch CUDA installation
print_header "Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('WARNING: CUDA not available!')
"

# Create directory structure
print_header "Creating directory structure..."
mkdir -p data
mkdir -p models
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

# Create requirements.txt
print_header "Creating requirements.txt..."
cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.0
accelerate>=0.20.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
wandb>=0.15.0
tensorboard>=2.13.0
scikit-learn>=1.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
nltk>=3.8.0
textstat>=0.7.0
EOF

# Create environment activation script
print_header "Creating environment activation script..."
cat > activate_mvlm.sh << 'EOF'
#!/bin/bash
# MVLM Environment Activation Script

cd ~/mvlm_training
source mvlm_env/bin/activate

echo "MVLM training environment activated!"
echo "Project directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU Memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")')"
else
    echo "WARNING: CUDA not available!"
fi

echo ""
echo "To start training, run:"
echo "python mvlm_trainer.py --data_dir data/mvlm_comprehensive_dataset --gradient_accumulation_steps 1"
echo ""
EOF

chmod +x activate_mvlm.sh

# Create training script wrapper
print_header "Creating training script wrapper..."
cat > train_mvlm.sh << 'EOF'
#!/bin/bash
# MVLM Training Script Wrapper

set -e

# Activate environment
source mvlm_env/bin/activate

# Check for data directory
if [ ! -d "data/mvlm_comprehensive_dataset" ]; then
    echo "ERROR: Training data not found!"
    echo "Please upload the training dataset to: data/mvlm_comprehensive_dataset"
    echo "You can download it from the provided link and extract it here."
    exit 1
fi

# Check GPU availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available!"
    echo "Please ensure you're using a GPU instance and NVIDIA drivers are installed."
    exit 1
fi

# Start training with optimal settings for Digital Ocean GPU
echo "Starting MVLM training..."
echo "Training data: data/mvlm_comprehensive_dataset"
echo "Output directory: outputs/"
echo ""

python mvlm_trainer.py \
    --data_dir data/mvlm_comprehensive_dataset \
    --output_dir outputs \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --gradient_accumulation_steps 1

echo ""
echo "Training completed!"
echo "Trained model saved to: outputs/mvlm_final"
echo "Training logs: mvlm_training.log"
echo "Training plots: outputs/training_plots.png"
EOF

chmod +x train_mvlm.sh

# Create monitoring script
print_header "Creating monitoring script..."
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# MVLM Training Monitoring Script

echo "MVLM Training Monitor"
echo "===================="

# Check if training is running
if pgrep -f "mvlm_trainer.py" > /dev/null; then
    echo "✅ Training is currently running"
    echo "Process ID: $(pgrep -f mvlm_trainer.py)"
else
    echo "❌ Training is not running"
fi

echo ""

# Show GPU usage
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    echo ""
fi

# Show recent training logs
if [ -f "mvlm_training.log" ]; then
    echo "Recent training logs:"
    echo "--------------------"
    tail -n 10 mvlm_training.log
else
    echo "No training logs found yet."
fi

echo ""

# Show disk usage
echo "Disk usage:"
df -h . | tail -n 1

echo ""

# Show training progress if available
if [ -d "outputs" ]; then
    echo "Training outputs:"
    ls -la outputs/
fi
EOF

chmod +x monitor_training.sh

# Create data download script
print_header "Creating data download script..."
cat > download_data.sh << 'EOF'
#!/bin/bash
# MVLM Training Data Download Script

echo "MVLM Training Data Download"
echo "=========================="

# Create data directory
mkdir -p data

echo "Please upload your training dataset to this server."
echo "You can use scp, rsync, or any file transfer method."
echo ""
echo "Expected structure:"
echo "data/"
echo "└── mvlm_comprehensive_dataset/"
echo "    ├── biblical_teachers/"
echo "    ├── classical_literature/"
echo "    ├── technical_documentation/"
echo "    ├── educational_content/"
echo "    └── philosophical_works/"
echo ""
echo "Example upload command from your local machine:"
echo "scp -r mvlm_training_dataset_complete.tar.gz user@your-server-ip:~/mvlm_training/data/"
echo ""
echo "Then extract with:"
echo "cd data && tar -xzf mvlm_training_dataset_complete.tar.gz"
EOF

chmod +x download_data.sh

# Create cleanup script
print_header "Creating cleanup script..."
cat > cleanup.sh << 'EOF'
#!/bin/bash
# MVLM Cleanup Script

echo "MVLM Cleanup Script"
echo "=================="

read -p "This will remove all training data and models. Are you sure? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up..."
    rm -rf data/
    rm -rf outputs/
    rm -rf checkpoints/
    rm -rf logs/
    rm -f mvlm_training.log
    echo "Cleanup completed."
else
    echo "Cleanup cancelled."
fi
EOF

chmod +x cleanup.sh

# Create system info script
print_header "Creating system info script..."
cat > system_info.sh << 'EOF'
#!/bin/bash
# System Information Script

echo "MVLM Training System Information"
echo "==============================="

echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo ""

echo "System:"
echo "-------"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

echo "CPU:"
echo "----"
lscpu | grep "Model name" | cut -d: -f2 | xargs
echo "Cores: $(nproc)"
echo ""

echo "Memory:"
echo "-------"
free -h | grep "Mem:" | awk '{print "Total: " $2 ", Used: " $3 ", Available: " $7}'
echo ""

echo "Disk:"
echo "-----"
df -h / | tail -n 1 | awk '{print "Total: " $2 ", Used: " $3 ", Available: " $4 ", Usage: " $5}'
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    echo "----"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

echo "Python Environment:"
echo "------------------"
if [ -f "mvlm_env/bin/activate" ]; then
    source mvlm_env/bin/activate
    echo "Python: $(python --version)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
    deactivate
else
    echo "Virtual environment not found"
fi
EOF

chmod +x system_info.sh

# Create final setup summary
print_header "Creating setup summary..."
cat > SETUP_COMPLETE.md << EOF
# MVLM Training Setup Complete!

Your Digital Ocean GPU instance is now ready for MVLM training.

## 📁 Project Structure
\`\`\`
~/mvlm_training/
├── mvlm_env/                 # Python virtual environment
├── data/                     # Training data directory
├── models/                   # Pre-trained models
├── outputs/                  # Training outputs
├── checkpoints/              # Training checkpoints
├── logs/                     # Training logs
├── mvlm_trainer.py          # Main training script
├── activate_mvlm.sh         # Environment activation
├── train_mvlm.sh            # Training wrapper script
├── monitor_training.sh      # Training monitor
├── download_data.sh         # Data download helper
├── cleanup.sh               # Cleanup script
├── system_info.sh           # System information
└── requirements.txt         # Python dependencies
\`\`\`

## 🚀 Quick Start

1. **Activate environment:**
   \`\`\`bash
   ./activate_mvlm.sh
   \`\`\`

2. **Upload training data:**
   - Upload your \`mvlm_training_dataset_complete.tar.gz\` to the \`data/\` directory
   - Extract: \`cd data && tar -xzf mvlm_training_dataset_complete.tar.gz\`

3. **Start training:**
   \`\`\`bash
   ./train_mvlm.sh
   \`\`\`

4. **Monitor progress:**
   \`\`\`bash
   ./monitor_training.sh
   \`\`\`

## 📊 System Information
- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Not detected")
- **Python:** $(python3 --version)
- **PyTorch:** Installed with CUDA support
- **Project Directory:** $PROJECT_DIR

## 💡 Estimated Training Time
- **Small dataset (17M words):** 1-2 hours
- **GPU cost:** \$6-16 total
- **Expected model size:** ~500MB

## 📝 Next Steps
1. Upload your training dataset
2. Run the training script
3. Monitor progress with GPU usage
4. Download the trained model when complete

## 🆘 Support Commands
- \`./system_info.sh\` - Show system information
- \`./monitor_training.sh\` - Monitor training progress
- \`nvidia-smi\` - Check GPU status
- \`htop\` - Check CPU/memory usage

Training setup completed successfully! 🎉
EOF

# Final status
print_header "Setup completed successfully!"
print_status "Project directory: $PROJECT_DIR"
print_status "Virtual environment: mvlm_env"
print_status "GPU status: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not detected')"

echo ""
echo "=========================================="
echo "MVLM SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo "🐍 Python environment: mvlm_env"
echo "🎯 Ready for training!"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_DIR"
echo "2. Upload your training dataset to data/"
echo "3. Run: ./train_mvlm.sh"
echo ""
echo "For help: cat SETUP_COMPLETE.md"
echo ""

# Change to project directory
cd $PROJECT_DIR
pwd

