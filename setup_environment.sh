#!/bin/bash
set -e

echo "🚀 Setting up SIM-ONE Training Environment for H200 GPU"
echo "======================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if package is installed
package_installed() {
    dpkg -l | grep -q "^ii  $1 "
}

# Function to install system package if not present
install_if_missing() {
    if ! package_installed "$1"; then
        echo "📦 Installing $1..."
        sudo apt-get install -y "$1"
    else
        echo "✅ $1 already installed"
    fi
}

# Function to check Python module
python_module_exists() {
    python3 -c "import $1" >/dev/null 2>&1
}

# Update package lists first
echo "📦 Updating package lists..."
sudo apt-get update

# Check and install basic system requirements
echo "🔍 Checking system dependencies..."

# Check if running as root (not recommended for this script)
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Warning: Running as root. Consider running as regular user with sudo access."
fi

# Check for essential tools
if ! command_exists sudo; then
    echo "❌ Error: sudo not available. This script requires sudo access."
    exit 1
fi

if ! command_exists apt-get; then
    echo "❌ Error: apt-get not available. This script requires Ubuntu/Debian."
    exit 1
fi

# Check for Python 3
if ! command_exists python3; then
    echo "📦 Installing Python 3..."
    sudo apt-get install -y python3
else
    python_version=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python 3 found: $python_version"
fi

# Check for pip
if ! command_exists pip3; then
    echo "📦 Installing pip3..."
    sudo apt-get install -y python3-pip
else
    pip_version=$(pip3 --version | cut -d' ' -f2)
    echo "✅ pip3 found: $pip_version"
fi

# Check for python3-venv
if ! python_module_exists venv; then
    echo "📦 Installing python3-venv..."
    sudo apt-get install -y python3-venv
else
    echo "✅ python3-venv available"
fi

# Check if we're on a GPU instance
if ! command_exists nvidia-smi; then
    echo "❌ Error: nvidia-smi not found. Installing NVIDIA drivers..."
    echo "📦 Installing NVIDIA drivers and CUDA toolkit..."
    sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
    echo "⚠️  NVIDIA drivers installed. Please reboot and run this script again."
    exit 1
fi

# Check GPU
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: No NVIDIA GPU detected. This script requires GPU support."
    echo "💡 Make sure NVIDIA drivers are properly installed and the system has been rebooted."
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

# Install essential system packages
echo "📦 Installing essential system packages..."
essential_packages=(
    "build-essential"
    "python3-dev" 
    "git"
    "wget"
    "curl"
    "htop"
    "tree"
    "unzip"
    "screen"
    "tmux"
    "software-properties-common"
    "ca-certificates"
    "gnupg"
    "lsb-release"
)

for package in "${essential_packages[@]}"; do
    install_if_missing "$package"
done

# Create virtual environment
VENV_NAME="sim-one-venv"
echo "🐍 Setting up Python virtual environment: $VENV_NAME"

if [ ! -d "$VENV_NAME" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_NAME"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Verify virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip in virtual environment
echo "📦 Upgrading pip in virtual environment..."
python -m pip install --upgrade pip setuptools wheel

# Check if PyTorch is already installed
if python_module_exists torch; then
    echo "🔍 Checking existing PyTorch installation..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print('✅ PyTorch with CUDA already installed')
else:
    print('⚠️  PyTorch found but CUDA not available')
"
    
    # Check if CUDA is available, reinstall if not
    if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        echo "🔥 Reinstalling PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
    fi
else
    # Install PyTorch with CUDA support (H200 compatible)
    echo "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Verify PyTorch CUDA installation
echo "✅ Verifying PyTorch CUDA installation..."
python -c "
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

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found in current directory"
    echo "💡 Make sure you're running this script from the SIM-ONE training repository root"
    exit 1
fi

# Install remaining requirements
echo "📦 Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Try to install flash-attention for H200 optimization
echo "⚡ Installing Flash Attention for H200 optimization..."
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed (optional)"

# Try to install xformers for memory efficiency
echo "⚡ Installing xFormers for memory optimization..."
pip install xformers || echo "⚠️  xFormers installation failed (optional)"

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

# Create activation script for future use
echo "📝 Creating virtual environment activation script..."
cat << 'EOF' > activate_simone.sh
#!/bin/bash
# Activate SIM-ONE training environment
source sim-one-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
echo "🚀 SIM-ONE environment activated!"
echo "🔥 To train models, run: python train_all_models.py"
EOF
chmod +x activate_simone.sh

# Final comprehensive system check
echo "🔍 Final Comprehensive System Check:"
python -c "
import sys
import torch

print('=' * 50)
print('🐍 PYTHON ENVIRONMENT')
print('=' * 50)
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')
print(f'Virtual environment: {sys.prefix}')

print()
print('=' * 50)
print('🔥 PYTORCH & CUDA')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1e9:.1f} GB')
        print(f'  Compute capability: {props.major}.{props.minor}')
else:
    print('❌ CUDA not available!')
    exit(1)

print()
print('=' * 50)
print('📦 DEPENDENCIES CHECK')
print('=' * 50)

required_modules = ['torch', 'numpy', 'transformers', 'datasets', 'tqdm']
optional_modules = ['flash_attn', 'xformers']

for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module} - installed')
    except ImportError:
        print(f'❌ {module} - missing')

for module in optional_modules:
    try:
        __import__(module)
        print(f'✅ {module} - installed (optional)')
    except ImportError:
        print(f'⚠️  {module} - not installed (optional)')

print()
print('🎯 System ready for SIM-ONE training on H200 GPU!')
"

echo ""
echo "🎉 Setup complete! Ready to train SIM-ONE models on H200 GPU!"
echo ""
echo "💡 Next steps:"
echo "   1. Activate environment: source activate_simone.sh"
echo "   2. Start training: python train_all_models.py"
echo "   3. Monitor training: tail -f logs/training_*.log"