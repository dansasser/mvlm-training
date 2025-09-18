#!/bin/bash
echo "Setting up test environment for Enhanced SIM-ONE optimizations..."
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_simone_test
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_simone_test/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch CPU version
echo "Installing PyTorch CPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
echo "Installing additional requirements..."
pip install numpy matplotlib scipy scikit-learn
pip install transformers tokenizers datasets evaluate
pip install accelerate tensorboard
pip install pandas tqdm nltk regex
pip install psutil

# Install the local package
echo "Installing local SIM-ONE package..."
cd "SIM-ONE Training"
pip install -e .
cd ..

echo
echo "✅ Test environment setup complete!"
echo
echo "To activate the environment, run:"
echo "  source venv_simone_test/bin/activate"
echo
echo "To run tests:"
echo "  python \"SIM-ONE Training/test_phase1_optimizations.py\""
echo
echo "To run benchmarks:"
echo "  python \"SIM-ONE Training/benchmark_phase1_improvements.py\""
echo