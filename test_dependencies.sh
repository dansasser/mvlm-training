#!/bin/bash

echo "🔍 Testing SIM-ONE Dependency Check (No Installation)"
echo "===================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if package is installed
package_installed() {
    dpkg -l | grep -q "^ii  $1 " 2>/dev/null
}

# Function to check Python module
python_module_exists() {
    python3 -c "import $1" >/dev/null 2>&1
}

echo "🔍 System Requirements Check:"
echo "=============================="

# Check essential tools
echo -n "sudo: "
if command_exists sudo; then
    echo "✅ Available"
else
    echo "❌ Missing"
fi

echo -n "apt-get: "
if command_exists apt-get; then
    echo "✅ Available"
else
    echo "❌ Missing"
fi

# Check for Python 3
echo -n "python3: "
if command_exists python3; then
    python_version=$(python3 --version | cut -d' ' -f2)
    echo "✅ Available ($python_version)"
else
    echo "❌ Missing"
fi

# Check for pip
echo -n "pip3: "
if command_exists pip3; then
    pip_version=$(pip3 --version | cut -d' ' -f2)
    echo "✅ Available ($pip_version)"
else
    echo "❌ Missing"
fi

# Check for python3-venv
echo -n "python3-venv: "
if python_module_exists venv; then
    echo "✅ Available"
else
    echo "❌ Missing"
fi

# Check for GPU tools
echo -n "nvidia-smi: "
if command_exists nvidia-smi; then
    echo "✅ Available"
    if nvidia-smi > /dev/null 2>&1; then
        echo "  🔥 GPU detected"
    else
        echo "  ⚠️  Command exists but no GPU detected"
    fi
else
    echo "❌ Missing"
fi

echo ""
echo "🐍 Python Environment Check:"
echo "============================="

# Check if virtual environment exists
if [ -d "sim-one-venv" ]; then
    echo "✅ Virtual environment 'sim-one-venv' exists"
    
    # Check if we can activate it
    if [ -f "sim-one-venv/bin/activate" ]; then
        echo "✅ Virtual environment is properly configured"
        
        # Test activation (in subshell to avoid affecting current session)
        (
            source sim-one-venv/bin/activate
            if [[ "$VIRTUAL_ENV" != "" ]]; then
                echo "✅ Virtual environment can be activated"
                echo "  📍 Location: $VIRTUAL_ENV"
                
                # Check for PyTorch in venv
                if python -c "import torch" >/dev/null 2>&1; then
                    pytorch_version=$(python -c "import torch; print(torch.__version__)")
                    cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
                    echo "✅ PyTorch installed in venv ($pytorch_version)"
                    echo "  🔥 CUDA available: $cuda_available"
                else
                    echo "❌ PyTorch not installed in virtual environment"
                fi
            else
                echo "❌ Failed to activate virtual environment"
            fi
        )
    else
        echo "❌ Virtual environment missing activation script"
    fi
else
    echo "❌ Virtual environment 'sim-one-venv' does not exist"
fi

echo ""
echo "📦 Required Dependencies Check:"
echo "==============================="

# Check for requirements.txt
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
    echo "📋 Required packages:"
    grep -E "^[a-zA-Z]" requirements.txt | head -10 | while read package; do
        echo "  - $package"
    done
else
    echo "❌ requirements.txt not found"
fi

echo ""
echo "🏗️  Essential System Packages Check:"
echo "===================================="

essential_packages=(
    "build-essential"
    "python3-dev" 
    "git"
    "wget"
    "curl"
)

for package in "${essential_packages[@]}"; do
    echo -n "$package: "
    if package_installed "$package"; then
        echo "✅ Installed"
    else
        echo "❌ Missing"
    fi
done

echo ""
echo "🎯 Readiness Summary:"
echo "===================="

ready_count=0
total_checks=5

# Check Python
if command_exists python3; then
    echo "✅ Python 3 ready"
    ((ready_count++))
else
    echo "❌ Python 3 missing"
fi

# Check pip
if command_exists pip3; then
    echo "✅ pip3 ready"
    ((ready_count++))
else
    echo "❌ pip3 missing"
fi

# Check venv capability
if python_module_exists venv; then
    echo "✅ Virtual environment capability ready"
    ((ready_count++))
else
    echo "❌ Virtual environment capability missing"
fi

# Check GPU
if command_exists nvidia-smi && nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU ready"
    ((ready_count++))
else
    echo "❌ GPU not ready"
fi

# Check requirements.txt
if [ -f "requirements.txt" ]; then
    echo "✅ Requirements file ready"
    ((ready_count++))
else
    echo "❌ Requirements file missing"
fi

echo ""
echo "📊 Readiness Score: $ready_count/$total_checks"

if [ $ready_count -eq $total_checks ]; then
    echo "🎉 System is ready for SIM-ONE setup!"
    echo "💡 Run ./setup_environment.sh to complete installation"
elif [ $ready_count -ge 3 ]; then
    echo "⚠️  System partially ready - some dependencies missing"
    echo "💡 Run ./setup_environment.sh to install missing components"
else
    echo "❌ System needs significant setup"
    echo "💡 Ensure you're on a GPU-enabled system and run ./setup_environment.sh"
fi