@echo off
echo Setting up test environment for Enhanced SIM-ONE optimizations...
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv_simone_test
if %errorlevel% neq 0 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv_simone_test\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch CPU version
echo Installing PyTorch CPU...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other required packages
echo Installing additional requirements...
pip install numpy matplotlib scipy scikit-learn
pip install transformers tokenizers datasets evaluate
pip install accelerate tensorboard
pip install pandas tqdm nltk regex
pip install psutil

REM Install the local package
echo Installing local SIM-ONE package...
cd "SIM-ONE Training"
pip install -e .
cd ..

echo.
echo ✅ Test environment setup complete!
echo.
echo To activate the environment, run:
echo   venv_simone_test\Scripts\activate.bat
echo.
echo To run tests:
echo   python "SIM-ONE Training/test_phase1_optimizations.py"
echo.
echo To run benchmarks:
echo   python "SIM-ONE Training/benchmark_phase1_improvements.py"
echo.
pause