@echo off
echo Running Enhanced SIM-ONE Phase 1 Tests...
echo.

REM Check if virtual environment exists
if not exist "venv_simone_test\Scripts\activate.bat" (
    echo Virtual environment not found. Please run setup_test_environment.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv_simone_test\Scripts\activate.bat

echo Environment activated. Running tests...
echo.

REM Run the test suite
echo 🧪 Running Phase 1 optimization tests...
python "SIM-ONE Training/test_phase1_optimizations.py"

if %errorlevel% equ 0 (
    echo.
    echo ✅ All tests passed! Running benchmarks...
    echo.
    
    REM Run benchmarks
    echo 📊 Running performance benchmarks...
    python "SIM-ONE Training/benchmark_phase1_improvements.py"
    
    if %errorlevel% equ 0 (
        echo.
        echo 🎉 Phase 1 optimizations validated successfully!
        echo Check phase1_benchmark_results.json for detailed results.
    ) else (
        echo.
        echo ⚠️ Benchmarks completed with warnings.
    )
) else (
    echo.
    echo ❌ Some tests failed. Please check the output above.
)

echo.
pause