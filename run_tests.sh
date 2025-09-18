#!/bin/bash
echo "Running Enhanced SIM-ONE Phase 1 Tests..."
echo

# Check if virtual environment exists
if [ ! -f "venv_simone_test/bin/activate" ]; then
    echo "Virtual environment not found. Please run setup_test_environment.sh first."
    exit 1
fi

# Activate virtual environment
source venv_simone_test/bin/activate

echo "Environment activated. Running tests..."
echo

# Run the test suite
echo "🧪 Running Phase 1 optimization tests..."
python "SIM-ONE Training/test_phase1_optimizations.py"

if [ $? -eq 0 ]; then
    echo
    echo "✅ All tests passed! Running benchmarks..."
    echo
    
    # Run benchmarks
    echo "📊 Running performance benchmarks..."
    python "SIM-ONE Training/benchmark_phase1_improvements.py"
    
    if [ $? -eq 0 ]; then
        echo
        echo "🎉 Phase 1 optimizations validated successfully!"
        echo "Check phase1_benchmark_results.json for detailed results."
    else
        echo
        echo "⚠️ Benchmarks completed with warnings."
    fi
else
    echo
    echo "❌ Some tests failed. Please check the output above."
fi

echo