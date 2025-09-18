#!/bin/bash
echo "Running Complete Enhanced SIM-ONE Test Suite..."
echo

# Check if virtual environment exists
if [ ! -f "venv_simone_test/bin/activate" ]; then
    echo "Virtual environment not found. Please run setup_test_environment.sh first."
    exit 1
fi

# Activate virtual environment
source venv_simone_test/bin/activate

echo "Environment activated. Running complete test suite..."
echo

# Run Phase 1 tests
echo "=========================================="
echo "🧪 PHASE 1: Critical Fixes & Performance"
echo "=========================================="
python "SIM-ONE Training/test_phase1_optimizations.py"

if [ $? -ne 0 ]; then
    echo
    echo "❌ Phase 1 tests failed. Stopping here."
    exit 1
fi

echo
echo "✅ Phase 1 tests passed! Running Phase 1 benchmarks..."
echo

python "SIM-ONE Training/benchmark_phase1_improvements.py"

if [ $? -ne 0 ]; then
    echo
    echo "⚠️ Phase 1 benchmarks had issues, but continuing..."
fi

# Run Phase 2 tests
echo
echo "=========================================="
echo "🧪 PHASE 2: Architectural Optimizations"
echo "=========================================="
python "SIM-ONE Training/test_phase2_optimizations.py"

if [ $? -ne 0 ]; then
    echo
    echo "❌ Phase 2 tests failed. Check the output above."
    exit 1
fi

echo
echo "✅ Phase 2 tests passed! Running Phase 2 benchmarks..."
echo

python "SIM-ONE Training/benchmark_phase2_improvements.py"

if [ $? -ne 0 ]; then
    echo
    echo "⚠️ Phase 2 benchmarks had issues, but tests passed."
fi

echo
echo "=========================================="
echo "🎉 COMPLETE TEST SUITE RESULTS"
echo "=========================================="
echo "✅ Phase 1: Critical fixes and performance optimizations working"
echo "✅ Phase 2: Architectural optimizations functional"
echo
echo "📊 Benchmark results saved to:"
echo "  - phase1_benchmark_results.json"
echo "  - phase2_benchmark_results.json"
echo
echo "🚀 Enhanced SIM-ONE is ready for deployment!"
echo "Check the JSON files for detailed performance metrics."
echo