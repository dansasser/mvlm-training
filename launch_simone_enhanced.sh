#!/usr/bin/env bash
# Launcher for Enhanced SIM-ONE training on H200 with screen sessions
# Usage: ./launch_simone_enhanced.sh

set -euo pipefail

# Activate environment
if [ -f ./activate_simone.sh ]; then
  source ./activate_simone.sh
else
  echo "activate_simone.sh not found. Run ./setup_environment.sh first." >&2
  exit 1
fi

# Ensure logs directory exists
mkdir -p logs

# Start preflight + training in a detached screen session
SCREEN_NAME="simone-enhanced"
CMD='export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/SIM-ONE Training"; \
python3 enhanced_preflight.py --data_dir ./mvlm_training_dataset_complete/mvlm_comprehensive_dataset || true; \
python3 train_all_models.py'

if command -v screen >/dev/null 2>&1; then
  screen -S "$SCREEN_NAME" -dm bash -lc "$CMD"
  echo "Started training in screen session: $SCREEN_NAME"
else
  echo "screen not found; running training in the foreground..."
  bash -lc "$CMD"
fi

# Start GPU monitor in screen
if command -v screen >/dev/null 2>&1; then
  if screen -list | grep -q "gpuwatch"; then
    echo "gpuwatch screen already running"
  else
    screen -S gpuwatch -dm bash -lc 'if command -v gpustat >/dev/null 2>&1; then gpustat -i 1 | tee -a logs/gpu_watch.log; else nvidia-smi -l 5 | tee -a logs/gpu_watch.log; fi'
    echo "Started GPU monitor in screen session: gpuwatch"
  fi
fi

# Start logs tail in screen
if command -v screen >/dev/null 2>&1; then
  if screen -list | grep -q "simone-logs"; then
    echo "simone-logs screen already running"
  else
    screen -S simone-logs -dm bash -lc 'tail -F logs/h200_training_*.log logs/simone_enhanced_training.log'
    echo "Started log tail in screen session: simone-logs"
  fi
fi

# Show sessions
if command -v screen >/dev/null 2>&1; then
  screen -ls || true
  echo "Attach with: screen -r simone-enhanced (Ctrl+A then D to detach)"
fi
