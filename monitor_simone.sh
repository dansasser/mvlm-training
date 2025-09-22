#!/usr/bin/env bash
# Unified monitor for Enhanced SIM-ONE training
# Usage: ./monitor_simone.sh

set -euo pipefail

echo "SIM-ONE Enhanced Training Monitor"
echo "================================="
echo

# Show active screen sessions if available
if command -v screen >/dev/null 2>&1; then
  echo "Screen sessions:"
  screen -ls || true
  echo
fi

# Detect running enhanced training process
PIDS_ENH=$(pgrep -f "enhanced_train.py" || true)
if [ -n "$PIDS_ENH" ]; then
  echo "✅ Enhanced SIM-ONE training is running (PID(s): $PIDS_ENH)"
else
  echo "❌ Enhanced SIM-ONE training is not running"
fi

echo

# GPU status (prefer gpustat)
if command -v gpustat >/dev/null 2>&1; then
  echo "GPU Status (gpustat):"
  gpustat -i 1 | head -n 3 || true
  echo
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU Status (nvidia-smi):"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits || true
  echo
fi

# Recent logs
echo "Recent logs:"
for f in logs/h200_training_*.log logs/simone_enhanced_training.log; do
  if [ -f "$f" ]; then
    echo "---- $f (last 20 lines) ----"
    tail -n 20 "$f" || true
    echo
  fi
done

# Disk usage
echo "Disk usage:"
df -h . | tail -n 1 || true
echo

# Output directory listing
OUT_DIR="models/simone_enhanced"
if [ -d "$OUT_DIR" ]; then
  echo "Outputs in $OUT_DIR (top 30 entries):"
  ls -la "$OUT_DIR" | head -n 30 || true
  echo
fi

echo "Done."
