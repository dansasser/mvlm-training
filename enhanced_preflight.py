#!/usr/bin/env python3
"""
Enhanced SIM-ONE preflight checker.

Validates environment readiness before launching training:
- GPU availability (CUDA)
- Importability of prioritary_mvlm and simone_transformer
- Dataset presence and .txt/.json pairing
- Disk space availability

Usage:
  python enhanced_preflight.py \
    --data_dir ./mvlm_training_dataset_complete \
    [--train_subdir train] [--val_subdir val] \
    [--min_free_gb 50]
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Tuple


def check_gpu() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"CUDA available: {name} ({mem_gb:.1f} GB)"
    except Exception as e:
        return False, f"Torch/CUDA check failed: {e}"


def check_imports(repo_root: Path, simone_dir: Path) -> Tuple[bool, str]:
    try:
        sys.path.extend([str(repo_root), str(simone_dir)])
        import prioritary_mvlm  # type: ignore # noqa: F401
        import simone_transformer  # type: ignore # noqa: F401
        return True, "Imports OK (prioritary_mvlm, simone_transformer)"
    except Exception as e:
        return False, f"Import failure: {e}"


def count_pairs(base: Path) -> Tuple[int, int, int]:
    """Return (txt_count, json_count, paired_count)."""
    txts = list(base.rglob("*.txt"))
    jsons = {p.with_suffix('.json') for p in base.rglob("*.txt")}
    existing_jsons = set(p for p in base.rglob("*.json"))
    paired = sum(1 for j in jsons if j in existing_jsons)
    return len(txts), len(existing_jsons), paired


def check_dataset(data_dir: Path, train_subdir: str, val_subdir: str) -> Tuple[bool, str]:
    if not data_dir.exists():
        return False, f"Dataset dir not found: {data_dir}"

    # If train/val subdirs exist, report both. Otherwise, report root only.
    train_dir = data_dir / train_subdir
    val_dir = data_dir / val_subdir

    if train_dir.exists() or val_dir.exists():
        parts = []
        if train_dir.exists():
            t_txt, t_json, t_pair = count_pairs(train_dir)
            parts.append(f"train: txt={t_txt}, json={t_json}, pairs={t_pair}")
        else:
            parts.append("train: MISSING")
        if val_dir.exists():
            v_txt, v_json, v_pair = count_pairs(val_dir)
            parts.append(f"val: txt={v_txt}, json={v_json}, pairs={v_pair}")
        else:
            parts.append("val: MISSING (will use holdout split)")
        summary = "; ".join(parts)

        # Consider OK if at least some pairs exist in train
        ok = train_dir.exists()
        if ok:
            _, _, t_pair = count_pairs(train_dir)
            ok = t_pair > 0
        return ok, summary
    else:
        # No subdirs; look at root
        txt, js, pair = count_pairs(data_dir)
        ok = pair > 0
        return ok, f"root: txt={txt}, json={js}, pairs={pair} (no explicit train/val; will rely on validation_split)"


def check_disk_space(path: Path, min_free_gb: int) -> Tuple[bool, str]:
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    ok = free_gb >= min_free_gb
    return ok, f"Free disk: {free_gb:.1f} GB (min required: {min_free_gb} GB)"


def main():
    """
    Run a suite of preflight checks for training readiness and exit with a status code.
    
    Performs GPU, import, dataset, and disk-space checks using configured defaults or
    command-line flags, prints a one-line summary and a line per check, then exits
    with code 0 if all checks pass or 1 if any check fails.
    
    Command-line flags:
      --data_dir       Path to the dataset root (default: ./mvlm_training_dataset_complete/mvlm_comprehensive_dataset)
      --train_subdir   Name of the training subdirectory under the dataset root (default: "train")
      --val_subdir     Name of the validation subdirectory under the dataset root (default: "val")
      --min_free_gb    Minimum required free disk space in GB at the repository root (default: 50)
    """
    ap = argparse.ArgumentParser(description="SIM-ONE Enhanced preflight checker")
    ap.add_argument("--data_dir", type=str, default="./mvlm_training_dataset_complete/mvlm_comprehensive_dataset")
    ap.add_argument("--train_subdir", type=str, default="train")
    ap.add_argument("--val_subdir", type=str, default="val")
    ap.add_argument("--min_free_gb", type=int, default=50)
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    simone_dir = (repo_root / "SIM-ONE Training").resolve()
    data_dir = Path(args.data_dir).resolve()

    checks = []

    ok_gpu, msg_gpu = check_gpu()
    checks.append((ok_gpu, f"GPU: {msg_gpu}"))

    ok_imp, msg_imp = check_imports(repo_root, simone_dir)
    checks.append((ok_imp, f"Imports: {msg_imp}"))

    ok_ds, msg_ds = check_dataset(data_dir, args.train_subdir, args.val_subdir)
    checks.append((ok_ds, f"Dataset: {msg_ds}"))

    ok_disk, msg_disk = check_disk_space(repo_root, args.min_free_gb)
    checks.append((ok_disk, f"Disk: {msg_disk}"))

    print("SIM-ONE Enhanced Preflight Summary")
    print("==================================")
    all_ok = True
    for ok, msg in checks:
        print(("✅" if ok else "❌"), msg)
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
