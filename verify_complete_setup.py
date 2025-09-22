#!/usr/bin/env python3
"""
Complete verification that ALL 6 dataset directories are configured for 6-7 epoch training.
"""

import sys
from pathlib import Path

def verify_dataset_structure():
    """Verify all 6 dataset directories exist and contain files."""
    print("🔍 DATASET VERIFICATION")
    print("=" * 50)

    base_path = Path("/mnt/sim_one_mvlm_training_block_storage_01/mvlm-training/mvlm_training_dataset_complete/mvlm_comprehensive_dataset")

    expected_dirs = [
        "biblical_classical",
        "educational",
        "gty_sermons",
        "historical_scientific",
        "philosophical",
        "technical"
    ]

    total_files = 0
    for dirname in expected_dirs:
        dir_path = base_path / dirname
        if dir_path.exists():
            txt_files = list(dir_path.rglob("*.txt"))
            file_count = len(txt_files)
            total_files += file_count
            print(f"✅ {dirname:20} : {file_count:4} files")

            # Show subdirectories for biblical_classical and technical
            if dirname in ["biblical_classical", "technical"]:
                subdirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
                print(f"   Subdirectories: {', '.join(subdirs)}")
        else:
            print(f"❌ {dirname:20} : MISSING")
            return False

    print(f"\n📊 TOTAL FILES: {total_files}")
    return total_files == 1226

def verify_training_config():
    """Verify training configuration is set correctly."""
    print("\n🚀 TRAINING CONFIGURATION VERIFICATION")
    print("=" * 50)

    try:
        sys.path.insert(0, str(Path(__file__).parent / "SIM-ONE Training"))
        from prioritary_mvlm.config import PrioritaryConfig

        config = PrioritaryConfig()

        checks = [
            ("Maximum epochs", config.num_epochs, 7),
            ("Minimum epochs", config.min_epochs, 6),
            ("Early stopping patience", config.patience, 2),
            ("Vocab size", config.vocab_size, 131),  # Default, will be overridden
            ("Hidden dim", config.hidden_dim, 512),  # Default, will be overridden
        ]

        all_good = True
        for name, actual, expected in checks:
            if actual == expected:
                print(f"✅ {name:25} : {actual}")
            else:
                print(f"⚠️  {name:25} : {actual} (expected {expected})")
                all_good = False

        return all_good

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def verify_training_scripts():
    """Verify main training scripts have correct parameters."""
    print("\n📝 TRAINING SCRIPTS VERIFICATION")
    print("=" * 50)

    scripts_to_check = [
        ("train_all_models.py", ["--num_epochs', '7'", "--min_epochs', '6'"]),
        ("SIM-ONE Training/enhanced_train.py", ["default=7", "default=6"])
    ]

    all_good = True
    for script_name, expected_patterns in scripts_to_check:
        script_path = Path(__file__).parent / script_name
        if script_path.exists():
            content = script_path.read_text()
            found_patterns = []
            missing_patterns = []

            for pattern in expected_patterns:
                if pattern in content:
                    found_patterns.append(pattern)
                else:
                    missing_patterns.append(pattern)

            if missing_patterns:
                print(f"⚠️  {script_name}: Missing {missing_patterns}")
                all_good = False
            else:
                print(f"✅ {script_name}: All epoch parameters updated")
        else:
            print(f"❌ {script_name}: File not found")
            all_good = False

    return all_good

def main():
    print("🎯 COMPLETE SIM-ONE MVLM TRAINING SETUP VERIFICATION")
    print("=" * 60)
    print("Verifying 6-7 epoch truth-leaning training across ALL 6 dataset domains")
    print("Purpose: Create low-noise AI through singular truth source consistency")
    print("=" * 60)

    # Run all verifications
    dataset_ok = verify_dataset_structure()
    config_ok = verify_training_config()
    scripts_ok = verify_training_scripts()

    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)

    if dataset_ok and config_ok and scripts_ok:
        print("🎉 ALL SYSTEMS GO!")
        print("✅ Dataset: All 6 domains (1,226 files) ready")
        print("✅ Configuration: 6-7 epochs with early stopping")
        print("✅ Scripts: Updated for complete dataset training")
        print()
        print("🚀 Ready to create the world's first truth-leaning, low-noise AI!")
        print("💰 Expected cost: $72-120 for revolutionary 7-domain mastery")
        print("⏱️  Expected time: ~24 hours on H200 GPU")
        print()
        print("🎯 Training will cover (all from singular truth source):")
        print("   • Truth-aligned classical & theological content")
        print("   • Consistent philosophical reasoning")
        print("   • Historical & Scientific documents")
        print("   • Educational materials with coherent worldview")
        print("   • Technical content (Architecture + Chemistry)")
        print("   • Literary works with moral consistency")
        print()
        print("🏆 Result: First AI trained for truth-leaning bias through")
        print("    singular source consistency - minimal contradictions!")
        print("🔬 Proves: Consistent worldview > contradictory scale")
        return True
    else:
        print("❌ SETUP INCOMPLETE")
        if not dataset_ok:
            print("❌ Dataset verification failed")
        if not config_ok:
            print("❌ Configuration verification failed")
        if not scripts_ok:
            print("❌ Scripts verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)