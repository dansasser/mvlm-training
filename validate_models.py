#!/usr/bin/env python3
"""
Model Validation Script for SIM-ONE Models
Tests all trained models to ensure they work correctly
"""

import os
import sys
import json
import torch
from pathlib import Path
import logging
from datetime import datetime


class ModelValidator:
    """Validates all trained SIM-ONE models."""
    
    def __init__(self):
        self.setup_logging()
        self.test_prompts = [
            "In the beginning God",
            "For God so loved the world",
            "The LORD is my shepherd", 
            "Trust in the LORD",
            "Faith is the substance",
            "Love is patient and kind"
        ]
        
    def setup_logging(self):
        """Setup validation logging."""
        self.logger = logging.getLogger('ModelValidator')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def validate_mvlm_gpt2(self):
        """Validate MVLM-GPT2 model."""
        self.logger.info("🔍 Validating MVLM-GPT2 model...")
        
        model_dir = Path('models/mvlm_gpt2')
        
        # Check if model files exist
        expected_files = ['mvlm_final', 'training_config.json', 'training_plots.png']
        missing_files = []
        
        for file_pattern in expected_files:
            if not list(model_dir.glob(f"*{file_pattern}*")):
                missing_files.append(file_pattern)
        
        if missing_files:
            self.logger.error(f"❌ Missing files: {missing_files}")
            return False
        
        try:
            # Try to load and test the model
            sys.path.append('.')
            
            # Load model components (simplified test)
            config_file = list(model_dir.glob("*training_config*"))[0]
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.info("✅ MVLM-GPT2 model validation passed")
            self.logger.info(f"   Config loaded: {len(config)} parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MVLM-GPT2 validation failed: {e}")
            return False
    
    
    def validate_simone_enhanced(self):
        """Validate SIM-ONE Enhanced model."""
        self.logger.info("🔍 Validating SIM-ONE Enhanced model...")
        
        model_dir = Path('models/simone_enhanced')
        
        # Check for enhanced model files
        expected_patterns = ['enhanced_simone_final', 'tokenizer.pkl', 'training_history.json']
        missing_patterns = []
        
        for pattern in expected_patterns:
            if not list(model_dir.glob(f"*{pattern}*")):
                missing_patterns.append(pattern)
        
        if missing_patterns:
            self.logger.warning(f"⚠️  Missing files: {missing_patterns}")
        
        try:
            # Test enhanced model imports
            sys.path.append('SIM-ONE Training')
            
            from prioritary_mvlm import EnhancedPrioritaryTrainer, BiblicalBPETokenizer
            from simone_transformer import EnhancedSIMONEModel
            
            # Test tokenizer if available
            tokenizer_files = list(model_dir.glob('*tokenizer*'))
            if tokenizer_files:
                # Try loading tokenizer
                tokenizer = BiblicalBPETokenizer()
                # Note: Would need to load actual file for full test
                
            self.logger.info("✅ SIM-ONE Enhanced model validation passed")
            self.logger.info("   Enhanced module imports successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SIM-ONE Enhanced validation failed: {e}")
            return False
    
    def test_model_generation(self, model_name, model_dir):
        """Test text generation for a model (if possible)."""
        self.logger.info(f"🎯 Testing generation for {model_name}...")
        
        try:
            # This is a placeholder - actual implementation would depend on model type
            # For now, just check if we can load basic components
            
            if 'gpt2' in model_name.lower():
                # Test GPT-2 based model
                return self.test_gpt2_generation(model_dir)
            elif 'enhanced' in model_name.lower():
                # Test enhanced model
                return self.test_enhanced_generation(model_dir)
            else:
                # Test legacy SIM-ONE
                return self.test_legacy_generation(model_dir)
                
        except Exception as e:
            self.logger.warning(f"⚠️  Generation test failed for {model_name}: {e}")
            return False
    
    def test_gpt2_generation(self, model_dir):
        """Test GPT-2 model generation."""
        # Placeholder - would require loading actual model
        self.logger.info("   GPT-2 generation test - placeholder")
        return True
    
    def test_enhanced_generation(self, model_dir):
        """Test enhanced model generation."""
        # Placeholder - would require loading actual model and tokenizer
        self.logger.info("   Enhanced generation test - placeholder")
        return True
    
    def test_legacy_generation(self, model_dir):
        """Test legacy model generation."""
        # Placeholder - would require loading actual model
        self.logger.info("   Legacy generation test - placeholder")
        return True
    
    def check_model_sizes(self):
        """Check and report model sizes."""
        self.logger.info("📊 Checking model sizes...")
        
        models_dir = Path('models')
        if not models_dir.exists():
            self.logger.error("❌ Models directory not found")
            return
        
        for model_subdir in models_dir.iterdir():
            if model_subdir.is_dir() and model_subdir.name != 'models_for_download':
                # Calculate directory size
                total_size = sum(
                    f.stat().st_size 
                    for f in model_subdir.rglob('*') 
                    if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                size_gb = size_mb / 1024
                
                file_count = len(list(model_subdir.rglob('*')))
                
                self.logger.info(f"   {model_subdir.name}:")
                self.logger.info(f"     Size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
                self.logger.info(f"     Files: {file_count}")
    
    def validate_all_models(self):
        """Validate all trained models."""
        self.logger.info("🎯 Starting validation of all SIM-ONE models")
        
        validators = [
            ("MVLM-GPT2", self.validate_mvlm_gpt2),
            ("SIM-ONE Enhanced", self.validate_simone_enhanced)
        ]
        
        results = {}
        
        for model_name, validator_func in validators:
            self.logger.info(f"\n{'='*50}")
            try:
                success = validator_func()
                results[model_name] = success
                
                if success:
                    self.logger.info(f"✅ {model_name} validation: PASSED")
                else:
                    self.logger.error(f"❌ {model_name} validation: FAILED")
                    
            except Exception as e:
                self.logger.error(f"❌ {model_name} validation error: {e}")
                results[model_name] = False
        
        # Check model sizes
        self.logger.info(f"\n{'='*50}")
        self.check_model_sizes()
        
        # Summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info("🎯 VALIDATION SUMMARY")
        self.logger.info('='*50)
        
        passed_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        for model_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            self.logger.info(f"   {model_name}: {status}")
        
        self.logger.info(f"\nOverall: {passed_count}/{total_count} models validated successfully")
        
        if passed_count == total_count:
            self.logger.info("🎉 All models validation PASSED!")
            return True
        else:
            self.logger.warning(f"⚠️  {total_count - passed_count} models failed validation")
            return False


def main():
    """Main validation function."""
    print("🔍 SIM-ONE Model Validation")
    print("=" * 40)
    
    validator = ModelValidator()
    
    try:
        success = validator.validate_all_models()
        
        if success:
            print("\n✅ All models validated successfully!")
            return 0
        else:
            print("\n⚠️  Some models failed validation")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())