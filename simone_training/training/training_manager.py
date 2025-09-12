"""Training manager for coordinating multiple model training."""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

from ..config import BaseConfig, MVLMConfig, EnhancedConfig


class TrainingManager:
    """Manages training of multiple SIM-ONE models."""
    
    def __init__(self, config: BaseConfig = None):
        """Initialize training manager."""
        self.config = config or BaseConfig()
        self.logger = logging.getLogger(__name__)
        self.models_trained = []
        
    def train_mvlm(self, config: MVLMConfig = None) -> bool:
        """Train MVLM-GPT2 model using existing trainer."""
        config = config or MVLMConfig()
        
        # Call existing mvlm_trainer.py
        root_path = Path(__file__).parent.parent.parent
        trainer_script = root_path / "mvlm_trainer.py"
        
        if not trainer_script.exists():
            self.logger.error(f"MVLM trainer script not found: {trainer_script}")
            return False
            
        # Prepare arguments
        args = [
            sys.executable, str(trainer_script),
            "--data_dir", str(config.data_dir),
            "--output_dir", str(config.output_dir / "mvlm_gpt2"),
            "--batch_size", str(config.batch_size),
            "--learning_rate", str(config.learning_rate),
            "--num_epochs", str(config.num_epochs),
            "--max_length", str(config.max_length)
        ]
        
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            self.models_trained.append("MVLM-GPT2")
            self.logger.info("MVLM-GPT2 training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"MVLM-GPT2 training failed: {e}")
            return False
    
    def train_enhanced(self, config: EnhancedConfig = None) -> bool:
        """Train Enhanced SIM-ONE model using existing trainer."""
        config = config or EnhancedConfig()
        
        # Call existing enhanced_train.py
        root_path = Path(__file__).parent.parent.parent
        trainer_script = root_path / "SIM-ONE Training" / "enhanced_train.py"
        
        if not trainer_script.exists():
            self.logger.error(f"Enhanced trainer script not found: {trainer_script}")
            return False
            
        # Prepare arguments
        args = [
            sys.executable, str(trainer_script),
            "--data_dir", str(Path("..") / config.data_dir),  # Relative from SIM-ONE Training dir
            "--output_dir", str(Path("..") / config.output_dir / "simone_enhanced"),
            "--vocab_size", str(config.vocab_size),
            "--hidden_dim", str(config.hidden_dim),
            "--num_layers", str(config.num_layers),
            "--batch_size", str(config.batch_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--learning_rate", str(config.learning_rate),
            "--num_epochs", str(config.num_epochs),
            "--warmup_steps", str(config.warmup_steps)
        ]
        
        # Change to SIM-ONE Training directory
        original_cwd = Path.cwd()
        training_dir = root_path / "SIM-ONE Training"
        
        try:
            import os
            os.chdir(training_dir)
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            self.models_trained.append("Enhanced-SIM-ONE")
            self.logger.info("Enhanced SIM-ONE training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Enhanced SIM-ONE training failed: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def train_all(self, mvlm_config: MVLMConfig = None, enhanced_config: EnhancedConfig = None) -> Dict[str, bool]:
        """Train all models sequentially."""
        results = {}
        
        self.logger.info("Starting sequential model training")
        
        # Train MVLM-GPT2
        self.logger.info("Training MVLM-GPT2...")
        results["MVLM-GPT2"] = self.train_mvlm(mvlm_config)
        
        # Train Enhanced SIM-ONE
        self.logger.info("Training Enhanced SIM-ONE...")
        results["Enhanced-SIM-ONE"] = self.train_enhanced(enhanced_config)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        self.logger.info(f"Training complete: {success_count}/{total_count} models successful")
        
        return results