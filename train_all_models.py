#!/usr/bin/env python3
"""
SIM-ONE Enhanced Training Orchestrator for H200 GPU
Runs the Enhanced SIM-ONE trainer and logs progress.
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import logging
import torch


class H200ModelTrainer:
    """Manages training of the Enhanced SIM-ONE model on H200 GPU."""
    
    def __init__(self):
        self.start_time = datetime.now()
        # Single-model configuration: Enhanced SIM-ONE only
        self.models = [
            {
                'name': 'SIM-ONE-Enhanced',
                'description': 'Enhanced SIM-ONE with RoPE, SwiGLU, BPE, and advanced losses',
                'script': 'SIM-ONE Training/enhanced_train.py',
                'output_dir': 'models/simone_enhanced',
                'args': [
                    '--data_dir', '../mvlm_training_dataset_complete',
                    '--output_dir', '../models/simone_enhanced',
                    '--vocab_size', '32000',
                    '--hidden_dim', '768',
                    '--num_layers', '12',
                    '--batch_size', '12',
                    '--gradient_accumulation_steps', '4',
                    '--learning_rate', '3e-4',
                    '--num_epochs', '3',
                    '--warmup_steps', '2000',
                    '--log_file', '../logs/simone_enhanced_training.log'
                ]
            }
        ]
        
        # Setup logging
        self.setup_logging()
        
        # Create output directories
        for model in self.models:
            Path(model['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Check H200 GPU
        self.check_gpu()
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger('H200Trainer')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = log_dir / f'h200_training_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("H200 SIM-ONE Enhanced Training Started")
        self.logger.info(f"Log file: {log_file}")
    
    def check_gpu(self):
        """Check H200 GPU availability and configuration."""
        if not torch.cuda.is_available():
            self.logger.error("❌ CUDA not available!")
            sys.exit(1)
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        self.logger.info(f"🔥 GPU Detection:")
        self.logger.info(f"   GPU Count: {gpu_count}")
        self.logger.info(f"   GPU Name: {gpu_name}")
        self.logger.info(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        # Set optimal H200 settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        self.logger.info("✅ H200 GPU optimizations applied")
    
    def monitor_gpu_usage(self):
        """Get current GPU usage statistics."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': int(gpu_util),
                    'memory_used_mb': int(mem_used),
                    'memory_total_mb': int(mem_total),
                    'temperature_c': int(temp),
                    'memory_utilization': int(mem_used) / int(mem_total) * 100
                }
        except Exception as e:
            self.logger.warning(f"Could not get GPU stats: {e}")
        
        return None
    
    def train_model(self, model_config):
        """Train a single model."""
        model_name = model_config['name']
        script_path = model_config['script']
        output_dir = model_config['output_dir']
        args = model_config['args']
        
        self.logger.info("="*60)
        self.logger.info(f"🚀 Starting training: {model_name}")
        self.logger.info(f"📁 Output directory: {output_dir}")
        self.logger.info(f"📜 Script: {script_path}")
        self.logger.info("="*60)
        
        # Check initial GPU state
        gpu_stats = self.monitor_gpu_usage()
        if gpu_stats:
            self.logger.info(f"🔍 Initial GPU state: {gpu_stats['gpu_utilization']}% util, "
                           f"{gpu_stats['memory_utilization']:.1f}% memory, {gpu_stats['temperature_c']}°C")
        
        # Prepare command
        cmd = ['python3', script_path] + args
        
        # Set up environment for this training run
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        # Ensure both repo root and SIM-ONE Training are importable during run
        repo_root = str(Path('.').resolve())
        simone_dir = str((Path('SIM-ONE Training')).resolve())
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [env.get('PYTHONPATH', ''), repo_root, simone_dir]))
        
        # Create model-specific log file
        log_file = Path('logs') / f'{model_name.lower().replace("-", "_")}_training.log'
        
        start_time = time.time()
        
        try:
            # Start training process
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    cwd=Path(script_path).parent if '/' in script_path else '.'
                )
                
                # Stream output
                for line in process.stdout:
                    print(line, end='')  # Print to console
                    f.write(line)  # Write to log file
                    f.flush()
                
                # Wait for completion
                process.wait()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
        
        except subprocess.CalledProcessError as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Training failed for {model_name} after {training_time:.2f}s")
            self.logger.error(f"   Command: {' '.join(cmd)}")
            self.logger.error(f"   Return code: {e.returncode}")
            self.logger.error(f"   Check log: {log_file}")
            return False
        
        except KeyboardInterrupt:
            self.logger.warning(f"⚠️  Training interrupted for {model_name}")
            if 'process' in locals():
                process.terminate()
                process.wait()
            return False
        
        training_time = time.time() - start_time
        
        # Check final GPU state
        gpu_stats = self.monitor_gpu_usage()
        if gpu_stats:
            self.logger.info(f"🔍 Final GPU state: {gpu_stats['gpu_utilization']}% util, "
                           f"{gpu_stats['memory_utilization']:.1f}% memory, {gpu_stats['temperature_c']}°C")
        
        # Verify outputs
        output_path = Path(output_dir)
        if output_path.exists() and any(output_path.iterdir()):
            self.logger.info(f"✅ Training completed successfully: {model_name}")
            self.logger.info(f"   Training time: {training_time:.2f}s ({training_time/60:.1f}m)")
            self.logger.info(f"   Output files: {len(list(output_path.rglob('*')))}")
            self.logger.info(f"   Log file: {log_file}")
            return True
        else:
            self.logger.error(f"❌ No output files found for {model_name}")
            return False
    
    def create_model_summary(self):
        """Create summary of the trained model."""
        summary = {
            'training_session': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'gpu_info': {
                    'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown',
                    'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
                }
            },
            'models': []
        }
        
        for model in self.models:
            output_path = Path(model['output_dir'])
            model_info = {
                'name': model['name'],
                'description': model['description'],
                'output_directory': str(output_path),
                'files_created': [],
                'total_size_mb': 0
            }
            
            if output_path.exists():
                # List all files
                for file_path in output_path.rglob('*'):
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / 1e6
                        model_info['files_created'].append({
                            'name': file_path.name,
                            'path': str(file_path.relative_to(output_path)),
                            'size_mb': round(size_mb, 2)
                        })
                        model_info['total_size_mb'] += size_mb
                
                model_info['total_size_mb'] = round(model_info['total_size_mb'], 2)
                model_info['status'] = 'completed'
            else:
                model_info['status'] = 'failed'
            
            summary['models'].append(model_info)
        
        # Save summary
        summary_file = Path('models') / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Training summary saved: {summary_file}")
        return summary
    
    def prepare_for_download(self):
        """Prepare models for download."""
        self.logger.info("📦 Preparing models for download...")
        
        # Create download directory
        download_dir = Path('models_for_download')
        download_dir.mkdir(exist_ok=True)
        
        # Copy and compress each model
        for model in self.models:
            model_name = model['name']
            output_dir = Path(model['output_dir'])
            
            if output_dir.exists() and any(output_dir.iterdir()):
                # Create tar.gz archive
                archive_name = f"{model_name.lower().replace('-', '_')}_model.tar.gz"
                archive_path = download_dir / archive_name
                
                self.logger.info(f"📦 Compressing {model_name} to {archive_name}...")
                
                # Use tar to compress
                subprocess.run([
                    'tar', '-czf', str(archive_path), '-C', str(output_dir.parent), output_dir.name
                ], check=True)
                
                # Get size
                size_mb = archive_path.stat().st_size / 1e6
                self.logger.info(f"   ✅ Created {archive_name} ({size_mb:.1f} MB)")
        
        # Create download instructions
        instructions = Path('models_for_download') / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(instructions, 'w') as f:
            f.write("# SIM-ONE Model Downloads\n\n")
            f.write("## Trained Models\n\n")
            
            for model in self.models:
                model_name = model['name']
                archive_name = f"{model_name.lower().replace('-', '_')}_model.tar.gz"
                f.write(f"### {model_name}\n")
                f.write(f"- **File**: `{archive_name}`\n")
                f.write(f"- **Description**: {model['description']}\n")
                f.write(f"- **Extract with**: `tar -xzf {archive_name}`\n\n")
            
            f.write("## Usage\n\n")
            f.write("1. Download all `.tar.gz` files\n")
            f.write("2. Extract each archive: `tar -xzf model_name.tar.gz`\n")
            f.write("3. Load models using the respective training scripts\n\n")
            f.write("## Training Summary\n\n")
            f.write("See `training_summary.json` for detailed training statistics.\n")
        
        self.logger.info(f"📋 Download instructions: {instructions}")
        
        # List all download files
        download_files = list(download_dir.glob('*'))
        total_size = sum(f.stat().st_size for f in download_files if f.is_file()) / 1e6
        
        self.logger.info("📦 Download package ready:")
        self.logger.info(f"   Directory: {download_dir}")
        self.logger.info(f"   Files: {len(download_files)}")
        self.logger.info(f"   Total size: {total_size:.1f} MB")
        
        return download_dir
    
    def run_sequential_training(self):
        """Run complete sequential training pipeline."""
        self.logger.info("🎯 Starting sequential training of all SIM-ONE models")
        
        successful_models = []
        failed_models = []
        
        for i, model in enumerate(self.models, 1):
            self.logger.info(f"\n📍 Training model {i}/{len(self.models)}: {model['name']}")
            
            # Clear GPU cache before each model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU cache cleared")
            
            # Train model
            success = self.train_model(model)
            
            if success:
                successful_models.append(model['name'])
                self.logger.info(f"✅ {model['name']} completed successfully")
            else:
                failed_models.append(model['name'])
                self.logger.error(f"❌ {model['name']} failed")
                
                # Ask user if they want to continue
                response = input(f"\n⚠️  {model['name']} failed. Continue with remaining models? [y/N]: ")
                if response.lower() != 'y':
                    self.logger.info("🛑 Training stopped by user")
                    break
        
        # Training complete
        total_time = datetime.now() - self.start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 SEQUENTIAL TRAINING COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"⏱️  Total training time: {total_time}")
        self.logger.info(f"✅ Successful models: {len(successful_models)}")
        self.logger.info(f"❌ Failed models: {len(failed_models)}")
        
        if successful_models:
            self.logger.info("✅ Successfully trained:")
            for model in successful_models:
                self.logger.info(f"   - {model}")
        
        if failed_models:
            self.logger.info("❌ Failed models:")
            for model in failed_models:
                self.logger.info(f"   - {model}")
        
        # Create summary
        summary = self.create_model_summary()
        
        # Prepare downloads if any models succeeded
        if successful_models:
            download_dir = self.prepare_for_download()
            self.logger.info(f"\n📦 Models ready for download in: {download_dir}")
        
        self.logger.info("\n🎯 H200 training session complete!")
        
        return len(successful_models), len(failed_models)


def main():
    """Main function."""
    print("🚀 H200 Sequential SIM-ONE Model Training")
    print("=" * 50)
    
    try:
        trainer = H200ModelTrainer()
        success_count, failure_count = trainer.run_sequential_training()
        
        if success_count > 0:
            print(f"\n✅ Training completed! {success_count} models successful, {failure_count} failed")
            print("📦 Check models_for_download/ directory for trained models")
            return 0
        else:
            print(f"\n❌ All models failed to train")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())