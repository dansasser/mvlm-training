"""Environment validation utilities."""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def validate_environment() -> Tuple[bool, Dict[str, any]]:
    """Validate training environment setup.
    
    Returns:
        Tuple of (is_valid, status_dict)
    """
    status = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": [],
        "issues": []
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        status["cuda_version"] = torch.version.cuda
        status["gpu_count"] = torch.cuda.device_count()
        
        for i in range(status["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = gpu_props.total_memory / 1e9  # Convert to GB
            
            status["gpu_names"].append(gpu_name)
            status["gpu_memory"].append(gpu_memory)
    else:
        status["issues"].append("CUDA not available - training will be slow")
    
    # Check Python version
    if sys.version_info < (3, 8):
        status["issues"].append("Python 3.8+ recommended for optimal performance")
    
    # Check PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 0):
        status["issues"].append("PyTorch 2.0+ recommended for compilation features")
    
    # Check for common issues
    try:
        # Test basic tensor operations
        x = torch.randn(2, 2)
        if torch.cuda.is_available():
            x = x.cuda()
            y = x @ x
    except Exception as e:
        status["issues"].append(f"Basic tensor operations failed: {e}")
    
    is_valid = len(status["issues"]) == 0
    return is_valid, status


def check_dataset_availability(data_dir: str) -> Tuple[bool, Dict[str, any]]:
    """Check if training dataset is available.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Tuple of (is_available, info_dict)
    """
    data_path = Path(data_dir)
    info = {
        "data_dir": str(data_path),
        "exists": data_path.exists(),
        "file_count": 0,
        "total_size_mb": 0,
        "issues": []
    }
    
    if not data_path.exists():
        info["issues"].append(f"Dataset directory not found: {data_path}")
        return False, info
    
    # Count files and calculate size
    try:
        files = list(data_path.rglob("*.txt"))  # Assuming text files
        info["file_count"] = len(files)
        
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        info["total_size_mb"] = total_size / (1024 * 1024)
        
        if info["file_count"] == 0:
            info["issues"].append("No .txt files found in dataset directory")
        
        if info["total_size_mb"] < 10:  # Less than 10MB
            info["issues"].append("Dataset appears very small (< 10MB)")
            
    except Exception as e:
        info["issues"].append(f"Error analyzing dataset: {e}")
    
    is_available = len(info["issues"]) == 0
    return is_available, info


def validate_training_setup(data_dir: str = "mvlm_training_dataset_complete") -> Dict[str, any]:
    """Comprehensive training setup validation.
    
    Args:
        data_dir: Path to training dataset
        
    Returns:
        Complete validation report
    """
    report = {
        "timestamp": None,
        "environment": {},
        "dataset": {},
        "overall_status": "unknown",
        "recommendations": []
    }
    
    from datetime import datetime
    report["timestamp"] = datetime.now().isoformat()
    
    # Validate environment
    env_valid, env_status = validate_environment()
    report["environment"] = env_status
    
    # Validate dataset
    data_valid, data_status = check_dataset_availability(data_dir)
    report["dataset"] = data_status
    
    # Overall status
    if env_valid and data_valid:
        report["overall_status"] = "ready"
    elif env_valid or data_valid:
        report["overall_status"] = "partial"
    else:
        report["overall_status"] = "not_ready"
    
    # Generate recommendations
    if not torch.cuda.is_available():
        report["recommendations"].append("Install CUDA-compatible PyTorch for GPU acceleration")
    
    if report["environment"]["gpu_count"] == 0:
        report["recommendations"].append("GPU recommended for efficient training")
    
    if not data_valid:
        report["recommendations"].append("Ensure training dataset is properly downloaded and accessible")
    
    if len(report["recommendations"]) == 0:
        report["recommendations"].append("Environment appears ready for training")
    
    return report