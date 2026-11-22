"""
Utility functions for MP301 MLOps pipeline.
"""

import hashlib
import os
from typing import Any, Dict
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Load configuration parameters from YAML file using yaml.safe_load.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file not found: {params_path}")
    
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def compute_file_sha256(file_path: str) -> str:
    """
    Compute SHA-256 hexadecimal digest of a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
