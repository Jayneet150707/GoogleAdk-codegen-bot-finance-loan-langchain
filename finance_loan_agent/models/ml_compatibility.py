"""
ML Compatibility Layer for Python 3.13.2

This module provides compatibility functions and classes to handle TensorFlow
compatibility issues with Python 3.13.2. It implements fallback mechanisms
using scikit-learn when TensorFlow is not available.
"""

import sys
import warnings
import importlib.util
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple, Callable

# Check if we're using Python 3.13+
PY_VERSION = sys.version_info
IS_PY_313_PLUS = PY_VERSION.major == 3 and PY_VERSION.minor >= 13

# Check for TensorFlow availability
def is_tensorflow_available() -> bool:
    """
    Check if TensorFlow is available in the current environment.
    
    Returns:
        bool: True if TensorFlow is available, False otherwise
    """
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        return False
    
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False

# Global flag for TensorFlow availability
TENSORFLOW_AVAILABLE = is_tensorflow_available()

if not TENSORFLOW_AVAILABLE:
    warnings.warn(
        "TensorFlow is not available in this environment. "
        "Using scikit-learn fallback models for machine learning functionality. "
        "Some advanced features may be limited."
    )

# Import scikit-learn for fallback models
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn is not available. Machine learning functionality will be limited. "
        "Please install scikit-learn for basic model support."
    )

class ModelNotAvailableError(Exception):
    """Exception raised when no ML libraries are available."""
    pass

def get_model_backend() -> str:
    """
    Determine which ML backend to use based on availability.
    
    Returns:
        str: 'tensorflow', 'sklearn', or 'none'
        
    Raises:
        ModelNotAvailableError: If no ML backends are available
    """
    if TENSORFLOW_AVAILABLE:
        return 'tensorflow'
    elif SKLEARN_AVAILABLE:
        return 'sklearn'
    else:
        raise ModelNotAvailableError(
            "No machine learning backends are available. "
            "Please install TensorFlow or scikit-learn."
        )

def load_model_safely(model_path: str, model_type: str = 'tensorflow') -> Any:
    """
    Safely load a model from disk, handling exceptions and fallbacks.
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ('tensorflow' or 'sklearn')
        
    Returns:
        The loaded model or None if loading fails
        
    Raises:
        ValueError: If an unsupported model type is specified
    """
    if model_type not in ['tensorflow', 'sklearn']:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    try:
        if model_type == 'tensorflow':
            if not TENSORFLOW_AVAILABLE:
                warnings.warn("TensorFlow not available, cannot load TensorFlow model")
                return None
                
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
            
        elif model_type == 'sklearn':
            if not SKLEARN_AVAILABLE:
                warnings.warn("scikit-learn not available, cannot load sklearn model")
                return None
                
            import joblib
            return joblib.load(model_path)
    
    except Exception as e:
        warnings.warn(f"Error loading {model_type} model from {model_path}: {str(e)}")
        return None

def save_model_safely(model: Any, model_path: str, model_type: str = 'tensorflow') -> bool:
    """
    Safely save a model to disk, handling exceptions.
    
    Args:
        model: The model to save
        model_path: Path where the model should be saved
        model_type: Type of model ('tensorflow' or 'sklearn')
        
    Returns:
        bool: True if saving was successful, False otherwise
        
    Raises:
        ValueError: If an unsupported model type is specified
    """
    if model_type not in ['tensorflow', 'sklearn']:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    try:
        if model_type == 'tensorflow':
            if not TENSORFLOW_AVAILABLE:
                warnings.warn("TensorFlow not available, cannot save TensorFlow model")
                return False
                
            model.save(model_path)
            return True
            
        elif model_type == 'sklearn':
            if not SKLEARN_AVAILABLE:
                warnings.warn("scikit-learn not available, cannot save sklearn model")
                return False
                
            import joblib
            joblib.dump(model, model_path)
            return True
    
    except Exception as e:
        warnings.warn(f"Error saving {model_type} model to {model_path}: {str(e)}")
        return False

