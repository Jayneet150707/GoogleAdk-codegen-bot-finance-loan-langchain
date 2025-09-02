"""
Credit Scoring Model with Python 3.13.2 compatibility

This module provides a credit scoring model that works with both TensorFlow
and scikit-learn, depending on availability. It automatically falls back to
scikit-learn when TensorFlow is not available (e.g., in Python 3.13.2).
"""

import numpy as np
import pandas as pd
import warnings
import os
import joblib
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the ML compatibility layer
try:
    from .ml_compatibility import (
        TENSORFLOW_AVAILABLE, 
        SKLEARN_AVAILABLE,
        get_model_backend,
        load_model_safely,
        save_model_safely,
        ModelNotAvailableError
    )
except ImportError:
    # Handle relative import error when running as script
    from ml_compatibility import (
        TENSORFLOW_AVAILABLE, 
        SKLEARN_AVAILABLE,
        get_model_backend,
        load_model_safely,
        save_model_safely,
        ModelNotAvailableError
    )

# Import TensorFlow if available
if TENSORFLOW_AVAILABLE:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Import scikit-learn for fallback models
if SKLEARN_AVAILABLE:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

class CreditScoringModel:
    """
    Credit scoring model with compatibility for both TensorFlow and scikit-learn.
    Automatically uses the best available backend.
    """
    
    def __init__(self):
        """Initialize the credit scoring model."""
        self.model = None
        self.scaler = StandardScaler()
        self.backend = get_model_backend() if TENSORFLOW_AVAILABLE or SKLEARN_AVAILABLE else 'none'
        
        if self.backend == 'none':
            warnings.warn(
                "No machine learning backends available. "
                "The credit scoring model will provide default predictions only."
            )
    
    def build_model(self, input_dim: int) -> Any:
        """
        Build a model for credit scoring based on the available backend.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            The built model
            
        Raises:
            ModelNotAvailableError: If no ML backends are available
        """
        if self.backend == 'tensorflow':
            return self._build_tensorflow_model(input_dim)
        elif self.backend == 'sklearn':
            return self._build_sklearn_model()
        else:
            raise ModelNotAvailableError("No machine learning backends available")
    
    def _build_tensorflow_model(self, input_dim: int) -> 'tf.keras.Model':
        """
        Build a deep learning model for credit scoring using TensorFlow.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            tf.keras.Model: The built TensorFlow model
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def _build_sklearn_model(self) -> RandomForestRegressor:
        """
        Build a random forest model for credit scoring using scikit-learn.
        
        Returns:
            RandomForestRegressor: The built scikit-learn model
        """
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess the input data.
        
        Args:
            X: Input features as DataFrame or numpy array
            
        Returns:
            np.ndarray: Preprocessed features
        """
        return self.scaler.transform(X)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
              validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32, 
              class_weights: Optional[Dict[int, float]] = None) -> Any:
        """
        Train the credit scoring model.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs (TensorFlow only)
            batch_size: Batch size for training (TensorFlow only)
            class_weights: Class weights for imbalanced data
            
        Returns:
            Training history or evaluation metrics
        """
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Standardize the data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build the model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Train the model based on backend
        if self.backend == 'tensorflow':
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weights
            )
            return history
        
        elif self.backend == 'sklearn':
            self.model.fit(X_train_scaled, y_train)
            train_score = self.model.score(X_train_scaled, y_train)
            val_score = self.model.score(X_val_scaled, y_val)
            
            return {
                'train_score': train_score,
                'validation_score': val_score
            }
        
        else:
            warnings.warn("No machine learning backend available, training skipped")
            return None
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            warnings.warn("Model not trained, returning default predictions")
            return np.full((X.shape[0], 1), 0.5)
        
        X_scaled = self.preprocess_data(X)
        
        if self.backend == 'tensorflow':
            return self.model.predict(X_scaled)
        
        elif self.backend == 'sklearn':
            # Convert sklearn predictions to match TensorFlow format
            preds = self.model.predict(X_scaled)
            return preds.reshape(-1, 1)
        
        else:
            # Default predictions if no model is available
            return np.full((X.shape[0], 1), 0.5)
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            filepath: Path where the model should be saved
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if self.model is None:
            warnings.warn("No model to save")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the scaler
        scaler_path = os.path.join(os.path.dirname(filepath), 'scaler.pkl')
        try:
            joblib.dump(self.scaler, scaler_path)
        except Exception as e:
            warnings.warn(f"Error saving scaler: {str(e)}")
        
        # Save the model using the appropriate backend
        if self.backend == 'tensorflow':
            return save_model_safely(self.model, filepath, 'tensorflow')
        elif self.backend == 'sklearn':
            return save_model_safely(self.model, filepath, 'sklearn')
        else:
            warnings.warn("No machine learning backend available, model not saved")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        # Try to load the scaler
        scaler_path = os.path.join(os.path.dirname(filepath), 'scaler.pkl')
        try:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        except Exception as e:
            warnings.warn(f"Error loading scaler: {str(e)}")
        
        # Try to load the model using the appropriate backend
        if self.backend == 'tensorflow':
            self.model = load_model_safely(filepath, 'tensorflow')
        elif self.backend == 'sklearn':
            self.model = load_model_safely(filepath, 'sklearn')
        else:
            warnings.warn("No machine learning backend available, model not loaded")
            return False
        
        return self.model is not None

