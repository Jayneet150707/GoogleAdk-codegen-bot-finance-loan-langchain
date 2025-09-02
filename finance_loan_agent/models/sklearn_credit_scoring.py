"""
Scikit-learn Credit Scoring Model for Python 3.13.2

This module provides a scikit-learn implementation of the credit scoring model
that can be used as a fallback when TensorFlow is not available.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class SklearnCreditScoringModel:
    """
    Credit scoring model using scikit-learn, compatible with Python 3.13.2.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        """
        Initialize the credit scoring model.
        
        Args:
            n_estimators: Number of trees in the random forest
            max_depth: Maximum depth of the trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
    
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
              validation_split: float = 0.2, **kwargs) -> Dict[str, float]:
        """
        Train the credit scoring model.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Fraction of data to use for validation
            **kwargs: Additional arguments (ignored, for compatibility with TensorFlow model)
            
        Returns:
            Dict: Training and validation metrics
        """
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Standardize the data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        X_scaled = self.preprocess_data(X)
        preds = self.model.predict(X_scaled)
        
        # Ensure predictions are in the range [0, 1] to match TensorFlow model
        preds = np.clip(preds, 0, 1)
        
        # Reshape to match TensorFlow model output shape
        return preds.reshape(-1, 1)
    
    def get_feature_importance(self, feature_names: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Get the importance of each feature in the model.
        
        Args:
            feature_names: Names of the features
            
        Returns:
            List[Dict]: Feature importance information
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance = []
        for i in indices:
            feature_importance.append({
                'feature': feature_names[i],
                'importance': float(importances[i])
            })
        
        return feature_importance
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            filepath: Path where the model should be saved
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the scaler
        scaler_path = os.path.join(os.path.dirname(filepath), 'sklearn_scaler.pkl')
        try:
            joblib.dump(self.scaler, scaler_path)
        except Exception as e:
            warnings.warn(f"Error saving scaler: {str(e)}")
        
        # Save the model
        try:
            joblib.dump(self.model, filepath)
            return True
        except Exception as e:
            warnings.warn(f"Error saving model: {str(e)}")
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
        scaler_path = os.path.join(os.path.dirname(filepath), 'sklearn_scaler.pkl')
        try:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        except Exception as e:
            warnings.warn(f"Error loading scaler: {str(e)}")
        
        # Try to load the model
        try:
            self.model = joblib.load(filepath)
            return True
        except Exception as e:
            warnings.warn(f"Error loading model: {str(e)}")
            return False

