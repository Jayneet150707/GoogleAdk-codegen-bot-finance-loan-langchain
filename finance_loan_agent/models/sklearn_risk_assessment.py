"""
Scikit-learn Risk Assessment Model for Python 3.13.2

This module provides a scikit-learn implementation of the risk assessment model
that is compatible with Python 3.13.2.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SklearnRiskAssessmentModel:
    """
    Risk assessment model using scikit-learn, compatible with Python 3.13.2.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        """
        Initialize the risk assessment model.
        
        Args:
            n_estimators: Number of trees in the random forest
            max_depth: Maximum depth of the trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
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
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the risk assessment model.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Fraction of data to use for validation
            
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
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=0)
        val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
        
        train_recall = recall_score(y_train, train_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)
        
        train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1': train_f1,
            'val_f1': val_f1
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
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Probability estimates
        """
        X_scaled = self.preprocess_data(X)
        return self.model.predict_proba(X_scaled)
    
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
        scaler_path = os.path.join(os.path.dirname(filepath), 'sklearn_risk_scaler.pkl')
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
        scaler_path = os.path.join(os.path.dirname(filepath), 'sklearn_risk_scaler.pkl')
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

