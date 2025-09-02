import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class RiskAssessmentModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X):
        """Preprocess the input data."""
        return self.scaler.transform(X)
    
    def train(self, X, y, validation_split=0.2):
        """Train the risk assessment model."""
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Standardize the data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        return {
            'train_score': train_score,
            'validation_score': val_score
        }
    
    def predict(self, X):
        """Make predictions using the trained model."""
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get probability estimates for each class."""
        X_scaled = self.preprocess_data(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """Get the importance of each feature in the model."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance = []
        for i in indices:
            feature_importance.append({
                'feature': feature_names[i],
                'importance': importances[i]
            })
        
        return feature_importance

