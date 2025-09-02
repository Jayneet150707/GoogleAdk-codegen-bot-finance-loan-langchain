"""
Test file for ML models with Python 3.13.2 compatibility.
"""

import sys
import unittest
import numpy as np
import pandas as pd
import os
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models
from finance_loan_agent.models.credit_scoring import CreditScoringModel
from finance_loan_agent.models.risk_assessment import RiskAssessmentModel
from finance_loan_agent.models.ml_compatibility import (
    TENSORFLOW_AVAILABLE, 
    SKLEARN_AVAILABLE,
    get_model_backend
)

class TestModels(unittest.TestCase):
    """Test ML models with Python 3.13.2 compatibility."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.X = pd.DataFrame({
            'income': [50000, 75000, 30000, 100000],
            'credit_score': [700, 800, 600, 750],
            'debt_to_income': [0.3, 0.2, 0.5, 0.1],
            'loan_amount': [20000, 30000, 10000, 50000],
            'loan_term': [36, 60, 24, 48]
        })
        self.y = np.array([0, 0, 1, 0])  # 0 = no default, 1 = default
        
        # Print environment info
        print(f"Python version: {sys.version}")
        print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
        print(f"scikit-learn available: {SKLEARN_AVAILABLE}")
        print(f"Model backend: {get_model_backend()}")
    
    def test_credit_scoring_model(self):
        """Test credit scoring model."""
        model = CreditScoringModel()
        
        # Build model
        model.build_model(self.X.shape[1])
        
        # Train model
        history = model.train(self.X, self.y, epochs=2, batch_size=2)
        
        # Make predictions
        preds = model.predict(self.X)
        
        # Check predictions shape
        self.assertEqual(preds.shape[0], self.X.shape[0])
        
        # Check predictions range
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds <= 1))
    
    def test_risk_assessment_model(self):
        """Test risk assessment model."""
        model = RiskAssessmentModel()
        
        # Train model
        history = model.train(self.X, self.y)
        
        # Make predictions
        preds = model.predict(self.X)
        
        # Check predictions shape
        self.assertEqual(preds.shape[0], self.X.shape[0])
        
        # Get probabilities
        probs = model.predict_proba(self.X)
        
        # Check probabilities shape
        self.assertEqual(probs.shape[0], self.X.shape[0])
        
        # Check feature importance
        if hasattr(model.model, 'feature_importances_'):
            importances = model.get_feature_importance(self.X.columns.tolist())
            self.assertEqual(len(importances), self.X.shape[1])

if __name__ == '__main__':
    unittest.main()

