"""
Train Model Module with Python 3.13.2 compatibility

This module provides functionality to train the credit scoring model
and create embeddings for loan applications.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from dotenv import load_dotenv

# Check if we're using Python 3.13+
PY_VERSION = sys.version_info
IS_PY_313_PLUS = PY_VERSION.major == 3 and PY_VERSION.minor >= 13

# Import models and tools with proper error handling
try:
    from .models.credit_scoring import CreditScoringModel
    from .models.ml_compatibility import TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE
    from .tools.mongodb_tools import (
        get_collection, 
        create_text_embeddings, 
        store_embeddings_in_mongodb,
        create_vector_search_index
    )
except ImportError:
    # Handle relative import error when running as script
    from models.credit_scoring import CreditScoringModel
    from models.ml_compatibility import TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE
    from tools.mongodb_tools import (
        get_collection, 
        create_text_embeddings, 
        store_embeddings_in_mongodb,
        create_vector_search_index
    )

# Load environment variables
load_dotenv()

def train_credit_scoring_model() -> Tuple[CreditScoringModel, pd.DataFrame]:
    """
    Train the credit scoring model and save it.
    
    Returns:
        Tuple: (Trained model, Training data DataFrame)
    """
    print("Loading data...")
    
    try:
        df = pd.read_csv('finance_loan_agent/data/sample_loan_data.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/sample_loan_data.csv')
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find sample_loan_data.csv in either "
                "finance_loan_agent/data/ or data/ directories."
            )

    # Features & target
    X = df.drop(['loan_id', 'default'], axis=1)
    y = df['default']

    print(f"Data loaded. Shape: {X.shape}, Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved successfully.")

    # Class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weights_dict}")

    # Train the model
    print("Training model...")
    model = CreditScoringModel()
    
    if TENSORFLOW_AVAILABLE:
        print("Using TensorFlow backend for training")
    elif SKLEARN_AVAILABLE:
        print("Using scikit-learn backend for training (TensorFlow not available)")
    else:
        print("WARNING: No ML backends available. Training will be limited.")
    
    try:
        history = model.train(X, y, epochs=50, class_weights=class_weights_dict)
        
        # Save model
        model.save_model('models/credit_scoring_model.h5')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Model training failed. Using default model.")

    return model, df

def create_and_store_embeddings(df: pd.DataFrame, mongodb_enabled: bool = False) -> None:
    """
    Create embeddings and store them in MongoDB.
    
    Args:
        df: DataFrame containing loan application data
        mongodb_enabled: Whether MongoDB storage is enabled
    """
    print("Creating text descriptions for vector search...")

    text_descriptions = []
    for _, row in df.iterrows():
        description = (
            f"Applicant with income {row['income']}, "
            f"credit score {row['credit_score']}, "
            f"debt-to-income ratio {row['debt_to_income']}, "
            f"requesting loan amount {row['loan_amount']} "
            f"for {row['loan_term']} months."
        )
        text_descriptions.append(description)

    print("Creating embeddings...")
    
    if TENSORFLOW_AVAILABLE:
        print("Using TensorFlow for embeddings generation")
    else:
        print("WARNING: TensorFlow not available. Using fallback embedding method.")
        print("These embeddings are not suitable for production use.")
    
    try:
        embeddings = create_text_embeddings(text_descriptions)
        print(f"Created {len(embeddings)} embeddings with dimension {embeddings[0].shape}")
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        print("Embeddings creation failed.")
        return

    if mongodb_enabled:
        try:
            print("Storing embeddings in MongoDB...")
            documents = []
            for i, row in df.iterrows():
                doc = row.to_dict()
                doc['text_description'] = text_descriptions[i]
                documents.append(doc)

            collection = get_collection("loan_applications")
            store_embeddings_in_mongodb(collection, documents, embeddings)
            print("Embeddings stored successfully!")
            
            # Create vector search index
            print("Creating vector search index...")
            if create_vector_search_index("loan_applications"):
                print("Vector search index created successfully!")
            else:
                print("Failed to create vector search index.")

            print("""
            To verify or manually create a vector search index in MongoDB Atlas:
            1. Go to the MongoDB Atlas UI
            2. Navigate to your cluster
            3. Go to the "Search" tab
            4. Create a new index with the following configuration:
            {
              "mappings": {
                "dynamic": true,
                "fields": {
                  "embedding": {
                    "dimensions": 128,
                    "similarity": "cosine",
                    "type": "knnVector"
                  }
                }
              }
            }
            """)
        except Exception as e:
            print(f"Error storing embeddings in MongoDB: {str(e)}")
            print("Skipping MongoDB storage. Make sure your MongoDB connection is configured correctly.")
    else:
        print("MongoDB storage skipped. Set mongodb_enabled=True to store embeddings.")

if __name__ == "__main__":
    print("Starting model training and embedding creation...")
    print(f"Python version: {PY_VERSION.major}.{PY_VERSION.minor}.{PY_VERSION.micro}")
    
    if IS_PY_313_PLUS:
        print("⚠️ Running on Python 3.13+. Some features may use fallback implementations.")
    
    if not TENSORFLOW_AVAILABLE:
        print("⚠️ TensorFlow is not available. Using scikit-learn for model training.")
    
    if not SKLEARN_AVAILABLE:
        print("⚠️ scikit-learn is not available. Model training will be limited.")

    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    mongodb_enabled = mongodb_connection_string is not None and mongodb_connection_string != "your_mongodb_connection_string_here"

    if not mongodb_enabled:
        print("WARNING: MongoDB connection string not found or is default value.")
        print("Vector search functionality will not be available.")

    try:
        # Train + save model & scaler
        model, df = train_credit_scoring_model()

        # Create & optionally store embeddings
        create_and_store_embeddings(df, mongodb_enabled)

        print("Process completed successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print("Process failed to complete successfully.")

