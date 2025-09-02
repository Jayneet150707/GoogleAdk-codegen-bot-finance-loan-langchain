# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.utils import class_weight
# from models.credit_scoring import CreditScoringModel
# from tools.mongodb_tools import get_collection, create_text_embeddings, store_embeddings_in_mongodb
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def train_credit_scoring_model():
#     """Train the credit scoring model and save it."""
#     print("Loading data...")
#     # Load and prepare data
#     df = pd.read_csv('finance_loan_agent/data/sample_loan_data.csv')

#     # Prepare features and target
#     X = df.drop(['loan_id', 'default'], axis=1)
#     y = df['default']

#     print(f"Data loaded. Shape: {X.shape}, Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}")

#     # Calculate class weights to handle imbalanced data
#     class_weights = class_weight.compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y),
#         y=y
#     )
#     class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
#     print(f"Class weights: {class_weights_dict}")

#     # Train the model
#     print("Training model...")
#     model = CreditScoringModel()
#     history = model.train(X, y, epochs=50, class_weights=class_weights_dict)

#     # Create directory if it doesn't exist
#     os.makedirs('models', exist_ok=True)
    
#     # Save the model
#     print("Saving model...")
#     model.save_model('models/credit_scoring_model.h5')
#     print("Model saved successfully!")
    
#     return model, df

# def create_and_store_embeddings(df, mongodb_enabled=False):
#     """Create embeddings and store them in MongoDB."""
#     print("Creating text descriptions for vector search...")
#     # Create text descriptions for vector search
#     text_descriptions = []
#     for _, row in df.iterrows():
#         description = (
#             f"Applicant with income {row['income']}, "
#             f"credit score {row['credit_score']}, "
#             f"debt-to-income ratio {row['debt_to_income']}, "
#             f"requesting loan amount {row['loan_amount']} "
#             f"for {row['loan_term']} months."
#         )
#         text_descriptions.append(description)

#     print("Creating embeddings...")
#     # Create embeddings
#     embeddings = create_text_embeddings(text_descriptions)
#     print(f"Created {len(embeddings)} embeddings with dimension {embeddings[0].shape}")

#     if mongodb_enabled:
#         try:
#             print("Storing embeddings in MongoDB...")
#             # Prepare documents for MongoDB
#             documents = []
#             for i, row in df.iterrows():
#                 doc = row.to_dict()
#                 doc['text_description'] = text_descriptions[i]
#                 documents.append(doc)

#             # Store in MongoDB
#             collection = get_collection("loan_applications")
#             store_embeddings_in_mongodb(collection, documents, embeddings)
#             print("Embeddings stored successfully!")
            
#             print("""
#             To create a vector search index in MongoDB Atlas:
#             1. Go to the MongoDB Atlas UI
#             2. Navigate to your cluster
#             3. Go to the "Search" tab
#             4. Create a new index with the following configuration:
#             {
#               "mappings": {
#                 "dynamic": true,
#                 "fields": {
#                   "embedding": {
#                     "dimensions": 128,
#                     "similarity": "cosine",
#                     "type": "knnVector"
#                   }
#                 }
#               }
#             }
#             """)
#         except Exception as e:
#             print(f"Error storing embeddings in MongoDB: {e}")
#             print("Skipping MongoDB storage. Make sure your MongoDB connection is configured correctly.")
#     else:
#         print("MongoDB storage skipped. Set mongodb_enabled=True to store embeddings.")

# if __name__ == "__main__":
#     print("Starting model training and embedding creation...")
    
#     # Check if MongoDB is configured
#     mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
#     mongodb_enabled = mongodb_connection_string is not None and mongodb_connection_string != "your_mongodb_connection_string_here"
    
#     if not mongodb_enabled:
#         print("WARNING: MongoDB connection string not found or is default value.")
#         print("Vector search functionality will not be available.")
    
#     # Train model and get data
#     model, df = train_credit_scoring_model()
    
#     # Create and store embeddings
#     create_and_store_embeddings(df, mongodb_enabled)
    
#     print("Process completed successfully!")

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from models.credit_scoring import CreditScoringModel
from tools.mongodb_tools import get_collection, create_text_embeddings, store_embeddings_in_mongodb
from dotenv import load_dotenv

load_dotenv()

def train_credit_scoring_model():
    """Train the credit scoring model and save it."""
    print("Loading data...")
    df = pd.read_csv('finance_loan_agent/data/sample_loan_data.csv')

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
    history = model.train(X_scaled, y, epochs=50, class_weights=class_weights_dict)

    # Save model
    model.save_model('models/credit_scoring_model.h5')
    print("Model saved successfully!")

    return model, df

def create_and_store_embeddings(df, mongodb_enabled=False):
    """Create embeddings and store them in MongoDB."""
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
    embeddings = create_text_embeddings(text_descriptions)
    print(f"Created {len(embeddings)} embeddings with dimension {embeddings[0].shape}")

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

            print("""
            To create a vector search index in MongoDB Atlas:
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
            print(f"Error storing embeddings in MongoDB: {e}")
            print("Skipping MongoDB storage. Make sure your MongoDB connection is configured correctly.")
    else:
        print("MongoDB storage skipped. Set mongodb_enabled=True to store embeddings.")

if __name__ == "__main__":
    print("Starting model training and embedding creation...")

    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    mongodb_enabled = mongodb_connection_string is not None and mongodb_connection_string != "your_mongodb_connection_string_here"

    if not mongodb_enabled:
        print("WARNING: MongoDB connection string not found or is default value.")
        print("Vector search functionality will not be available.")

    # Train + save model & scaler
    model, df = train_credit_scoring_model()

    # Create & optionally store embeddings
    create_and_store_embeddings(df, mongodb_enabled)

    print("Process completed successfully!")
