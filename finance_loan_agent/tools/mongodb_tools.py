# import os
# import numpy as np
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from tensorflow.keras.layers import TextVectorization
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# load_dotenv()

# def get_mongodb_client():
#     """Connect to MongoDB Atlas."""
#     connection_string = os.getenv("MONGODB_CONNECTION_STRING")
#     client = MongoClient(connection_string)
#     return client

# def get_database(db_name="finance_db"):
#     """Get a reference to the database."""
#     client = get_mongodb_client()
#     return client[db_name]

# def get_collection(collection_name, db_name="finance_db"):
#     """Get a reference to a collection."""
#     db = get_database(db_name)
#     return db[collection_name]
# def create_text_embeddings(text_data, embedding_dim=128):
#     """Create embeddings for text data using Keras TextVectorization."""
#     import numpy as np
    
#     # Convert list of strings to np.array
#     text_array = np.array(text_data)
    
#     # Create a text vectorization layer
#     vectorize_layer = TextVectorization(
#         max_tokens=10000,
#         output_mode='int',
#         output_sequence_length=250
#     )
#     vectorize_layer.adapt(text_array)
    
#     # Create a simple embedding model
#     model = Sequential([
#         vectorize_layer,
#         Embedding(10000, embedding_dim),
#         GlobalAveragePooling1D(),
#         Dense(embedding_dim, activation='relu')
#     ])
    
#     # Generate embeddings
#     embeddings = model.predict(text_array, verbose=0)
#     return embeddings
# # def create_text_embeddings(text_data, embedding_dim=128):
# #     """Create embeddings for text data."""
# #     # Create a text vectorization layer
# #     vectorize_layer = TextVectorization(
# #         max_tokens=10000,
# #         output_mode='int',
# #         output_sequence_length=250
# #     )
# #     vectorize_layer.adapt(text_data)
    
# #     # Create a simple embedding model
# #     model = Sequential([
# #         vectorize_layer,
# #         Embedding(10000, embedding_dim),
# #         GlobalAveragePooling1D(),
# #         Dense(embedding_dim, activation='relu')
# #     ])
    
# #     # Generate embeddings
# #     embeddings = model.predict(text_data)
# #     return embeddings

# def store_embeddings_in_mongodb(collection, documents, embeddings):
#     """Store documents with their embeddings in MongoDB."""
#     for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
#         doc['embedding'] = embedding.tolist()
#         collection.insert_one(doc)

import os
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential

load_dotenv()

# ----------------------
# MongoDB Utilities
# ----------------------
def get_mongodb_client():
    """Connect to MongoDB Atlas."""
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    if not connection_string or connection_string == "your_mongodb_connection_string_here":
        raise ValueError("MongoDB connection string not set in .env")
    return MongoClient(connection_string)

def get_database(db_name="finance_db"):
    """Get a reference to the database."""
    client = get_mongodb_client()
    return client[db_name]

def get_collection(collection_name, db_name="finance_db"):
    """Get a reference to a collection."""
    db = get_database(db_name)
    return db[collection_name]

# ----------------------
# Keras-based Embeddings
# ----------------------
def create_text_embeddings(text_data, embedding_dim=128):
    """
    Create embeddings for a list of text strings using Keras TextVectorization.
    Returns a NumPy array of embeddings.
    """
    # Convert list of strings to tf.Tensor
    text_tensor = tf.convert_to_tensor(text_data, dtype=tf.string)

    # Text vectorization layer
    vectorize_layer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=250
    )
    vectorize_layer.adapt(text_tensor)

    # Simple embedding model
    model = Sequential([
        vectorize_layer,
        Embedding(input_dim=10000, output_dim=embedding_dim),
        GlobalAveragePooling1D(),
        Dense(embedding_dim, activation='relu')
    ])

    # Generate embeddings
    embeddings = model.predict(text_tensor, verbose=0)
    return embeddings

# ----------------------
# Store Embeddings in MongoDB
# ----------------------
def store_embeddings_in_mongodb(collection, documents, embeddings, embedding_field="embedding"):
    """
    Store documents with their embeddings in MongoDB.
    Each document will get a field containing its vector.
    """
    if len(documents) != len(embeddings):
        raise ValueError("Number of documents and embeddings must match")
    
    for doc, emb in zip(documents, embeddings):
        doc[embedding_field] = emb.tolist()
        collection.insert_one(doc)
