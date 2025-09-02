"""
MongoDB Tools with Python 3.13.2 compatibility

This module provides MongoDB utilities for connecting to MongoDB Atlas,
creating embeddings, and storing data with vector embeddings.
"""

import os
import numpy as np
import warnings
import sys
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from pymongo import MongoClient
from dotenv import load_dotenv

# Check if we're using Python 3.13+
PY_VERSION = sys.version_info
IS_PY_313_PLUS = PY_VERSION.major == 3 and PY_VERSION.minor >= 13

# Import the ML compatibility layer
try:
    from ..models.ml_compatibility import TENSORFLOW_AVAILABLE
except ImportError:
    # Handle relative import error when running as script
    try:
        from models.ml_compatibility import TENSORFLOW_AVAILABLE
    except ImportError:
        # Default to False if we can't import the compatibility layer
        TENSORFLOW_AVAILABLE = False

# Load environment variables
load_dotenv()

# ----------------------
# MongoDB Utilities
# ----------------------
def get_mongodb_client() -> MongoClient:
    """
    Connect to MongoDB Atlas.
    
    Returns:
        MongoClient: MongoDB client
        
    Raises:
        ValueError: If MongoDB connection string is not set
    """
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    if not connection_string or connection_string == "your_mongodb_connection_string_here":
        raise ValueError("MongoDB connection string not set in .env")
    
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        # Verify connection
        client.admin.command('ping')
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

def get_database(db_name: str = "finance_db") -> Any:
    """
    Get a reference to the database.
    
    Args:
        db_name: Name of the database
        
    Returns:
        Database: MongoDB database
    """
    client = get_mongodb_client()
    return client[db_name]

def get_collection(collection_name: str, db_name: str = "finance_db") -> Any:
    """
    Get a reference to a collection.
    
    Args:
        collection_name: Name of the collection
        db_name: Name of the database
        
    Returns:
        Collection: MongoDB collection
    """
    db = get_database(db_name)
    return db[collection_name]

# ----------------------
# Embeddings Utilities
# ----------------------
def create_text_embeddings(text_data: List[str], embedding_dim: int = 128) -> np.ndarray:
    """
    Create embeddings for a list of text strings.
    
    This function will use TensorFlow if available, otherwise it will
    use a simpler fallback method based on character counts.
    
    Args:
        text_data: List of text strings to embed
        embedding_dim: Dimension of the embeddings
        
    Returns:
        np.ndarray: Array of embeddings
    """
    if TENSORFLOW_AVAILABLE:
        return _create_tensorflow_embeddings(text_data, embedding_dim)
    else:
        return _create_fallback_embeddings(text_data, embedding_dim)

def _create_tensorflow_embeddings(text_data: List[str], embedding_dim: int = 128) -> np.ndarray:
    """
    Create embeddings using TensorFlow.
    
    Args:
        text_data: List of text strings to embed
        embedding_dim: Dimension of the embeddings
        
    Returns:
        np.ndarray: Array of embeddings
    """
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
    from tensorflow.keras.models import Sequential
    
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

def _create_fallback_embeddings(text_data: List[str], embedding_dim: int = 128) -> np.ndarray:
    """
    Create simple embeddings without TensorFlow.
    
    This is a very basic fallback that creates embeddings based on character
    frequencies and text length. It's not suitable for production use but
    allows the code to run without TensorFlow.
    
    Args:
        text_data: List of text strings to embed
        embedding_dim: Dimension of the embeddings
        
    Returns:
        np.ndarray: Array of embeddings
    """
    warnings.warn(
        "Using fallback embedding method. These embeddings are not suitable for "
        "production use. Install TensorFlow for better embeddings."
    )
    
    embeddings = np.zeros((len(text_data), embedding_dim))
    
    for i, text in enumerate(text_data):
        # Use character frequencies as a simple embedding
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Create a seed based on the text
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        
        # Fill the embedding with random values influenced by character counts
        embedding = np.random.normal(0, 0.1, embedding_dim)
        
        # Modify the embedding based on character frequencies
        for j, char in enumerate(sorted(char_counts.keys())):
            if j < embedding_dim:
                embedding[j] += char_counts[char] / len(text)
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        embeddings[i] = embedding
    
    return embeddings

# ----------------------
# Store Embeddings in MongoDB
# ----------------------
def store_embeddings_in_mongodb(collection: Any, documents: List[Dict[str, Any]], 
                               embeddings: np.ndarray, embedding_field: str = "embedding") -> None:
    """
    Store documents with their embeddings in MongoDB.
    Each document will get a field containing its vector.
    
    Args:
        collection: MongoDB collection
        documents: List of documents to store
        embeddings: Array of embeddings
        embedding_field: Name of the field to store embeddings
        
    Raises:
        ValueError: If number of documents and embeddings don't match
    """
    if len(documents) != len(embeddings):
        raise ValueError("Number of documents and embeddings must match")
    
    for doc, emb in zip(documents, embeddings):
        doc[embedding_field] = emb.tolist()
        try:
            collection.insert_one(doc)
        except Exception as e:
            warnings.warn(f"Error inserting document: {str(e)}")

def create_vector_search_index(collection_name: str, db_name: str = "finance_db", 
                              embedding_field: str = "embedding", 
                              embedding_dim: int = 128) -> bool:
    """
    Create a vector search index in MongoDB.
    
    Args:
        collection_name: Name of the collection
        db_name: Name of the database
        embedding_field: Name of the field containing embeddings
        embedding_dim: Dimension of the embeddings
        
    Returns:
        bool: True if index creation was successful, False otherwise
    """
    try:
        db = get_database(db_name)
        
        # Create the index
        index_model = {
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field: {
                        "dimensions": embedding_dim,
                        "similarity": "cosine",
                        "type": "knnVector"
                    }
                }
            }
        }
        
        db.command({
            "createSearchIndex": collection_name,
            "name": f"{collection_name}_index",
            "definition": index_model
        })
        
        return True
    except Exception as e:
        warnings.warn(f"Error creating vector search index: {str(e)}")
        return False

