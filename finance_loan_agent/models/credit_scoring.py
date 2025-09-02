import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class CreditScoringModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim):
        """Build a deep learning model for credit scoring."""
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
    
    def preprocess_data(self, X):
        """Preprocess the input data."""
        return self.scaler.transform(X)
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32, class_weights=None):
        """Train the credit scoring model."""
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Standardize the data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build the model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights
        )
        
        return history
    
    def predict(self, X):
        """Make predictions using the trained model."""
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save the model to disk."""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a saved model from disk."""
        self.model = tf.keras.models.load_model(filepath)

