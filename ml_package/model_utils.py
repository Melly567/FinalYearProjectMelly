import joblib
import os
import pandas as pd
import numpy as np

class PurchasePredictionModel:
    def __init__(self, model_path=None):
        """
        Initialize the purchase prediction model
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model from disk"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        """
        Make a prediction using the loaded model
        
        Args:
            features: Dictionary or DataFrame containing the features
            
        Returns:
            bool: True for purchase prediction, False otherwise
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # If input is a dictionary, convert to DataFrame
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        # Ensure the model gets properly formatted data
        # Handle the categorical variables as your model expects
        # (Your Django code indicates categorical handling is needed)
        
        prediction = self.model.predict(features)[0]
        return bool(prediction)
    