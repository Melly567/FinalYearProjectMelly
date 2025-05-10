import joblib
import os
import pandas as pd

class PurchasePredictionModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        prediction = self.model.predict(features)[0]
        return bool(prediction)

def get_model_path(base_dir=None):
    if base_dir:
        return os.path.join(base_dir, 'ml_models/best_model.pkl')
    
    possible_locations = [
        'ml_models/best_model.pkl',
        '../ml_models/best_model.pkl',
        '../../ml_models/best_model.pkl',
        os.path.join(os.getcwd(), 'ml_models/best_model.pkl'),
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
            
    return 'ml_models/best_model.pkl'