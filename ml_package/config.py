import os

# Default locations
DEFAULT_MODEL_PATH = 'ml_models/best_model.pkl'

def get_model_path(base_dir=None):
    """
    Get the absolute path to the model file
    
    Args:
        base_dir: Optional base directory (default is current directory)
        
    Returns:
        str: Full path to model file
    """
    if base_dir:
        return os.path.join(base_dir, DEFAULT_MODEL_PATH)
    
    # Try to locate the model in common locations
    possible_locations = [
        # Current directory
        os.path.join(os.getcwd(), DEFAULT_MODEL_PATH),
        # Parent directory
        os.path.join(os.path.dirname(os.getcwd()), DEFAULT_MODEL_PATH),
        # Django project directory (relative to this file)
        os.path.join(os.path.dirname(os.path.dirname(__file__)), DEFAULT_MODEL_PATH),
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
            
    # If not found, return the default path and let the caller handle missing file
    return DEFAULT_MODEL_PATH