"""
model saving and loading functionality using joblib
"""

import joblib
from pathlib import Path

def save_model(model, filename: str = "model.joblib"):
    """Save trained model to disk."""
    filepath = Path(filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath.resolve()}")

def load_model(filename: str):
    """Load a previously saved model."""
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    print(f" Model loaded from {filepath.resolve()}")
    return model