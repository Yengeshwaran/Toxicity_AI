import joblib
import pandas as pd
import os

# Correct import
from src.features import smiles_to_features

# Fix model path (works from anywhere)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

# Load model
model = joblib.load(model_path)


def predict(smiles):
    # Convert SMILES to features
    features = smiles_to_features(smiles)

    if features is None:
        return {"error": "Invalid SMILES string"}

    # Convert to DataFrame
    X = pd.DataFrame([features])

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    result = {
        "toxicity": "Toxic" if prediction == 1 else "Non-Toxic",
        "confidence": float(probability)
    }

    return result


# Test block
if __name__ == "__main__":
    print(predict("CCO"))
    print(predict("CCN"))
    print(predict("c1ccccc1"))