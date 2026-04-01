import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generate_dummy_model():
    # Placeholder model predicting toxicity (0 or 1)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Generate some dummy data representing 4 features: MolWt, LogP, TPSA, NumRotatableBonds
    X_dummy = np.random.rand(100, 4) * 100
    y_dummy = np.random.randint(0, 2, 100) # 0 = Non-toxic, 1 = Toxic
    
    # Train dummy model
    model.fit(X_dummy, y_dummy)
    
    # Save the model
    joblib.dump(model, "model.pkl")
    print("Successfully generated dummy 'model.pkl'.")

if __name__ == "__main__":
    generate_dummy_model()
