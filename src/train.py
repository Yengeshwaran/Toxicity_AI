import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

from features import build_dataset

# Load dataset
df = pd.read_csv("../data/tox21_clean.csv")

print("Building features... (this may take time)")
dataset = build_dataset(df)

# Split features and label
X = dataset.drop("label", axis=1)
y = dataset["label"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
print("Training model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "../models/model.pkl")

print("✅ Model saved successfully!")

import shap
import matplotlib.pyplot as plt

print("Generating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("../outputs/shap_values.png")

print("✅ SHAP plot saved!")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

importance.to_csv("../outputs/feature_importance.csv", index=False)

print("✅ Feature importance saved!")