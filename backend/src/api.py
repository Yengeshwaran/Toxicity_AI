import os
import io
import base64
import joblib
import numpy as np
import shap
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Draw
from dotenv import load_dotenv
load_dotenv()  # this loads variables from .env

# Import local modules
from src.features import extract_features
from src.report import generate_pdf_report, REPORTS_DIR

app = FastAPI(title="ToxiGuard AI", description="API for Drug Toxicity Prediction")

# Enable CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini AI Configuration ────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Variables to store loaded models
MODEL_PATH = "model.pkl"
model = None
explainer = None


@app.on_event("startup")
def load_model():
    """Loads the model.pkl and initialize the SHAP explainer on server startup."""
    global model, explainer
    try:
        # Load pre-trained model
        model = joblib.load(MODEL_PATH)

        # Initialize SHAP explainer
        try:
            # We use TreeExplainer as our placeholder model is a RandomForest
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print("SHAP TreeExplainer could not be loaded; using generic explainer logic.", e)
            try:
                # Fallback for other scikit-learn model types
                explainer = shap.Explainer(model)
            except Exception as inner_e:
                print("Could not initialize SHAP explainer natively:", inner_e)
    except Exception as e:
        print(f"Warning: model.pkl not found or failed to load. Error: {e}")
        print("Please run `python src/generate_dummy_model.py` to create a placeholder model.")


# ── Request / Response schemas ─────────────────────────────────

class PredictRequest(BaseModel):
    smiles: str


class PredictResponse(BaseModel):
    toxicity: str
    confidence: float
    top_features: list[str]
    molecule_image: str
    report_url: str
    ai_response: str


# ── Helper functions ───────────────────────────────────────────

def get_base64_image(smiles: str) -> str:
    """Generate a base64 encoded PNG representation of the molecule using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ""
    img = Draw.MolToImage(mol, size=(300, 300))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def get_gemini_explanation(
    smiles: str,
    toxicity: str,
    confidence: float,
    top_features: list[str],
    feature_names: list[str],
    feature_values: list[float],
) -> str:
    """Generate a detailed AI explanation using Google Gemini. Falls back gracefully."""
    # Build a feature summary string
    feature_summary = ", ".join(
        f"{name} = {val:.4f}" for name, val in zip(feature_names, feature_values)
    )

    prompt = (
        f"You are a pharmaceutical toxicology expert AI assistant.\n\n"
        f"A machine learning model has analyzed the molecule with SMILES notation: {smiles}\n\n"
        f"Results:\n"
        f"- Prediction: {toxicity}\n"
        f"- Confidence Score: {confidence:.2%}\n"
        f"- Top Contributing Features (SHAP): {', '.join(top_features)}\n"
        f"- All Extracted Features: {feature_summary}\n\n"
        f"Please provide a detailed explanation covering:\n"
        f"1. What this molecule likely is based on the SMILES structure\n"
        f"2. Why the model predicted it as '{toxicity}' with {confidence:.2%} confidence\n"
        f"3. How the top contributing features (like {', '.join(top_features)}) relate to toxicity\n"
        f"4. What the molecular descriptors (MolWt, LogP, TPSA, NumRotatableBonds) indicate about this compound's safety profile\n"
        f"5. Any general pharmacological concerns\n\n"
        f"Keep the explanation informative, professional, and concise (3-5 paragraphs)."
    )

    # If no API key, return a structured fallback
    if not GEMINI_API_KEY or gemini_client is None:
        return (
            f"AI explanation unavailable (no GEMINI_API_KEY set). "
            f"The molecule '{smiles}' was predicted as {toxicity} with {confidence:.2%} confidence. "
            f"Key features driving the prediction: {', '.join(top_features)}. "
            f"Molecular descriptors: {feature_summary}."
        )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return (
            f"AI explanation could not be generated (API error). "
            f"The molecule '{smiles}' was predicted as {toxicity} with {confidence:.2%} confidence. "
            f"Key features driving the prediction: {', '.join(top_features)}. "
            f"Molecular descriptors: {feature_summary}."
        )


# ── Endpoints ──────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict_toxicity(request: PredictRequest):
    smiles = request.smiles.strip()

    # Input validation
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty")

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")

    # Verify model is available
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please ensure 'model.pkl' exists in the root directory and restart the server.",
        )

    # 1. Feature extraction
    try:
        feature_values, feature_names = extract_features(smiles)
        X = np.array([feature_values])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {e}")

    # 2. Prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # 0 = Non-toxic, 1 = Toxic (based on the dummy model labels)
    toxicity = "Toxic" if prediction == 1 else "Non-toxic"
    confidence = float(max(probabilities))

    # 3. SHAP Explainer output for top features
    top_features = feature_names[:2]  # Fallback
    try:
        if explainer is not None:
            shap_values = explainer.shap_values(X)

            # Scikit-learn RandomForest with SHAP TreeExplainer returns list of arrays
            if isinstance(shap_values, list):
                class_index = int(prediction)
                imp = np.abs(shap_values[class_index][0])
            else:
                if len(shap_values.shape) == 3:
                    imp = np.abs(shap_values[0, :, int(prediction)])
                else:
                    imp = np.abs(shap_values[0])

            top_indices = np.argsort(imp)[::-1][:2]
            top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
    except Exception as e:
        print(f"SHAP explanation process failed during prediction: {e}")

    # 4. Molecule Image generation (base64 for frontend)
    molecule_image = get_base64_image(smiles)

    # 5. Gemini AI detailed explanation
    ai_response = get_gemini_explanation(
        smiles=smiles,
        toxicity=toxicity,
        confidence=confidence,
        top_features=top_features,
        feature_names=feature_names,
        feature_values=feature_values,
    )

    # 6. PDF Report generation
    report_filename = generate_pdf_report(
        smiles=smiles,
        toxicity=toxicity,
        confidence=confidence,
        top_features=top_features,
        feature_values=feature_values,
        feature_names=feature_names,
        ai_explanation=ai_response,
    )
    report_url = f"/report/download/{report_filename}"

    return PredictResponse(
        toxicity=toxicity,
        confidence=round(confidence, 2),
        top_features=top_features,
        molecule_image=molecule_image,
        report_url=report_url,
        ai_response=ai_response,
    )


@app.get("/report/download/{filename}")
async def download_report(filename: str):
    """Download a previously generated PDF report."""
    filepath = os.path.join(REPORTS_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/pdf",
    )
