# 🧬 ToxiGuard AI — Backend

> AI-powered drug toxicity prediction system built with **FastAPI**, **RDKit**, **SHAP**, and **scikit-learn**.

ToxiGuard AI takes a [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) string as input and returns a toxicity prediction, confidence score, explainable top features, a 2D molecule image, and a downloadable PDF report.

---

## 📁 Project Structure

```
toxiguard-ai/
├── model.pkl                   # Trained ML model (auto-generated or provided)
├── requirements.txt            # Python dependencies
├── README.md
├── reports/                    # Auto-generated PDF reports
└── src/
    ├── api.py                  # FastAPI application — endpoints & server logic
    ├── features.py             # RDKit molecular feature extraction pipeline
    ├── report.py               # PDF report generation module
    └── generate_dummy_model.py # Utility to create a placeholder model
```

---

## ⚙️ Setup & Installation

### Prerequisites

- **Python 3.10+**

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd toxiguard-ai
```

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate a placeholder model (if no real model is available)

```bash
python src/generate_dummy_model.py
```

This creates `model.pkl` in the project root — a basic Random Forest trained on dummy data so the API can run end-to-end.

> **Note:** Replace `model.pkl` with your team's actual trained model when ready. No code changes needed.

---

## ▶️ Running the Server

```bash
uvicorn src.api:app --reload
```

The server starts at **http://127.0.0.1:8000**

- 📖 **Swagger UI (Interactive Docs):** http://127.0.0.1:8000/docs
- 📄 **ReDoc:** http://127.0.0.1:8000/redoc

---

## 🔌 API Endpoints

### `POST /predict`

Predict toxicity for a given molecule.

**Request Body:**

```json
{
  "smiles": "CCO"
}
```

**Response (200 OK):**

```json
{
  "toxicity": "Toxic",
  "confidence": 0.87,
  "top_features": ["LogP", "MolWt"],
  "molecule_image": "<base64-encoded-PNG>",
  "report_url": "/report/download/toxiguard_report_CCO_20260331_183446.pdf"
}
```

| Field            | Type       | Description                                      |
|------------------|------------|--------------------------------------------------|
| `toxicity`       | `string`   | `"Toxic"` or `"Non-toxic"`                       |
| `confidence`     | `float`    | Model confidence score (0.0 – 1.0)               |
| `top_features`   | `string[]` | Top SHAP-identified contributing features         |
| `molecule_image` | `string`   | Base64-encoded PNG of the 2D molecule structure   |
| `report_url`     | `string`   | Relative URL to download the generated PDF report |

**Error Responses:**

| Code | Detail                        |
|------|-------------------------------|
| 400  | SMILES string cannot be empty |
| 400  | Invalid SMILES string         |
| 500  | Model is not loaded           |

---

### `GET /report/download/{filename}`

Download a previously generated PDF toxicity report.

**Example:**

```
GET http://127.0.0.1:8000/report/download/toxiguard_report_CCO_20260331_183446.pdf
```

Returns the PDF file directly as a download.

---

## 🧪 Testing via Swagger UI

1. Start the server with `uvicorn src.api:app --reload`
2. Open http://127.0.0.1:8000/docs
3. Expand **POST /predict** → Click **Try it out**
4. Enter a SMILES string, e.g. `{"smiles": "CCO"}`
5. Click **Execute**
6. Copy the `report_url` from the response
7. Open `http://127.0.0.1:8000` + `report_url` in your browser to download the PDF

---

## 🧠 How It Works

```
SMILES Input
    │
    ▼
┌─────────────────────┐
│  Input Validation    │  ← RDKit validates the SMILES string
│  (api.py)            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Extraction  │  ← Extracts MolWt, LogP, TPSA, NumRotatableBonds
│  (features.py)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Prediction    │  ← Loads model.pkl, returns Toxic/Non-toxic + confidence
│  (api.py)            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  SHAP Explanation    │  ← Identifies top contributing molecular features
│  (api.py)            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Molecule Image      │  ← RDKit renders 2D structure as base64 PNG
│  (api.py)            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  PDF Report          │  ← Generates a professional PDF with all results
│  (report.py)         │
└────────┬────────────┘
         │
         ▼
   JSON Response
```

---

## 📦 Key Dependencies

| Package        | Purpose                                    |
|----------------|--------------------------------------------|
| `fastapi`      | Web framework for building the REST API    |
| `uvicorn`      | ASGI server to run FastAPI                 |
| `rdkit`        | Chemistry toolkit for SMILES parsing & 2D rendering |
| `scikit-learn` | ML model loading and prediction            |
| `shap`         | Explainable AI — feature importance        |
| `joblib`       | Model serialization/deserialization        |
| `fpdf2`        | PDF report generation                      |
| `pydantic`     | Request/response data validation           |

---

## 🔗 Frontend Integration

The API has **CORS enabled** (`allow_origins=["*"]`), so any React/Next.js/Vue frontend can connect directly.

**Example fetch from frontend:**

```javascript
const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ smiles: "CCO" }),
});
const data = await response.json();

// Display molecule image
const imgSrc = `data:image/png;base64,${data.molecule_image}`;

// Download PDF report
window.open(`http://127.0.0.1:8000${data.report_url}`);
```

---

## 🔄 Replacing the Dummy Model

When your ML teammate provides the real trained model:

1. Drop their `model.pkl` into the project root (replacing the dummy one)
2. Ensure their model expects the same 4 features: `MolWt`, `LogP`, `TPSA`, `NumRotatableBonds`
3. If their model uses different features, update `src/features.py` accordingly
4. Restart the server — no other code changes needed

---

## 📝 License

This project was built for a hackathon. Use responsibly for research purposes only.
