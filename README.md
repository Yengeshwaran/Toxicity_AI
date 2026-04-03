# 🧬 ToxiGuard AI (CodeCure)

> Full-Stack AI-powered drug toxicity prediction system built for the **AI Biohackathon (Track A)**.

ToxiGuard AI takes a [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) string as input and returns a toxicity prediction, confidence score, SHAP-driven explainability, a 2D molecule image, a detailed AI-generated analysis (via Gemini), and a downloadable PDF report.

---

## 📁 Project Structure

This directory contains both the frontend and backend of the application:

```
toxiguard-ai/
├── backend/            # FastAPI + RDKit + SHAP + Gemini AI
├── frontend/           # React + Vite + Tailwind + shadcn/ui
└── README.md           # This file
```

---

## 🚀 Quick Start Guide

To run the full application, you need to start **both** the Backend API and the Frontend UI in separate terminal windows.

### ⚙️ Step 1: Start the Backend (API)
The backend is built with FastAPI and runs the ML model and RDKit molecule processing.

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a Virtual Environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up Gemini AI (Optional but recommended):
   Create a `.env` file in the `backend` folder and add:
   ```env
   GEMINI_API_KEY=your-api-key-here
   ```
5. *(Initial Run Only)* Generate the Placeholder ML Model:
   ```bash
   python src/generate_dummy_model.py
   ```
6. Start the API Server:
   ```bash
   uvicorn src.api:app --reload
   ```
> The API will now be running at **http://127.0.0.1:8000**

---

### 💻 Step 2: Start the Frontend (UI)
The frontend is a modern React application built with Vite and Tailwind CSS.

1. Open a **new** terminal window and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install Node Dependencies:
   ```bash
   npm install
   ```
3. Start the Development Server:
   ```bash
   npm run dev
   ```
> The Website will now be running at **http://localhost:5173**

---

## ✨ System Features

### 🖥️ Frontend UI 
- **Modern Interface**: Clean input for SMILES strings with quick-test molecule examples.
- **Visual Analytics**: Interactive displays for model confidence, and top contributing factors.
- **RDKit Molecule Viewer**: Displays the beautiful 2D rendering of the molecular structure retrieved from the backend.
- **Downloadable PDF Reports**: Quick access to formalized toxicity findings.

### 🧠 Backend AI & ML
- **Scikit-learn Prediction**: Loads trained `.pkl` models to identify toxicity likelihood.
- **Feature Extraction Pipeline**: Translates SMILES directly to `MolWt`, `LogP`, `TPSA`, and `NumRotatableBonds`.
- **SHAP Explainability**: Returns quantified metric importance (e.g., `#1 LogP, #2 TPSA`) detailing exactly *why* the model made its decision.
- **Gemini NLP Interpretation**: Seamlessly analyzes the parameters to provide a human-readable pharmaceutical summary.

---

## 🧪 How to use

1. Go to **http://localhost:5173**
2. Enter a SMILES string like `CCO` (Ethanol) or `c1ccccc1` (Benzene).
3. Hit **Predict**. 
4. The frontend will dispatch the string to the backend, which will stream back the generated molecule image, Toxicity Confidence, SHAP feature metrics, and the detailed Gemini AI analysis.
5. Click **Download Full PDF Report** to save everything for your records.

---

## 📝 License
Created under Hackathon guidelines. For research and demonstration purposes only.
