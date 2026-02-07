import torch
import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.preprocessing import Churn_Modelling as ChurnPreprocessor # Import your class
from train import ChurnNet # Import your model structure

# --- 1. SETUP & LOADING ---
app = FastAPI(title="Churn Prediction API", version="1.0")

# Get the absolute path of the directory where main.py sits (the 'app' folder)
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point directly to the model folder inside 'app'
MODEL_PATH = os.path.join(CURRENT_FILE_DIR, "model", "churn_model.pth")
PREPROCESSOR_PATH = os.path.join(CURRENT_FILE_DIR, "model", "preprocessor.joblib")
# Global variables to hold the model in memory
model = None
preprocessor = None

# This runs ONE TIME when the server starts
@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    print(f"DEBUG: Looking for model at {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: File not found at {MODEL_PATH}")
    try:
        # Load the Translator (Preprocessor)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Preprocessor loaded from {PREPROCESSOR_PATH}")

        # Load the Brain (Model)
        # We must initialize the architecture first!
        # Note: We need to know input_dim. Based on your training, let's assume ~12 features.
        # Ideally, you save input_dim in a config file, but we'll infer it or hardcode for now.
        # Let's use a dummy input to check preprocessor output size if possible, 
        # or just hardcode 12 if that's what X_train.shape[1] was.
        # SAFE BET: The preprocessor determines output columns. 
        # For Churn dataset, it's usually 11, 12, or 13 depending on OneHot.
        # Let's check the saved model layer size strictly.
        checkpoint = torch.load(MODEL_PATH)
        input_dim = checkpoint['layers.0.weight'].shape[1] # Engineer Trick: Read shape from weights!
        
        model = ChurnNet(input_dim=input_dim)
        model.load_state_dict(checkpoint)
        model.eval() # Set to evaluation mode (No dropout)
        print(f"✅ Model loaded from {MODEL_PATH} (Input Dim: {input_dim})")
        
    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}")
        raise e

# --- 2. THE BOUNCER (Data Validation) ---
class EmployeeData(BaseModel):
    # These must match your CSV column names exactly!
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# --- 3. THE ENDPOINT (The Waiter) ---
@app.post("/predict")
def predict_churn(data: EmployeeData):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Convert JSON -> Pandas DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # 2. Preprocess (The Translator)
        # Using the exact same logic as training!
        processed_data = preprocessor.transform(df)
        
        # 3. Model Inference (The Brain)
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)
        
        with torch.no_grad():
            logits = model(tensor_data)
            probability = torch.sigmoid(logits).item()
        
        # 4. Business Logic (Threshold 0.6 based on your analysis)
        threshold = 0.6
        churn_risk = probability > threshold
        
        return {
            "probability": round(probability, 4),
            "is_churn": bool(churn_risk),
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
            "message": "Employee is at risk of leaving." if churn_risk else "Employee is likely to stay."
        }

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return {"error": str(e)}

# --- 4. HEALTH CHECK ---
@app.get("/Output")
def home():
    return {"status": "System Operational", "model_version": "v1.0"}

