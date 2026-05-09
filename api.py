from pydantic import BaseModel
from fastapi import FastAPI
import os 
import joblib
import pandas as pd
from train import ChurnNet
from fastapi.middleware.cors import CORSMiddleware
import torch 


# Global variables to hold the model in memory
model = None
preprocessor = None

app = FastAPI()

class DataSchema(BaseModel):
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
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_FILE_DIR,"model","churn-model.pth")
preprocessor = os.path.join(CURRENT_FILE_DIR,"model","preprocessor.joblib")
@app.on_event("Startup")
def load_artifacts():

    global model, preprocessor
    if not os.path.exists(model_path):
        print(f"ERROR : File not Found at path {model_path}")
    try:
        # Load the Translator (Preprocessor)
        preprocessor = joblib.load(preprocessor)
        print(f"✅ Preprocessor loaded from {preprocessor}")

        # Load the Brain (Model)
        # We must initialize the architecture first!
        # Note: We need to know input_dim. Based on your training, let's assume ~12 features.
        # Ideally, you save input_dim in a config file, but we'll infer it or hardcode for now.
        # Let's use a dummy input to check preprocessor output size if possible, 
        # or just hardcode 12 if that's what X_train.shape[1] was.
        # SAFE BET: The preprocessor determines output columns. 
        # For Churn dataset, it's usually 11, 12, or 13 depending on OneHot.
        # Let's check the saved model layer size strictly.
        checkpoint = torch.load(model_path)
        input_dim = checkpoint['layers.0.weight'].shape[1] # Engineer Trick: Read shape from weights!
        
        model = ChurnNet(input_dim=input_dim)
        model.load_state_dict(checkpoint)
        model.eval() # Set to evaluation mode (No dropout)
        print(f"✅ Model loaded from {model_path} (Input Dim: {input_dim})")
    except Exception as e:
        print(f"Failed to Load Artifacts: {e}")
        raise e 
    

@app.get("/Home")
def home():
    return {"message":"ALive"}


@app.post("/predict")
def predict_churn(data : DataSchema):
    input_data = data.model_dump_json()
    df = pd.DataFrame 