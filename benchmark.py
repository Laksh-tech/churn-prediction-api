import torch
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from app.core.preprocessing import Churn_Modelling as ChurnPreprocessor
from train import ChurnNet # Import your NN class

# 1. SETUP DATA (Same pipeline as before)
DATA_URL = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"
MODEL_PATH = "app/model/churn_model.pth"

print("⏳ Loading Data...")
df = pd.read_csv(DATA_URL)
X = df.drop('Exited', axis=1)
y = df['Exited'].values

# Preprocess
processor = ChurnPreprocessor()
X_processed = processor.fit_transform(X) # Note: In real bench, use the SAVED preprocessor!
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- CONTENDER 1: NEURAL NETWORK ---
print("\n🥊 Loading Neural Network...")
checkpoint = torch.load(MODEL_PATH)
input_dim = X_train.shape[1]
nn_model = ChurnNet(input_dim)
nn_model.load_state_dict(checkpoint)
nn_model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    nn_logits = nn_model(X_tensor)
    nn_probs = torch.sigmoid(nn_logits).numpy().flatten()
    nn_preds = (nn_probs > 0.6).astype(int) # Using your threshold

# --- CONTENDER 2: XGBOOST ---
print("🥊 Training XGBoost (The Challenger)...")
# XGBoost handles scale well, but we use the same processed data for fairness
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1] # Probability of class 1
xgb_preds = (xgb_probs > 0.6).astype(int)

# --- THE SCOREBOARD ---
def print_metrics(name, y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    print(f"📊 {name:<15} | Acc: {acc:.2%} | F1: {f1:.2%} | AUC: {auc:.4f}")

print("\n🏆 FINAL RESULTS 🏆")
print("-" * 60)
print_metrics("Neural Network", y_test, nn_preds, nn_probs)
print_metrics("XGBoost", y_test, xgb_preds, xgb_probs)
print("-" * 60)
