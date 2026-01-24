from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib # Better than pickle for scikit-learn objects
from sklearn.model_selection import train_test_split
from app.core.preprocessing import Churn_Modelling as ChurnPreprocessor
from sklearn.metrics import precision_score, recall_score, f1_score
# 1. CONFIG
# UPDATE THIS LINE IN train.py
DATA_URL = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"
MODEL_PATH = "D:\\Datasets\\Churn-Prediction\\app\\model\\churn_model.pth"
PREPROCESSOR_PATH = "D:\\Datasets\\Churn-Prediction\\app\\model\\preprocessor.joblib"
 
# 2. THE PYTORCH MODEL
class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super(ChurnNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), # Batch Norm helps with real data stability
            nn.Dropout(0.3),    # Dropout prevents overfitting
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1),
            # nn.Sigmoid() # Moved to loss function for numerical stability
        )
        
    def forward(self, x):
        return self.layers(x)

# 3. EXECUTION
def main():
    print("⏳ Downloading Real Data...")
    df = pd.read_csv(DATA_URL)
    print(f"   Data Shape: {df.shape}")
    
    # Separate Target
    X = df.drop('Exited', axis=1)
    y = df['Exited'].values
    
    # --- ENGINEERING SKILL SHOWCASE ---
    print("⚙️ Running Preprocessing Pipeline...")
    processor = ChurnPreprocessor()
    
    # This learns 'Average Age', 'Countries present', etc.
    X_processed = processor.fit_transform(X)
    
    # Convert to Tensor
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    print(f"   Input Features: {X_train.shape[1]}") # Should be around 11-13 depending on encoding
    
    # --- TRAINING ---
    model = ChurnNet(input_dim=X_train.shape[1])
    # Calculate weight: (Negatives / Positives) roughly 4.0 for this dataset
    pos_weight = torch.tensor([4.0]) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 1. Define Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    # --- BEFORE THE LOOP (Crucial!) ---
    best_f1 = 0.0
    patience_counter = 0
    print("🚀 Training Model...")

    for epoch in range(50):
        # 1. Training Step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 2. Validation Step (Every 10 epochs)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_test)
                
                # FIX A: Apply Sigmoid to convert Logits -> Probabilities
                val_probs = torch.sigmoid(val_logits)
                
                # Convert to Binary (Threshold 0.5)
                val_preds = (val_probs > 0.5).float()
                
                # FIX B: Calculate Metrics FIRST (Before using them in 'if' check)
                y_true_np = y_test.cpu().numpy()
                y_pred_np = val_preds.cpu().numpy()
                
                val_acc = (val_preds == y_test).float().mean().item()
                prec = precision_score(y_true_np, y_pred_np, zero_division=0)
                rec = recall_score(y_true_np, y_pred_np, zero_division=0)
                f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

                # Print Dashboard
                print(f"  Epoch {epoch+1} | Loss: {loss.item():.4f}")
                print(f"      📊 Acc: {val_acc:.2%} | Prec: {prec:.2%} | Rec: {rec:.2%} | F1: {f1:.2%}")

                # FIX C: Now we can check 'if f1 > best_f1'
                if f1 > best_f1:
                    best_f1 = f1
                    patience_counter = 0
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"    💾 Saved Best F1 Model (F1: {best_f1:.2%})")
                else:
                    patience_counter += 1
                
                if patience_counter > 10:
                    print("🛑 Early Stopping triggered.")
                    break
            # 5. Dashboard Metrics (Using Scikit-Learn)
            # MUST move tensors to CPU and convert to numpy for sklearn
                y_true_np = y_test.cpu().numpy()
                y_pred_np = val_preds.cpu().numpy()
            
                prec = precision_score(y_true_np, y_pred_np, zero_division=0)
                rec = recall_score(y_true_np, y_pred_np, zero_division=0)
                f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

                print(f"  Epoch {epoch+1} | Loss: {loss.item():.4f}")
                print(f"      📊 Acc: {val_acc:.2%} | Prec: {prec:.2%} | Rec: {rec:.2%} | F1: {f1:.2%}")
                # --- SAVING ARTIFACTS (CRITICAL FOR PROD) ---
    
    # torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(processor, PREPROCESSOR_PATH)
    print("✅ Done! Ready for Deployment.")
    # --- POST-TRAINING ANALYSIS ---
    print("\n🔍 FINDING OPTIMAL THRESHOLD (The Business Decision)")

    # Load the best model we saved
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_test.cpu().numpy()

# Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"{'Thresh':<10} | {'Prec':<10} | {'Rec':<10} | {'F1':<10} | {'Impact'}")
    print("-" * 60)

    for t in thresholds:
        preds = (probs > t).astype(float)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
    
        impact = ""
        if r > 0.7: impact = "🔥 High Recall (Aggressive)"
        elif p > 0.7: impact = "🎯 High Precision (Safe)"
    
        print(f"{t:<10} | {p:.2%}     | {r:.2%}     | {f:.2%}     | {impact}")
if __name__ == "__main__":
    main()
