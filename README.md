# Churn Prediction API

End-to-end bank customer churn prediction system with a live interactive dashboard.

**Live Demo:** https://churn-prediction-api.vercel.app  
**API:** https://lakshtech-churn-api.hf.space

---

## Architecture
GitHub → Hugging Face Spaces (FastAPI + PyTorch) ← POST /predict → Vercel (Dashboard)

API and UI are deployed independently. HF Spaces handles heavy ML dependencies inside Docker. Vercel serves the static dashboard via CDN.

---

## Data

**Dataset:** [Churn Modelling Dataset](https://github.com/sharmaroshan/Churn-Modelling-Dataset) — 10,000 bank customers, 14 features, 20% churn rate (imbalanced).

**Cleaning & Preprocessing:**
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- One-hot encoded `Geography` (France / Germany / Spain)
- Label encoded `Gender`
- Standard scaled all numeric features: `CreditScore`, `Age`, `Balance`, `EstimatedSalary`
- Built as a scikit-learn `Pipeline` — fit on train, transform on test and production, preventing data leakage

---

## Model

**Architecture:** Multi-Layer Perceptron (PyTorch)
Input (13) → Linear(64) → ReLU → BatchNorm → Dropout(0.3)
→ Linear(32) → ReLU → BatchNorm → Dropout(0.3)
→ Linear(1)  → Sigmoid

- `BCEWithLogitsLoss` with `pos_weight=4.0` to handle class imbalance
- Adam optimizer with `ReduceLROnPlateau` scheduler
- Early stopping on F1 score (patience=10)

---

## Threshold Tuning

Default classification threshold is 0.5. For imbalanced churn data, this misses too many actual churners (low recall).

Evaluated thresholds from 0.3 → 0.7:

| Threshold | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| 0.3       | 41.2%     | 88.1%  | 56.1% |
| 0.4       | 52.3%     | 81.4%  | 63.7% |
| **0.5**   | **58.1%** | 76.2%  | 65.9% |
| **0.6**   | **64.3%** | 75.3%  | **61.9%** |
| 0.7       | 71.2%     | 61.4%  | 65.9% |

**Chosen: 0.6** — best balance of precision and recall for a business use case where missing a churner is more costly than a false alarm.

---

## Stack

| Layer | Technology |
|---|---|
| Model | PyTorch MLP |
| Preprocessing | scikit-learn Pipeline |
| API | FastAPI + Pydantic |
| Containerization | Docker |
| API Hosting | Hugging Face Spaces |
| Frontend | Vanilla HTML/CSS/JS |
| Frontend Hosting | Vercel |
| CI/CD | GitHub Actions |

---

## Project Structure
churn-prediction-api/
├── app/
│   ├── main.py              # FastAPI app, endpoints, model loading
│   ├── core/
│   │   └── preprocessing.py # sklearn pipeline
│   └── model/
│       ├── churn_model.pth  # trained weights
│       └── preprocessor.joblib
├── frontend/
│   └── index.html           # dashboard UI
├── .github/workflows/
│   └── ci.yml               # GitHub Actions
├── train.py                 # model training script
├── Dockerfile
└── requirements.txt
---

## API

**POST** `/predict`

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 38,
  "Tenure": 5,
  "Balance": 75000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 80000.0
}
```

Response:
```json
{
  "probability": 0.3241,
  "is_churn": false,
  "risk_level": "Low",
  "message": "Employee is likely to stay."
}
```

**GET** `/Output` — health check

---

## Run Locally

```bash
git clone https://github.com/Laksh-tech/churn-prediction-api.git
cd churn-prediction-api
pip install -r requirements.txt
python train.py                          # generates model weights
uvicorn app.main:app --reload            # starts API on localhost:8000
```

Open `frontend/index.html` in browser, update `API_BASE` to `http://localhost:8000`.
