**🏦 Bank Churn Prediction API**
Live API: [https://churn-prediction-api-6vtf.onrender.com/docs]

**Status: 🟢 Deployed & Operational**

A high-performance PyTorch deep learning model served via FastAPI to predict bank customer attrition. This project demonstrates a full machine learning lifecycle—from data preprocessing and F1-score optimization to cloud deployment.

**🔗 Project Links**
GitHub Repository: [https://github.com/Laksh-tech/churn-prediction-api]

Live Endpoint: [https://churn-prediction-api-6vtf.onrender.com/docs] (Swagger UI)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**⚡ Quick Start (Usage)**
You can test the API directly through the live Swagger documentation or via curl:

curl -X 'POST' \
  'https://churn-prediction-api-6vtf.onrender.com' \
  -H 'Content-Type: application/json' \
  -d '{
  "CreditScore": 650,
  "Geography": "Germany",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 12500.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "EstimatedSalary": 50000.0
}'


**🏗️ Technical Architecture**
1. Model: Neural Network built with PyTorch featuring Dropout layers for regularization.

2. Pipeline: Custom Scikit-Learn pipeline for handling categorical encodings and feature scaling.

3. Optimization: The decision threshold is tuned to 0.6 to maximize the F1-Score, ensuring a balance between detecting actual churners and avoiding false positives.

4. API Layer: FastAPI provides asynchronous processing, making the inference fast and scalable.

5. CI/CD: Automatically deployed to Render via GitHub integration.

📊 **Model Performance**
**|| Metric   | Value
||Recall    |  75.32%
||F1-Score  | 61.92%
||Threshold |  0.6**
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📝 Future Improvements
[ ] Implement a frontend dashboard using Streamlit.

[ ] Batch prediction support via CSV upload.

[ ] Real-time logging and monitoring for model drift.

Author: Laksh-tech

2nd Year Student | Machine Learning Enthusiast
