import pytest
from fastapi.testclient import TestClient
from app.main import app  # Assuming your script is named main.py inside an app folder

# # 1. Initialize the 'Judge'
# client = TestClient(app)

# 2. Test Case: Check if the Health Check works
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/Output")
        assert response.status_code == 200
        assert response.json() == {"status": "System Operational", "model_version": "v1.0"}

# 3. Test Case: Valid Data (Expect a successful prediction)
def test_predict_success():
    with TestClient(app) as client:
        payload = {
            "CreditScore": 600, "Geography": "France", "Gender": "Female", 
            "Age": 40, "Tenure": 3, "Balance": 60000.0, "NumOfProducts": 2, 
            "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000.0
        }
        response = client.post("/predict", json=payload)
    
    # Assertions: Did it pass the "LeetCode" requirements?
        assert response.status_code == 200, f"API failed with: {response.json()}"
        data = response.json()
        assert "probability" in data
        assert "is_churn" in data
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0  # Probabilities must be between 0 and 1

# 4. Test Case: Invalid Data (Expect a 422 Unprocessable Entity error)
def test_predict_invalid_data():
    with TestClient(app) as client:
    # Sending a string for 'Age' instead of an int
        payload = {"CreditScore": "Six Hundred", "Age": "Young"} 
        response = client.post("/predict", json=payload)
    
        assert response.status_code == 422  # FastAPI bouncer should catch this automatically!

# 5. Test Case: Logic Check (Threshold)
def test_threshold_logic():
    # Mock a payload
    with TestClient(app) as client:
        payload = {
        "CreditScore": 600, "Geography": "Spain", "Gender": "Female", 
        "Age": 40, "Tenure": 3, "Balance": 60000.0, "NumOfProducts": 1, 
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000.0
        }
        response = client.post("/predict", json=payload)
        data = response.json()
    
    # If prob > 0.6, is_churn must be True
        if data["probability"] > 0.6:
            assert data["is_churn"] is True
        else:
            assert data["is_churn"] is False