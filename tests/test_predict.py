# tests/test_predict.py
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the /health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["model_loaded"] is True
    print("Health check passed!")

def test_info():
    """Test the /info endpoint"""
    response = requests.get(f"{BASE_URL}/info")
    assert response.status_code == 200
    json_data = response.json()

    assert "model_type" in json_data
    assert "features_expected" in json_data

    expected_features = ["lag_1", "lag_2", "lag_52"]
    assert json_data["features_expected"] == expected_features

    print("Info check passed!")

def test_predict_valid():
    """Test the /predict endpoint with valid data"""
    test_data = {
        "lag_1": 100000,
        "lag_2": 95000,
        "lag_52": 90000
    }

    response = requests.post(f"{BASE_URL}/predict", json=test_data)

    assert response.status_code == 200
    json_data = response.json()

    assert json_data["status"] == "success"
    assert "prediction" in json_data

    print(f"Prediction successful: {json_data['prediction']}")

def test_predict_missing_feature():
    """Test the /predict endpoint with missing features"""
    test_data = {
        "lag_1": 100000,
        "lag_2": 95000
    }

    response = requests.post(f"{BASE_URL}/predict", json=test_data)

    assert response.status_code == 422
    json_data = response.json()

    assert "detail" in json_data

    print("Missing feature test passed!")
    