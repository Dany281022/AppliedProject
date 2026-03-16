# tests/test_integration.py
import requests
import time

API_URL = "http://localhost:8000"

def test_api_health():
    """Test API is running"""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    print("✅ API health check passed")

def test_prediction_valid():
    """Test prediction with valid data"""
    data = {
        "lag_1": 100.0,
        "lag_2": 95.0,
        "lag_52": 88.0
    }
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    print(f"✅ Valid prediction: {response.json()['prediction']}")

def test_prediction_missing_field():
    """Test prediction with missing field"""
    data = {"lag_1": 100.0}  # Missing lag_2 and lag_52
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 422
    print("✅ Missing field handled correctly")

def test_prediction_invalid_type():
    """Test prediction with invalid data type"""
    data = {"lag_1": "invalid", "lag_2": 95.0, "lag_52": 88.0}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code in [400, 422, 500]
    print("✅ Invalid type handled correctly")

def test_prediction_empty_request():
    """Test prediction with empty request"""
    response = requests.post(f"{API_URL}/predict", json={})
    assert response.status_code == 422
    print("✅ Empty request handled correctly")

def test_prediction_minimum_values():
    """Test prediction with minimum values"""
    data = {"lag_1": 0.0, "lag_2": 0.0, "lag_52": 0.0}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 200
    print("✅ Minimum values work")

def test_prediction_large_values():
    """Test prediction with large values"""
    data = {"lag_1": 99999.0, "lag_2": 99999.0, "lag_52": 99999.0}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 200
    print("✅ Large values work")

def test_response_time():
    """Test that response time is acceptable"""
    data = {"lag_1": 100.0, "lag_2": 95.0, "lag_52": 88.0}
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json=data)
    elapsed = time.time() - start
    assert elapsed < 5.0
    print(f"✅ Response time: {elapsed:.2f}s")

def test_api_info():
    """Test /info endpoint"""
    response = requests.get(f"{API_URL}/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features_expected" in data
    print(f"✅ Model info: {data['model_type']}, features: {data['features_expected']}")

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*50)
    print("INTEGRATION TESTS - Weekly Sales Prediction")
    print("="*50 + "\n")

    tests = [
        test_api_health,
        test_prediction_valid,
        test_prediction_missing_field,
        test_prediction_invalid_type,
        test_prediction_empty_request,
        test_prediction_minimum_values,
        test_prediction_large_values,
        test_response_time,
        test_api_info,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "="*50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*50)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
    