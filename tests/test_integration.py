# tests/test_integration.py
import requests
import time

# Use 127.0.0.1 instead of localhost to avoid Windows DNS resolution
# overhead (~2s delay caused by IPv6 fallback on Windows systems)
API_URL = "http://127.0.0.1:8000"

VALID_DATA = {
    "lag_1": 100.0,
    "lag_2": 95.0,
    "lag_4": 90.0,
    "lag_8": 88.0,
    "lag_12": 85.0,
    "lag_26": 80.0,
    "lag_52": 75.0,
    "ma_4": 93.0,
    "ma_12": 87.0,
    "std_4": 5.0,
    "weekofyear": 12,
    "month": 3,
    "year": 2026
}

def test_api_health():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ API health check passed")

def test_api_info():
    response = requests.get(f"{API_URL}/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features_expected" in data
    assert data["num_features"] == 13
    print(f"✅ Model info: {data['model_type']}, features: {data['num_features']}")

def test_prediction_valid():
    response = requests.post(f"{API_URL}/predict", json=VALID_DATA)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert "status" in result
    assert "response_time_ms" in result
    assert result["status"] == "success"
    print(f"✅ Valid prediction: ${result['prediction']:,.2f} in {result['response_time_ms']}ms")

def test_prediction_missing_field():
    data = {"lag_1": 100.0, "lag_2": 95.0}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 422
    print("✅ Missing fields handled correctly (422)")

def test_prediction_invalid_type():
    data = {**VALID_DATA, "lag_1": "invalid"}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code in [400, 422, 500]
    print("✅ Invalid type handled correctly")

def test_prediction_empty_request():
    response = requests.post(f"{API_URL}/predict", json={})
    assert response.status_code == 422
    print("✅ Empty request handled correctly (422)")

def test_prediction_minimum_values():
    data = {**VALID_DATA, "lag_1": 0.0, "lag_2": 0.0, "lag_52": 0.0}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 200
    print("✅ Minimum values work")

def test_prediction_large_values():
    data = {**VALID_DATA, "lag_1": 99999.0, "lag_2": 99999.0, "lag_52": 99999.0}
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 200
    print("✅ Large values work")

def test_response_time():
    # First call may be slow due to Windows DNS — measure subsequent calls
    requests.post(f"{API_URL}/predict", json=VALID_DATA)  # warm-up
    times = []
    for _ in range(5):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=VALID_DATA)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    avg_ms = sum(times) / len(times)
    result = response.json()
    model_time = result.get("response_time_ms", 0)
    assert avg_ms < 1000, f"Average response time {avg_ms:.1f}ms exceeds 1000ms target"
    print(f"✅ Avg response time: {avg_ms:.1f}ms | model inference: {model_time}ms")

def test_prediction_response_structure():
    response = requests.post(f"{API_URL}/predict", json=VALID_DATA)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert "status" in result
    assert "response_time_ms" in result
    assert result["status"] == "success"
    print("✅ Response structure is correct (prediction, confidence, status, response_time_ms)")

def run_all_tests():
    print("\n" + "="*50)
    print("INTEGRATION TESTS - Weekly Sales Prediction")
    print("="*50 + "\n")

    tests = [
        test_api_health,
        test_api_info,
        test_prediction_valid,
        test_prediction_missing_field,
        test_prediction_invalid_type,
        test_prediction_empty_request,
        test_prediction_minimum_values,
        test_prediction_large_values,
        test_response_time,
        test_prediction_response_structure,
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