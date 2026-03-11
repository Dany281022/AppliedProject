import requests

# Base URL of your running Flask API
BASE_URL = "http://localhost:5000"

# -----------------------------
# Test 1: Health check
# -----------------------------
print("Testing /health endpoint...")
response = requests.get(BASE_URL + "/health")
print("Status Code:", response.status_code)
print("Response:", response.json())
print()

# -----------------------------
# Test 2: Model info
# -----------------------------
print("Testing /info endpoint...")
response = requests.get(BASE_URL + "/info")
print("Status Code:", response.status_code)
print("Response:", response.json())
print()

# -----------------------------
# Test 3: Prediction with valid data
# -----------------------------
print("Testing /predict endpoint with valid data...")

valid_data = {
    "lag_1": 150000,
    "lag_2": 149500,
    "lag_52": 145000
}

response = requests.post(BASE_URL + "/predict", json=valid_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
print()

# -----------------------------
# Test 4: Prediction with missing features
# -----------------------------
print("Testing /predict endpoint with missing feature(s)...")

missing_feature_data = {
    "lag_1": 150000,
    "lag_2": 149500
    # lag_52 is missing
}

response = requests.post(BASE_URL + "/predict", json=missing_feature_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
print()

# -----------------------------
# Test 5: Prediction with wrong data types
# -----------------------------
print("Testing /predict endpoint with wrong data type...")

wrong_type_data = {
    "lag_1": "not_a_number",
    "lag_2": 149500,
    "lag_52": 145000
}

response = requests.post(BASE_URL + "/predict", json=wrong_type_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
print()

# -----------------------------
# Test 6: Prediction with empty request
# -----------------------------
print("Testing /predict endpoint with empty request...")

empty_data = {}

response = requests.post(BASE_URL + "/predict", json=empty_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
print()