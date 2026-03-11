import requests

BASE_URL = 'http://127.0.0.1:5000'

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
# Test 3: Prediction
# -----------------------------
url = BASE_URL + "/predict"

# ----- Valid data -----
print("Test with valid data")
response = requests.post(url, json={
    "lag_1": 149000,
    "lag_2": 148500,
    "lag_52": 140000
})
print(response.status_code)
print(response.json())
print()

# ----- Missing feature -----
print("Test with missing data")
response = requests.post(url, json={
    "lag_1": 149000,
    "lag_2": 148500
})
print(response.status_code)
print(response.json())
print()

# ----- Wrong type -----
print("Test with wrong type")
response = requests.post(url, json={
    "lag_1": "not_a_number",
    "lag_2": 148500,
    "lag_52": 140000
})
print(response.status_code)
print(response.json())
print()

# ----- Empty request -----
print("Test with empty request")
response = requests.post(url, json={})
print(response.status_code)
print(response.json())
