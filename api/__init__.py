import requests

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
# Test 3: Prediction
# -----------------------------
print("Testing /predict endpoint...")

test_data = {
    "year": 2026,
    "month": 4,
    "previous_value": 150000
}

response = requests.post(
    BASE_URL + "/predict",
    json=test_data
)

print("Status Code:", response.status_code)
print("Response:", response.json())