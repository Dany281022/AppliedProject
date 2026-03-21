# test_api.py
import requests

BASE_URL = 'http://localhost:8000'

# Test 1: Health check
print("Testing /health...")
response = requests.get(f'{BASE_URL}/health')
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 2: Valid prediction
print("Testing /predict (valid)...")
payload = {
    'lag_1': 10.0,
    'lag_2': 20.0,
    'lag_52': 15.0
}
response = requests.post(f'{BASE_URL}/predict', json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 3: Empty request
print("Testing /predict (empty)...")
response = requests.post(f'{BASE_URL}/predict', json={})
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 4: Missing lag_2
print("Testing /predict (missing lag_2)...")
payload = {
    'lag_1': 10.0,
    'lag_52': 15.0
}
response = requests.post(f'{BASE_URL}/predict', json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")