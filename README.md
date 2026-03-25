# Weekly Sales Prediction App
AIE1014 — AI Applied Project Course | Milestone 4

## What This Project Does
This application predicts weekly sales figures for retail business managers using a RandomForestRegressor model trained on historical lag features. Users enter three lag values (previous week, 2 weeks ago, and 1 year ago) through a web interface and receive instant sales forecasts to support inventory and staffing decisions.

**Stakeholder:** Retail Business Manager  
**GitHub:** https://github.com/Dany281022/AppliedProject

## Team
| Name | Role |
|------|------|
| Dany Deugoue | MLOps Engineer |

## Prerequisites
- Python 3.8 or higher
- pip
- Git (optional)

## Installation
1. Clone the repository
```bash
git clone https://github.com/Dany281022/AppliedProject.git
cd Millestone04_TeamDany
```

2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Step 1 — Start the API (Terminal 1)
```bash
cd api
python app.py
```
API runs at: http://localhost:8000  
Interactive docs at: http://localhost:8000/docs

### Step 2 — Start the UI (Terminal 2)
```bash
cd ui
streamlit run app_ui.py
```
App runs at: http://localhost:8501

## Running the Tests
```bash
python tests/test_integration.py
```
Expected output: 9 passed, 0 failed

## Project Structure
```
Millestone04_TeamDany/
├── api/
│   ├── app.py               ← FastAPI server
│   ├── model.pkl            ← Trained RandomForestRegressor
│   └── requirements.txt
├── ui/
│   ├── app_ui.py            ← Streamlit interface
│   └── requirements.txt
├── tests/
│   ├── test_integration.py  ← Integration tests (9/9 passing)
│   └── test_results.txt     ← Test output
├── docs/
│   └── TeamDany_Milestone4_Report.pdf
├── README.md
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description      |
|----------|--------|------------------|
| /health  | GET    | Health check     |
| /predict | POST   | Make a prediction|
| /info    | GET    | Model information|

### Example Request
```json
POST /predict
{
  "lag_1": 100.0,
  "lag_2": 95.0,
  "lag_52": 88.0
}
```

### Example Response
```json
{
  "prediction": 45779299.68,
  "status": "success"
}
```

## Model Information
- **Algorithm:** RandomForestRegressor
- **Target:** Weekly sales figures (float)
- **Features:** lag_1 (previous week), lag_2 (2 weeks ago), lag_52 (1 year ago)
- **Tests:** 9/9 integration tests passing, response time ~2s

## Error Codes
| Code | Meaning |
|------|---------|
| 200  | Success |
| 422  | Missing or invalid input fields |
| 500  | Server/model error |

## Troubleshooting
| Problem | Solution |
|---------|----------|
| Port already in use | Kill the process or use `--port 8001` |
| Module not found | Run `pip install -r requirements.txt` |
| Cannot connect to API | Make sure the API is running in Terminal 1 |
| Model file not found | Ensure `model.pkl` is in the `api/` directory |

## Known Issues & Limitations
- UI requires the API running in a separate terminal before launch
- Model trained on only 3 lag features — cannot capture promotions or seasonality
- No authentication — API is open on localhost only

## Future Improvements
- Deploy API to Render and UI to Streamlit Cloud for public access
- Add more features (promotions, holidays, store location)
- Add prediction history chart to the UI

---
AIE1014 — AI Applied Project Course | Team Dany | Winter 2026