# Weekly Sales Prediction App
AIE1014 — AI Applied Project Course | Milestone 4

## What This Project Does
This application predicts weekly sales figures for retail business managers using a RandomForestRegressor model trained on historical lag features. Users enter three lag values (previous week, 2 weeks ago, and 1 year ago) through a web interface and receive instant sales forecasts to support inventory and staffing decisions.

## Team
| Name | Role | 
|------|------|
| Dany Deugoue | MLOps Engineer |

**Stakeholder:** Retail Business Manager  
**GitHub:** https://github.com/Dany281022/AppliedProject

## Prerequisites
- Python 3.8 or higher
- pip
- Git (optional)

## Installation
```bash
git clone https://github.com/Dany281022/AppliedProject.git
cd Millestone04_TeamDany
python -m pip install -r requirements.txt
```

## Running the Application

### Step 1 — Start the API
```bash
cd api
python app.py
```
API runs at: http://localhost:8000  
Interactive docs at: http://localhost:8000/docs

### Step 2 — Start the UI (open a new terminal)
```bash
cd ui
streamlit run app_ui.py
```
App runs at: http://localhost:8501

## Running the Tests
```bash
python tests/test_integration.py
```

## Project Structure
```
Millestone04_TeamDany/
├── api/
│   ├── app.py               ← FastAPI server
│   └── model.pkl            ← Trained RandomForestRegressor
│   └── requirements.txt
├── ui/
│   ├── app_ui.py            ← Streamlit interface
│   └── requirements.txt
├── tests/
│   └── test_integration.py  ← Integration tests (9/9 passing)
├── docs/
│   └── TeamDany_Milestone4_Report.pdf
├── models/
│   └── initial_model.pkl    ← Original model backup
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description       |
|----------|--------|-------------------|
| /health  | GET    | Health check      |
| /predict | POST   | Make a prediction |
| /info    | GET    | Model information |

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
- **Key features:** lag_1 (previous week), lag_2 (2 weeks ago), lag_52 (1 year ago)
- **Performance:** 9/9 integration tests passing, response time 2.02s

## Error Codes
| Code | Meaning |
|------|---------|
| 200 | Success |
| 422 | Missing or invalid input fields |
| 500 | Server/model error |

## Known Issues & Limitations
- UI requires the API running in a separate terminal before launch
- Model trained on only 3 lag features — cannot capture promotions, seasonality, or external factors
- No authentication — API is open on localhost only
- Predictions require manual input of historical values (no database connection)

## Next Steps
- Deploy API to Render and UI to Streamlit Cloud for public stakeholder access
- Add more features (promotions, holidays, store location) to improve accuracy
- Add prediction history chart to track forecasts over time

## Submission
- **Due:** Wednesday, March 25, 2026 at 12:30 PM
- **Submission:** Moodle → Assessments → Applied Activities → Applied Activity 04 (10%)
- **File:** TeamDany_Milestone4_Report.pdf
