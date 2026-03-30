# Weekly Sales Prediction App
AIE1014 — AI Applied Project Course | Milestone 4

## What This Project Does
This application predicts weekly sales figures for retail business managers using a RandomForestRegressor model trained on 13 historical lag and moving average features. Users enter sales history values through a web interface and receive instant sales forecasts with a prediction interval to support inventory and staffing decisions.

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
API runs at: http://127.0.0.1:8000  
Interactive docs at: http://127.0.0.1:8000/docs

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
Expected output: 10 passed, 0 failed

## Project Structure
```
Millestone04_TeamDany/
├── api/
│   ├── app.py               ← FastAPI server
│   ├── model.pkl            ← Trained RandomForestRegressor (13 features)
│   └── requirements.txt
├── ui/
│   ├── app_ui.py            ← Streamlit interface with dashboard
│   └── requirements.txt
├── tests/
│   ├── test_integration.py  ← Integration tests (10/10 passing)
│   └── test_results.txt     ← Test output
├── docs/
│   └── TeamDany_Milestone4_Report.pdf
├── README.md
└── requirements.txt
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
  "lag_1": 40000000.0,
  "lag_2": 39000000.0,
  "lag_4": 38000000.0,
  "lag_8": 37000000.0,
  "lag_12": 36000000.0,
  "lag_26": 35000000.0,
  "lag_52": 34000000.0,
  "ma_4": 38500000.0,
  "ma_12": 37000000.0,
  "std_4": 500000.0,
  "weekofyear": 13,
  "month": 3,
  "year": 2026
}
```

### Example Response
```json
{
  "prediction": 43560477.77,
  "confidence": "USD 39,204,430 - USD 47,916,526 (+-10% interval)",
  "status": "success",
  "response_time_ms": 13.1
}
```

## Model Information
- **Algorithm:** RandomForestRegressor
- **Target:** Weekly sales figures (float)
- **Features:** 13 (lag_1, lag_2, lag_4, lag_8, lag_12, lag_26, lag_52, ma_4, ma_12, std_4, weekofyear, month, year)
- **RMSE:** $2,034,160
- **MAE:** $1,472,779
- **R2 Score:** 0.3025
- **Tests:** 10/10 integration tests passing
- **Average response time:** 13.1ms (model inference: ~9ms)

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
| Slow response (~2s) | Use `http://127.0.0.1:8000` instead of `localhost` |

## Known Issues & Limitations
- UI requires the API running in a separate terminal before launch
- Model trained on Walmart dataset — may not generalize to all retail contexts
- No authentication — API is open on localhost only
- R2 score of 0.30 indicates moderate fit — predictions are directional estimates

## Future Improvements
- Deploy API to Render and UI to Streamlit Cloud for public access
- Add more features (promotions, holidays, store location) to improve R2
- Connect to a live database to auto-fill lag values

---
AIE1014 — AI Applied Project Course | Team Dany | Winter 2026