# Weekly Sales Prediction App

## What This Project Does
This application predicts weekly sales figures for retail business managers using a RandomForestRegressor model trained on historical lag features. Users enter three lag values (previous week, 2 weeks ago, and 1 year ago) through a web interface and receive instant sales forecasts to support inventory and staffing decisions.

## Team / Author
- Dany Deugoue, 6024

## Prerequisites
- Python 3.8 or higher
- pip

## Installation
```bash
git clone https://github.com/Dany281022/AppliedProject.git
cd Assignment03_TeamDany
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
Assignment03_TeamDany/
├── api/
│   ├── app.py               ← FastAPI server
│   └── model.pkl            ← Trained model
├── ui/
│   └── app_ui.py            ← Streamlit interface
├── tests/
│   └── test_integration.py
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description       |
|----------|--------|-------------------|
| /health  | GET    | Health check      |
| /predict | POST   | Make a prediction |
| /info    | GET    | Model information |

## Model Information
- **Algorithm:** RandomForestRegressor
- **Target:** Weekly sales figures (float)
- **Key features:** lag_1 (previous week), lag_2 (2 weeks ago), lag_52 (1 year ago)
- **Performance:** Best CV score from Assignment 02 training pipeline

## Known Issues & Limitations
- UI requires the API to be running in a separate terminal before launch
- Model is trained on only 3 lag features and cannot capture promotions, seasonality, or external factors affecting sales

## Next Steps
- Deploy API to Render and UI to Streamlit Cloud for public stakeholder access
- Add more features (promotions, holidays, store location) to improve model accuracy
- Add prediction history chart to track forecasts over time