# Weekly Sales Prediction App
AIE1014 — AI Applied Project Course

## Team
- Dany (MLOps Engineer)

## Project Description
This application predicts weekly sales figures using a trained machine learning model
based on historical lag features. Users can enter lag values through a web interface
and receive sales predictions instantly.

## Project Structure
```
assignment2/
├── api/
│   ├── main.py              # FastAPI application
│   ├── test_api.py          # API unit tests
│   └── __init__.py
├── code/
│   ├── data_pipeline.ipynb  # Data preprocessing
│   └── train_model.ipynb    # Model training
├── models/
│   └── initial_model.pkl    # Trained ML model
├── tests/
│   └── test_integration.py  # Integration tests
├── app_ui.py                # Streamlit user interface
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Quick Start

### Prerequisites
- Python 3.8+

### Installation
```bash
git clone [your-repo-url]
cd assignment2
python -m pip install -r requirements.txt
```

### Running the Application

**Terminal 1 — Start the API:**
```bash
python -m uvicorn api.main:app --reload
```
API runs at: http://localhost:8000

**Terminal 2 — Start the UI:**
```bash
streamlit run app_ui.py
```
UI opens at: http://localhost:8501

### Running Tests
```bash
python tests/test_integration.py
```

## API Endpoints

| Endpoint  | Method | Description          |
|-----------|--------|----------------------|
| /health   | GET    | Health check         |
| /predict  | POST   | Make a prediction    |
| /info     | GET    | Model information    |

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
  "prediction": 102.5,
  "status": "success"
}
```

## Error Codes
- 200: Success
- 422: Missing or invalid input fields
- 500: Server/model error

## Known Issues
- UI requires the API to be running in a separate terminal before launch

## Milestone 4
Due: Wednesday, March 25, 2026 at 12:30 PM
Submission: Moodle → TeamDany_Milestone4