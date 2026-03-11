Milestone 3 — Weekly Sales Forecasting (Walmart Dataset)
This project implements time series forecasting for weekly sales data using Naive baseline, SARIMA, and Random Forest models.
 Project Structure
Team_Milestone3/
├── code/
│   ├── data_pipeline.ipynb      Data loading, cleaning, feature engineering, train/test split
│   ├── train_model.ipynb        Model training and evaluation (Baseline, SARIMA, Random Forest)
│   └── requirements.txt         Python dependencies
├── data/
│   ├── raw/
│   │   └── Walmart.csv          Original dataset (source: Kaggle)
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   └── initial_model.pkl        Saved Random Forest model
├── report/
│   └── Team_Milestone3_Report.pdf
└── README.md
How to Run

 1. Install dependencies
run on the terminal poweshell
pip install -r code/requirements.txt


 2. Run the data pipeline
Open and run all cells in «code/data_pipeline.ipynb».
This will:
- Load «data/raw/Walmart.csv»
- Clean and preprocess the data
- Engineer features (lags, rolling stats, calendar features)
- Split the data chronologically (cutoff: January 1, 2012)
- Save processed files to «data/processed/»

 3. Train the model
Open and run all cells in «code/train_model.ipynb».
This will:
- Load the processed data from «data/processed/»
- Train the Naive baseline, SARIMA, and Random Forest models
- Evaluate and compare performance (RMSE, MAE)
- Save the trained model to «models/initial_model.pkl»

 Dataset
- Source: Walmart Store Sales — [Kaggle](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
- Size: 6,435 rows × 8 columns
- Period: February 2010 – October 2012
- Target: «Weekly_Sales»

Key Results:
Model	RMSE	MAE
Naive Baseline	2,481,006.64	1,859,967.56
SARIMA (m=12	5,939,294.72	5,353,531.52
Random Forest (tuned)	1,648,722.36	951,806.03

Random Forest improves RMSE by 33.5% and MAE by 48.8% over the baseline.

 Requirements
Python 3.9+, see «code/requirements.txt» for full list.
