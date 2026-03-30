# retrain_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# Rename columns
FEATURES = ['lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_12', 'lag_26', 'lag_52',
            'ma_4', 'ma_12', 'std_4', 'weekofyear', 'month', 'year']

X_train.columns = FEATURES
X_test.columns = FEATURES

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print(f'RMSE: {rmse:,.2f}')
print(f'MAE: {mae:,.2f}')
print(f'R2 Score: {r2:.4f}')

# Save model
joblib.dump(model, 'api/model.pkl')
print('Model saved in api/model.pkl')
print('Features:', model.feature_names_in_.tolist())