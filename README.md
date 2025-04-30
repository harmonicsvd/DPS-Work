# Accident Prediction API

A Flask API that predicts accident statistics using a Random Forest model trained on historical data.

## Features

- Predicts accident statistics based on year and month
- Uses Random Forest Regressor 
- Features: Year, Month
- Target: Accident value

## API Endpoints

- `POST /predict` - Get predictions  
  **Input:** `{"year": 2023, "month": 6}`  
  **Output:** `{"prediction": 41}` 

- `GET /model-status` - Check model status  
- `GET /versions` - View package versions

## Model Performance
- R² Score: 0.61
- MSE: 0.07
- Feature Importance:
  - Month: 79.5%
  - Year: 20.5%

## Deployment
Deployed on PythonAnywhere at:  
https://VD56.pythonanywhere.com
