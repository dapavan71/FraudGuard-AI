from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import numpy as np

# Load models and artifacts at startup
try:
    xgb_model = joblib.load('models/xgb_model.joblib')
    iso_model = joblib.load('models/iso_model.joblib')
    scaler_time = joblib.load('models/scaler_time.joblib')
    scaler_amount = joblib.load('models/scaler_amount.joblib')
    meta = joblib.load('models/meta.joblib')
    features = meta['features']
    tuned_threshold = meta['threshold']
except Exception as e:
    print(f"Error loading models: {e}")
    xgb_model, iso_model, scaler_time, scaler_amount, features = None, None, None, None, None
    tuned_threshold = 0.5

app = FastAPI(title="Credit Card Fraud Detection API", 
              description="Real-time API to detect fraudulent credit card transactions")

# Dynamically create Pydantic model for input validation
fields = {'Time': (float, 0.0), 'Amount': (float, 0.0)}
for i in range(1, 29):
    fields[f'V{i}'] = (float, 0.0)

TransactionInput = create_model('TransactionInput', **fields)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    anomaly_score: float
    is_anomaly: bool

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")
    
    # Convert input to DataFrame
    data = transaction.dict()
    df = pd.DataFrame([data])
    
    # Preprocess
    df['Scaled_Time'] = scaler_time.transform(df[['Time']])
    df['Scaled_Amount'] = scaler_amount.transform(df[['Amount']])
    
    # Ensure correct feature order
    # (The training script dropped Time and Amount and added Scaled_Time and Scaled_Amount at the end, 
    # but let's strictly order them by the 'features' list from meta.joblib)
    for f in features:
        if f not in df.columns:
             # Just in case
             df[f] = 0.0
             
    X = df[features]
    
    # XGBoost Predict
    prob = xgb_model.predict_proba(X)[0, 1]
    prediction = "Fraud" if prob >= tuned_threshold else "Normal"
    # Confidence 
    confidence = float(prob if prediction == "Fraud" else 1 - prob)
    
    # Isolation Forest Anomaly Detection
    # Lower score = more anomalous
    iso_score = iso_model.score_samples(X)[0]
    iso_pred = iso_model.predict(X)[0] # -1 for anomaly, 1 for normal
    is_anomaly = bool(iso_pred == -1)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        probability=float(prob),
        anomaly_score=float(iso_score),
        is_anomaly=is_anomaly
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API. Use the /predict endpoint."}
