from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
from typing import List, Dict

""" you need to install all the dependecies above in order to run this code."""

app = FastAPI(title="CO₂ Emissions API",
              description="API for CO₂ emissions forecasting and anomaly detection",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "co2_lstm_model.keras")  
FOREST_PATH = os.path.join(BASE_DIR, "isolation_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "co2_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_owid_co2_data.csv")

# Global variables for loaded models
lstm_model = None
iso_forest = None
scaler = None
look_back = 10  

@app.on_event("startup")
async def load_models():
    """Load ML models on startup"""
    global lstm_model, iso_forest, scaler
    
    try:
        # Load LSTM model 
        lstm_model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load Isolation Forest and scaler
        iso_forest = joblib.load(FOREST_PATH)
        scaler = joblib.load(SCALER_PATH)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

@app.get("/", tags=["Root"])
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "models_loaded": all(m is not None for m in [lstm_model, iso_forest, scaler])
    }

@app.get("/historical", response_model=List[Dict[str, float]], tags=["Data"])
async def get_historical_data():
    """
    Get historical CO₂ emissions data for Tunisia
    Returns list of {year: float, co2: float} dictionaries
    """
    try:
        data = pd.read_csv(DATA_PATH)
        tunisian_data = data[data["country"] == "Tunisia"][["year", "co2"]].dropna()
        return tunisian_data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", tags=["Forecasting"])
async def predict(future_years: int = 5):
    """
    Predict future CO₂ emissions
    
    Parameters:
    - future_years: Number of years to predict (default: 5)
    
    Returns:
    - years: List of future years
    - predictions: List of predicted CO₂ values
    """
    if not lstm_model or not scaler:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if future_years < 1 or future_years > 20:
        raise HTTPException(status_code=400, detail="future_years must be between 1 and 20")

    try:
        data = pd.read_csv(DATA_PATH)
        tunisian_data = data[data["country"] == "Tunisia"][["year", "co2"]].dropna()
        co2_values = tunisian_data["co2"].values.reshape(-1, 1)
        
        # Scale data
        co2_scaled = scaler.transform(co2_values)
        
        # Initialize with last look_back values
        last_sequence = co2_scaled[-look_back:].reshape(1, look_back, 1)
        
        # Generate predictions
        predictions = []
        for _ in range(future_years):
            next_pred = lstm_model.predict(last_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Generate future years
        last_year = tunisian_data["year"].iloc[-1]
        future_years_list = [int(last_year) + i + 1 for i in range(future_years)]
        
        return {
            "years": future_years_list,
            "predictions": predictions.flatten().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anomalies", tags=["Anomaly Detection"])
async def detect_anomalies():
    """
    Detect anomalous CO₂ emission years
    
    Returns:
    - anomalies: List of anomalous years with CO₂ values
    - all_data: Complete historical dataset
    """
    if not iso_forest:
        raise HTTPException(status_code=503, detail="Anomaly model not loaded")

    try:
        data = pd.read_csv(DATA_PATH)
        tunisian_data = data[data["country"] == "Tunisia"][["year", "co2"]].dropna()
        
        # Detect anomalies
        X = tunisian_data[["co2"]].values
        anomalies = iso_forest.predict(X)
        tunisian_data["anomaly"] = anomalies
        
        return {
            "anomalies": tunisian_data[tunisian_data["anomaly"] == -1]
                         .drop(columns=["anomaly"])
                         .to_dict(orient="records"),
            "all_data": tunisian_data.drop(columns=["anomaly"])
                        .to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)