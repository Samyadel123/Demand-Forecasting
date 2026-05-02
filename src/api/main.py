import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd

app = FastAPI(title="Resynor Inference API")

# 1. Load the model into memory exactly ONCE when the server starts
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)

# Use the Production alias we set during training
model_registry_name = "Resynor_Whale_Forecaster"
model_uri = f"models:/{model_registry_name}@Production"

# Use generic mlflow.pyfunc to support any flavor logged (XGBoost, Sklearn, etc.)
whale_model = mlflow.pyfunc.load_model(model_uri)


# 2. Use Pydantic to strictly validate incoming requests
class RawDemandRequest(BaseModel):
    warehouse: str
    product_category: str
    target_date: str  # e.g., "2026-11-23"


class DemandRequest(BaseModel):
    # Temporal indicators
    year: int
    month: int
    day_of_week: int
    week_of_year: int

    # Lag and Rolling features
    demand_T_minus_2: float
    rolling_7d_stddev: float
    rolling_7d_mean: float

    # Binary flags
    is_q4: int
    is_monday: int


# --- Mock Database Service ---
def fetch_recent_history(warehouse: str, category: str, date: str) -> list[float]:
    """
    In production, this queries your operational database (Redis/Postgres)
    for the last N days of actual demand leading up to the target date.
    """
    # Mocking a fast DB lookup returning 7 days of historical demand
    return [12000, 15000, 0, 0, 32000, 14000, 11000]


@app.post("/predict")
def predict_demand(request: RawDemandRequest):
    try:
        # Step 1: Parse the client's raw date to engineer Temporal Features
        date_obj = pd.to_datetime(request.target_date)
        
        year = int(date_obj.year)
        month = int(date_obj.month)
        # pd.Timestamp.dayofweek: 0=Mon, 6=Sun
        # Spark F.dayofweek: 1=Sun, 2=Mon, 7=Sat
        day_of_week = int((date_obj.dayofweek + 1) % 7 + 1)
        week_of_year = int(date_obj.isocalendar()[1])
        
        is_q4 = 1 if date_obj.month in [10, 11, 12] else 0
        is_monday = 1 if date_obj.dayofweek == 0 else 0

        # Step 2: Fetch the state from your fast database
        history = fetch_recent_history(
            request.warehouse, request.product_category, request.target_date
        )

        if len(history) < 7:
            raise ValueError(
                "Insufficient historical data to calculate rolling features."
            )

        # Step 3: Engineer the Lag/Rolling features on the backend
        rolling_7d_series = pd.Series(history)
        rolling_7d_stddev = float(rolling_7d_series.std())
        rolling_7d_mean = float(rolling_7d_series.mean())
        demand_T_minus_2 = float(history[-3])  # The value from 2 days prior to target

        # Step 4: Assemble the exact feature schema XGBoost expects
        # MUST match the order used during training:
        # year, month, day_of_week, week_of_year, demand_T_minus_2, rolling_7d_stddev, rolling_7d_mean, is_q4, is_monday
        feature_vector = {
            "year": year,
            "month": month,
            "day_of_week": day_of_week,
            "week_of_year": week_of_year,
            "demand_T_minus_2": demand_T_minus_2,
            "rolling_7d_stddev": rolling_7d_stddev,
            "rolling_7d_mean": rolling_7d_mean,
            "is_q4": is_q4,
            "is_monday": is_monday,
        }

        # Step 5: Run Inference
        # We ensure the columns are in the EXACT order expected by the model
        cols_order = [
            "year", "month", "day_of_week", "week_of_year", 
            "demand_T_minus_2", "rolling_7d_stddev", "rolling_7d_mean", 
            "is_q4", "is_monday"
        ]
        input_data = pd.DataFrame([feature_vector])[cols_order]
        
        prediction = whale_model.predict(input_data)

        # Step 6: Return the clean result to the client
        return {
            "warehouse": request.warehouse,
            "target_date": request.target_date,
            "predicted_demand": float(prediction[0]),
            "model_version": model_uri,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
