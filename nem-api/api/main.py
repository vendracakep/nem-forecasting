from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import xgboost as xgb
import pandas as pd
import numpy as np
import json

app = FastAPI(title="NEM Forecast API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
feature_names = []
buffer_df = None

# Pydantic models
class ForecastRequest(BaseModel):
    horizon_hours: int = 24
    confidence_level: float = 0.9

class ForecastPoint(BaseModel):
    timestamp: str
    forecast: float
    lower_bound: float
    upper_bound: float
    uncertainty: float

class ForecastResponse(BaseModel):
    forecasts: List[ForecastPoint]
    current_price: float
    last_update: str
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_points: int
    last_timestamp: Optional[str]

@app.on_event("startup")
async def load_resources():
    """Load model and data on startup"""
    global model, feature_names, buffer_df
    
    try:
        # Load model
        model = xgb.XGBRegressor()
        model.load_model("xgb_electricity_model.json")
        
        # Load feature names
        with open("feature_names.json", "r") as f:
            feature_names = json.load(f)
        
        # Load buffer data
        buffer_df = pd.read_csv("last_data_buffer.csv")
        buffer_df['ds'] = pd.to_datetime(buffer_df['ds'])
        buffer_df = buffer_df.sort_values('ds').reset_index(drop=True)
        
        print("Resources loaded successfully")
        
    except Exception as e:
        print(f"Error loading resources: {e}")
        raise

def create_features(df_input, fill_method='forward'):
    """Feature engineering function"""
    df = df_input.copy()
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}h'] = df['y'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f'rolling_mean_{window}h'] = df['y'].shift(1).rolling(window=window).mean()
    
    df['rolling_std_24h'] = df['y'].shift(1).rolling(window=24).std()
    df['rolling_min_24h'] = df['y'].shift(1).rolling(window=24).min()
    df['rolling_max_24h'] = df['y'].shift(1).rolling(window=24).max()
    
    # Time features
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_month'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Interaction features
    df['hour_dow'] = df['hour'] * df['day_of_week']
    
    # Handle NaN
    if fill_method == 'forward':
        df = df.fillna(method='ffill')
    
    return df

def predict_with_confidence(horizon_hours: int, confidence_level: float):
    """Generate forecast with confidence intervals"""
    global model, feature_names, buffer_df
    
    # Prepare future dates
    last_timestamp = buffer_df['ds'].iloc[-1]
    future_dates = [last_timestamp + timedelta(hours=i+1) for i in range(horizon_hours)]
    
    # Extended dataframe
    extended_df = buffer_df.copy()
    future_template = pd.DataFrame({'ds': future_dates, 'y': np.nan})
    extended_df = pd.concat([extended_df, future_template], ignore_index=True)
    
    predictions = []
    uncertainties = []
    
    # Iterative prediction
    for i in range(len(buffer_df), len(extended_df)):
        temp_df = create_features(extended_df.iloc[:i+1], fill_method='forward')
        current_row = temp_df.iloc[[-1]][feature_names]
        
        pred_val = model.predict(current_row)[0]
        predictions.append(float(pred_val))
        
        # Uncertainty estimation
        hours_ahead = i - len(buffer_df) + 1
        base_uncertainty = buffer_df['y'].std() * 0.1
        uncertainty = base_uncertainty * np.sqrt(hours_ahead)
        uncertainties.append(float(uncertainty))
        
        extended_df.loc[i, 'y'] = pred_val
    
    # Calculate confidence intervals
    z_score = 1.645 if confidence_level == 0.9 else 1.96
    
    forecast_points = []
    for i, (date, pred, unc) in enumerate(zip(future_dates, predictions, uncertainties)):
        forecast_points.append(ForecastPoint(
            timestamp=date.isoformat(),
            forecast=pred,
            lower_bound=pred - z_score * unc,
            upper_bound=pred + z_score * unc,
            uncertainty=unc
        ))
    
    return forecast_points

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        data_points=len(buffer_df) if buffer_df is not None else 0,
        last_timestamp=buffer_df['ds'].iloc[-1].isoformat() if buffer_df is not None else None
    )

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate electricity price forecast"""
    
    if model is None or buffer_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not (12 <= request.horizon_hours <= 72):
        raise HTTPException(status_code=400, detail="Horizon must be between 12 and 72 hours")
    
    if not (0.8 <= request.confidence_level <= 0.95):
        raise HTTPException(status_code=400, detail="Confidence level must be between 0.8 and 0.95")
    
    try:
        forecasts = predict_with_confidence(request.horizon_hours, request.confidence_level)
        
        return ForecastResponse(
            forecasts=forecasts,
            current_price=float(buffer_df['y'].iloc[-1]),
            last_update=buffer_df['ds'].iloc[-1].isoformat(),
            metadata={
                "horizon_hours": request.horizon_hours,
                "confidence_level": request.confidence_level,
                "data_points": len(buffer_df),
                "model_features": len(feature_names)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/current")
async def get_current_price():
    """Get current spot price"""
    if buffer_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "price": float(buffer_df['y'].iloc[-1]),
        "timestamp": buffer_df['ds'].iloc[-1].isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)