from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import requests
import pandas as pd
import joblib
import xgboost as xgb
import os

from models import MoistureLog, IrrigationLog
from database import Base, engine, SessionLocal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://turf-irrigation-dev1.netlify.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "moisture_model.pkl"
LATITUDE = 52.281624
LONGITUDE = -0.943448
ELEVATION = 95
VC_API_KEY = os.getenv("VC_API_KEY", "2ELL5E9A47JT5XB74WGXS7PFV")

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.get("/wilt-forecast")
def get_wilt_forecast(wilt_point: float = 18.0, upper_limit: float = 22.0):
    db = SessionLocal()
    entries = db.query(MoistureLog).order_by(MoistureLog.timestamp).all()
    db.close()

    if not entries:
        return {"message": "No moisture data available yet."}

    model_file_exists = os.path.exists(MODEL_FILE)
    if not model_file_exists:
        return {"message": "Model not trained yet."}

    model = joblib.load(MODEL_FILE)

    now = datetime.utcnow()
    start_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=5)).strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LATITUDE},{LONGITUDE}/{start_date}/{end_date}?unitGroup=metric&key={VC_API_KEY}&include=hours&elements=datetime,temp,humidity,windspeed,solarradiation,precip"
    response = requests.get(url)
    data = response.json()

    df_weather = []
    for day in data.get("days", []):
        for hour in day.get("hours", []):
            raw_ts = f"{day['datetime']}T{hour['datetime'][:5]}"
            solar_radiation = hour.get("solarradiation", 0) or 0
            et = round(0.408 * solar_radiation / 1000, 3)
            df_weather.append({
                "timestamp": raw_ts,
                "ET_mm_hour": et,
                "rainfall_mm": hour.get("precip", 0) or 0
            })

    df_weather = pd.DataFrame(df_weather)
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], format="%Y-%m-%dT%H:%M", errors="coerce")
    df_weather.dropna(subset=["timestamp"], inplace=True)
    df_weather.set_index("timestamp", inplace=True)

    db = SessionLocal()
    df_irrig = pd.DataFrame([
        {"timestamp": e.timestamp, "irrigation_mm": e.irrigation_mm}
        for e in db.query(IrrigationLog).all()
    ])
    df_moist = pd.DataFrame([
        {"timestamp": e.timestamp, "moisture_mm": e.moisture_mm}
        for e in entries
    ])
    db.close()

    df_irrig.set_index("timestamp", inplace=True)
    df_moist.set_index("timestamp", inplace=True)

    df = df_weather.join(df_irrig, how="left").fillna({"irrigation_mm": 0})
    df = df.sort_index()

    last_pred = df_moist.iloc[-1]["moisture_mm"]
    sample_count = len(df_moist)

    for ts, row in df.iterrows():
        features = pd.DataFrame([{
            "prev_moisture": last_pred,
            "irrigation_mm": row.get("irrigation_mm", 0),
            "hour": ts.hour,
            "dayofyear": ts.dayofyear
        }])

        model_pred = model.predict(features)[0]
        basic_estimate = last_pred - row.get("ET_mm_hour", 0) + row.get("rainfall_mm", 0) + row.get("irrigation_mm", 0)

        alpha = min(sample_count / 100, 1.0)
        predicted_moisture = (alpha * model_pred) + ((1 - alpha) * basic_estimate)
        predicted_moisture = max(min(predicted_moisture, 100), 0)

        if predicted_moisture < wilt_point:
            rec_irrig = max(0.0, upper_limit - predicted_moisture)
            return {
                "wilt_point_hit": ts.strftime("%Y-%m-%dT%H"),
                "recommended_irrigation_mm": round(rec_irrig, 1),
                "upper_limit": upper_limit,
                "message": f"Apply approx {round(rec_irrig, 1)} mm to reach {upper_limit}%"
            }

        last_pred = predicted_moisture

    return {"message": "No wilt point drop expected in forecast."}
