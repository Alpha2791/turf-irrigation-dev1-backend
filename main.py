from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import requests
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
VC_API_KEY = os.getenv("2ELL5E9A47JT5XB74WGXS7PFV")

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.get("/moisture-log")
def get_moisture_log():
    db = SessionLocal()
    entries = db.query(MoistureLog).order_by(MoistureLog.timestamp.desc()).all()
    db.close()
    return [{"timestamp": e.timestamp.isoformat(), "moisture_mm": e.moisture_mm} for e in entries]

@app.get("/irrigation-log")
def get_irrigation_log():
    db = SessionLocal()
    entries = db.query(IrrigationLog).order_by(IrrigationLog.timestamp.desc()).all()
    db.close()
    return [{"timestamp": e.timestamp.isoformat(), "irrigation_mm": e.irrigation_mm} for e in entries]

@app.get("/predicted-moisture")
def get_predicted_moisture():
    if not os.path.exists(MODEL_FILE):
        return []

    model = joblib.load(MODEL_FILE)

    now = datetime.utcnow()
    start_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=5)).strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LATITUDE},{LONGITUDE}/{start_date}/{end_date}?unitGroup=metric&key={VC_API_KEY}&include=hours&elements=datetime,temp,humidity,windspeed,solarradiation,precip"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    weather_data = []
    for day in data.get("days", []):
        for hour in day.get("hours", []):
            raw_ts = f"{day['datetime']}T{hour['datetime'][:5]}"
            solar_radiation = hour.get("solarradiation", 0) or 0
            et = round(0.408 * solar_radiation / 1000, 3)
            weather_data.append({
                "timestamp": raw_ts,
                "ET_mm_hour": et,
                "rainfall_mm": hour.get("precip", 0) or 0
            })

    df_weather = pd.DataFrame(weather_data)
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], format="%Y-%m-%dT%H:%M", errors="coerce")
    df_weather.dropna(subset=["timestamp"], inplace=True)
    df_weather.set_index("timestamp", inplace=True)

    print("📊 Weather forecast loaded:", len(df_weather))
    print(df_weather.head(3))

    db = SessionLocal()
    moisture_entries = db.query(MoistureLog).order_by(MoistureLog.timestamp).all()
    irrigation_entries = db.query(IrrigationLog).order_by(IrrigationLog.timestamp).all()
    db.close()

    print("🌱 Moisture entries:", len(moisture_entries))
    print("💧 Irrigation entries:", len(irrigation_entries))

    df_moist = pd.DataFrame([{"timestamp": e.timestamp, "moisture_mm": e.moisture_mm} for e in moisture_entries])
    df_irrig = pd.DataFrame([{"timestamp": e.timestamp, "irrigation_mm": e.irrigation_mm} for e in irrigation_entries])

    if not df_moist.empty:
        df_moist.set_index("timestamp", inplace=True)
    else:
        df_moist = pd.DataFrame(columns=["moisture_mm"])
        df_moist.index.name = "timestamp"

    if not df_irrig.empty:
        df_irrig.set_index("timestamp", inplace=True)
    else:
        df_irrig = pd.DataFrame(columns=["irrigation_mm"])
        df_irrig.index.name = "timestamp"

    print("🧪 df_moist sample:")
    print(df_moist.head(3))
    print("🧪 df_irrig sample:")
    print(df_irrig.head(3))

    df = df_weather.join(df_irrig, how="left").fillna({"irrigation_mm": 0})
    df = df.sort_index()

    results = []
    last_pred = df_moist.iloc[-1]["moisture_mm"] if not df_moist.empty else 25.0
    sample_count = len(df_moist)

    print("🔁 Starting prediction loop")

    for ts, row in df.iterrows():
        hour = ts.hour
        dayofyear = ts.dayofyear
        irrigation_mm = row["irrigation_mm"]
        rainfall_mm = row.get("rainfall_mm", 0)
        et_mm = row.get("ET_mm_hour", 0)

        features = pd.DataFrame([{
            "prev_moisture": last_pred,
            "irrigation_mm": irrigation_mm,
            "hour": hour,
            "dayofyear": dayofyear
        }])

        model_pred = model.predict(features)[0]
        basic_estimate = last_pred - et_mm + rainfall_mm + irrigation_mm
        alpha = min(sample_count / 100, 1.0)
        predicted_moisture = (alpha * model_pred) + ((1 - alpha) * basic_estimate)
        predicted_moisture = max(min(predicted_moisture, 100), 0)

        results.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H"),
            "ET_mm_hour": et_mm,
            "rainfall_mm": rainfall_mm,
            "irrigation_mm": irrigation_mm,
            "predicted_moisture_mm": round(float(predicted_moisture), 1)
        })

        last_pred = predicted_moisture

    print("✅ Predictions generated:", len(results))

    return results

@app.get("/wilt-forecast")
def get_wilt_forecast(wilt_point: float = 18.0, upper_limit: float = 22.0):
    predictions = get_predicted_moisture()

    if not predictions:
        return {"message": "Forecast unavailable. Please check weather API or model."}

    for row in predictions:
        moisture = row["predicted_moisture_mm"]
        if moisture < wilt_point:
            ts = row["timestamp"]
            rec_irrig = upper_limit - moisture
            return {
                "wilt_point_hit": ts,
                "recommended_irrigation_mm": round(rec_irrig, 1),
                "upper_limit": upper_limit,
                "message": f"Apply approx {round(rec_irrig, 1)} mm to reach {upper_limit}%"
            }

    return {"message": "No wilt point drop expected in forecast."}
