# final version of main.py for Turf Tracker Soil Moisture Tool with LSTM readiness

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import requests
import pandas as pd
import joblib
import os
from models import MoistureLog, IrrigationLog, WeatherHistory, PredictionMeta
from database import Base, engine, SessionLocal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://turf-irrigation-dev1.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "moisture_model.pkl"
LATITUDE = 52.281624
LONGITUDE = -0.943448
VC_API_KEY = "2ELL5E9A47JT5XB74WGXS7PFV"
ML_THRESHOLD = 365

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.get("/logging-progress")
def logging_progress():
    db = SessionLocal()
    count = db.query(MoistureLog).count()
    db.close()
    return {
        "current_samples": count,
        "target_for_ml": ML_THRESHOLD,
        "ml_status": "enabled" if count >= ML_THRESHOLD else "disabled"
    }

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

@app.post("/log-moisture")
def log_moisture(timestamp: str = Body(...), moisture_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = MoistureLog(timestamp=dt, moisture_mm=moisture_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "moisture logged"}

@app.post("/log-irrigation")
def log_irrigation(timestamp: str = Body(...), irrigation_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = IrrigationLog(timestamp=dt, irrigation_mm=irrigation_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "irrigation logged"}

@app.get("/predicted-moisture")
def predicted_moisture():
    try:
        db = SessionLocal()
        moist_entries = db.query(MoistureLog).order_by(MoistureLog.timestamp.asc()).all()
        irrig_entries = db.query(IrrigationLog).all()
        db.close()

        df_moist = pd.DataFrame([{"timestamp": e.timestamp, "moisture_mm": e.moisture_mm} for e in moist_entries])
        df_irrig = pd.DataFrame([{"timestamp": e.timestamp, "irrigation_mm": e.irrigation_mm} for e in irrig_entries])

        if df_moist.empty:
            return []

        df_moist["timestamp"] = pd.to_datetime(df_moist["timestamp"])
        df_moist.set_index("timestamp", inplace=True)
        df_moist.index = df_moist.index.tz_localize(None)

        df_irrig["timestamp"] = pd.to_datetime(df_irrig["timestamp"])
        df_irrig.set_index("timestamp", inplace=True)
        df_irrig.index = df_irrig.index.tz_localize(None)

        latest_log_ts = df_moist.index[-1]
        now = datetime.utcnow()
        forecast_end = now + timedelta(days=5)

        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{LATITUDE},{LONGITUDE}/{latest_log_ts.date()}/{forecast_end.date()}"
            f"?unitGroup=metric&key={VC_API_KEY}&include=hours"
            f"&elements=datetime,temp,humidity,windspeed,solarradiation,precip"
        )

        response = requests.get(url)
        weather_data = response.json()

        df_weather = []
        for day in weather_data.get("days", []):
            for hour in day.get("hours", []):
                raw_ts = f"{day['datetime']}T{hour['datetime'][:5]}"
                timestamp = datetime.strptime(raw_ts, "%Y-%m-%dT%H:%M")
                solar = hour.get("solarradiation", 0) or 0
                et = round(0.408 * solar / 1000, 3)
                df_weather.append({
                    "timestamp": timestamp,
                    "ET_mm_hour": et,
                    "rainfall_mm": hour.get("precip", 0),
                    "temp": hour.get("temp", 0),
                    "humidity": hour.get("humidity", 0),
                    "windspeed": hour.get("windspeed", 0)
                })

        df_weather = pd.DataFrame(df_weather).set_index("timestamp")
        df_weather.index = df_weather.index.tz_localize(None)

        df = df_weather.join(df_irrig, how="left").fillna({"irrigation_mm": 0}).sort_index()

        sample_count = len(df_moist)
        last_pred = df_moist.iloc[-1]["moisture_mm"]
        results = []

        for ts, row in df.iterrows():
            et = row["ET_mm_hour"]
            rain = row["rainfall_mm"]
            irr = row["irrigation_mm"]

            if sample_count < ML_THRESHOLD:
                predicted = last_pred - et + rain + irr
            else:
                # Reserved for future LSTM call
                predicted = last_pred - et + rain + irr  # temp fallback

            predicted = max(min(float(predicted), 100), 0)
            last_pred = predicted

            results.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H"),
                "ET_mm_hour": round(et, 3),
                "rainfall_mm": round(rain, 2),
                "irrigation_mm": round(irr, 2),
                "predicted_moisture_mm": round(predicted, 1)
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/wilt-forecast")
def wilt_forecast(wilt_point: float = 18.0, upper_limit: float = 22.0):
    predictions = predicted_moisture()
    for row in predictions:
        if row["predicted_moisture_mm"] < wilt_point:
            ts = row["timestamp"]
            rec = upper_limit - row["predicted_moisture_mm"]
            return {
                "wilt_point_hit": ts,
                "recommended_irrigation_mm": round(rec, 1),
                "upper_limit": upper_limit,
                "message": f"Apply approx {round(rec, 1)} mm to reach {upper_limit}%"
            }
    return {
        "wilt_point_hit": None,
        "recommended_irrigation_mm": None,
        "upper_limit": upper_limit,
        "message": "No wilt point drop expected in forecast."
    }
