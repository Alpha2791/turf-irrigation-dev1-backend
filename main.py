from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
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

LATITUDE = 52.281624
LONGITUDE = -0.943448
ELEVATION = 95
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

def calculate_et_fao56(solar, temp, humidity, wind):
    Rn = 0.408 * solar
    T = temp
    u2 = wind
    es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
    ea = es * (humidity / 100)
    delta = 4098 * es / ((T + 237.3) ** 2)
    P = 101.3 * (((293 - 0.0065 * ELEVATION) / 293) ** 5.26)
    gamma = 0.000665 * P
    return ((0.408 * delta * Rn) + (gamma * 900 / (T + 273) * u2 * (es - ea))) / (delta + gamma * (1 + 0.34 * u2))

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

        df_moist["timestamp"] = pd.to_datetime(df_moist["timestamp"]).dt.tz_localize(None)
        df_moist.set_index("timestamp", inplace=True)

        df_irrig["timestamp"] = pd.to_datetime(df_irrig["timestamp"]).dt.tz_localize(None)
        df_irrig.set_index("timestamp", inplace=True)

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
                et = calculate_et_fao56(solar, hour.get("temp", 0), hour.get("humidity", 0), hour.get("windspeed", 0))
                df_weather.append({
                    "timestamp": timestamp,
                    "ET_mm_hour": round(et, 3),
                    "rainfall_mm": hour.get("precip", 0) or 0
                })

        df_weather = pd.DataFrame(df_weather)
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"]).dt.tz_localize(None)
        df_weather.set_index("timestamp", inplace=True)

        # Join weather + irrigation, and fill missing rainfall/irrigation with 0
        df = df_weather.join(df_irrig, how="left").fillna({"irrigation_mm": 0, "rainfall_mm": 0})

        # Keep only forecast data at or after the last logged moisture
        df = df[df.index >= latest_log_ts]
        df = df.sort_index()

        results = []
        last_pred = df_moist.iloc[-1]["moisture_mm"]

        # Add the real logged moisture as the start of the forecast
        results.append({
            "timestamp": latest_log_ts.strftime("%Y-%m-%dT%H"),
            "ET_mm_hour": 0,
            "rainfall_mm": 0,
            "irrigation_mm": 0,
            "predicted_moisture_mm": round(last_pred, 1)
        })

        for ts, row in df.iterrows():
            et = row["ET_mm_hour"]
            rain = row["rainfall_mm"]
            irr = row["irrigation_mm"]

            predicted = last_pred - et + rain + irr
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
