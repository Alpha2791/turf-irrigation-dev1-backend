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

def calculate_et_fao56(solar_w_m2, temp, humidity, wind):
    """
    FAO-56 Penman-Monteith ET₀ hourly calculation, with correct unit conversions.
    Inputs:
        solar_w_m2 - Solar radiation in W/m² (from Visual Crossing)
        temp - Air temperature (°C)
        humidity - Relative humidity (%)
        wind - Wind speed (m/s)
    Returns:
        ET₀ in mm/hour
    """
    # Convert solar from W/m² to MJ/m²/hour
    solar_mj_m2_hr = (solar_w_m2 * 3600) / 1_000_000
    Rn = 0.408 * solar_mj_m2_hr  # Net radiation (MJ/m²/hour)

    T = temp
    u2 = wind
    es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
    ea = es * (humidity / 100)
    delta = 4098 * es / ((T + 237.3) ** 2)
    P = 101.3 * (((293 - 0.0065 * ELEVATION) / 293) ** 5.26)
    gamma = 0.000665 * P

    eto = ((0.408 * delta * Rn) + (gamma * 900 / (T + 273) * u2 * (es - ea))) / (
        delta + gamma * (1 + 0.34 * u2)
    )
    return max(0.0, eto)  # Ensure ET cannot be negative

@app.get("/predicted-moisture")
def predicted_moisture():
    db = SessionLocal()

    # Load moisture and irrigation logs
    moisture_logs = pd.read_sql(db.query(MoistureLog).statement, db.connect(), parse_dates=["timestamp"])
    irrigation_logs = pd.read_sql(db.query(IrrigationLog).statement, db.connect(), parse_dates=["timestamp"])

    if moisture_logs.empty:
        raise HTTPException(status_code=404, detail="No moisture data available")

    # Sort and get last entry
    moisture_logs.sort_values("timestamp", inplace=True)
    latest_log = moisture_logs.iloc[-1]
    latest_log_ts = latest_log["timestamp"]
    last_pred = latest_log["moisture_mm"]

    print("[DEBUG] Last few moisture logs:\n", moisture_logs.tail())
    print("[DEBUG] Starting moisture for forecast:", last_pred)

    # Forecast period: next 5 days
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    forecast_end = now + timedelta(days=5)
    forecast_df = pd.read_sql(
        db.query(WeatherHistory)
          .filter(WeatherHistory.timestamp >= latest_log_ts)
          .filter(WeatherHistory.timestamp <= forecast_end)
          .statement,
        db.connect(),
        parse_dates=["timestamp"]
    )

    forecast_df.sort_values("timestamp", inplace=True)
    print("[DEBUG] Forecast dataframe shape:", forecast_df.shape)

    # Build hourly predictions
    results = [{
        "timestamp": (latest_log_ts - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        "ET_mm_hour": 0.0,
        "rainfall_mm": 0.0,
        "irrigation_mm": 0.0,
        "predicted_moisture_mm": float(round(last_pred, 1))
    }]

    for i, row in forecast_df.iterrows():
        t = row["timestamp"]
        et = float(row.get("ET_mm_hour", 0.0))
        rain = float(row.get("rainfall_mm", 0.0))

        irr = 0.0
        if not irrigation_logs.empty:
            irr_logs = irrigation_logs[irrigation_logs["timestamp"] == t]
            if not irr_logs.empty:
                irr = float(irr_logs["irrigation_mm"].sum())

        last_pred = max(0, last_pred - et + rain + irr)

        results.append({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"),
            "ET_mm_hour": et,
            "rainfall_mm": rain,
            "irrigation_mm": irr,
            "predicted_moisture_mm": float(round(last_pred, 1))
        })

    print("[DEBUG] First 3 result rows:")
    print(results[:3])

    return results

    except Exception as e:
        print(f"[ERROR] {str(e)}")
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
