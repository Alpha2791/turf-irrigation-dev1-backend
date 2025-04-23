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
VC_API_KEY = "2ELL5E9A47JT5XB74WGXS7PFV"

class MoistureEntry(BaseModel):
    timestamp: str
    moisture_mm: float

class IrrigationEntry(BaseModel):
    timestamp: str
    irrigation_mm: float

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.post("/log-moisture")
def log_moisture(timestamp: str = Body(...), moisture_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = MoistureLog(timestamp=dt, moisture_mm=moisture_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "moisture logged"}

@app.get("/moisture-log")
def get_moisture_log():
    db = SessionLocal()
    entries = db.query(MoistureLog).order_by(MoistureLog.timestamp.desc()).all()
    db.close()
    return [{"timestamp": e.timestamp.isoformat(), "moisture_mm": e.moisture_mm} for e in entries]

@app.post("/log-irrigation")
def log_irrigation(timestamp: str = Body(...), irrigation_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = IrrigationLog(timestamp=dt, irrigation_mm=irrigation_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "irrigation logged"}

@app.get("/irrigation-log")
def get_irrigation_log():
    db = SessionLocal()
    entries = db.query(IrrigationLog).order_by(IrrigationLog.timestamp.desc()).all()
    db.close()
    return [{"timestamp": e.timestamp.isoformat(), "irrigation_mm": e.irrigation_mm} for e in entries]

def load_moisture_dataframe():
    db = SessionLocal()
    entries = db.query(MoistureLog).all()
    db.close()
    return pd.DataFrame([{
        "timestamp": e.timestamp,
        "moisture_mm": e.moisture_mm
    } for e in entries])

def load_irrigation_dataframe():
    db = SessionLocal()
    entries = db.query(IrrigationLog).all()
    db.close()
    return pd.DataFrame([{
        "timestamp": e.timestamp,
        "irrigation_mm": e.irrigation_mm
    } for e in entries])

@app.get("/train-model")
def train_model():
    df_moist = load_moisture_dataframe()
    df_irrig = load_irrigation_dataframe()

    df = pd.merge(df_moist.sort_values("timestamp"), df_irrig, how="left", on="timestamp").fillna(0)
    df["prev_moisture"] = df["moisture_mm"].shift(1)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df.dropna(inplace=True)

    if len(df) < 2:
        return {"message": "Not enough data to train model"}

    X = df[["prev_moisture", "irrigation_mm", "hour", "dayofyear"]]
    y = df["moisture_mm"]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    return {"message": f"Model trained with {len(X)} samples and saved to {MODEL_FILE}"}

@app.get("/predicted-moisture")
def get_predicted_moisture():
    print("⚙️ Running /predicted-moisture")

    if not os.path.exists(MODEL_FILE):
        return []

    model = joblib.load(MODEL_FILE)

    now = datetime.utcnow()
    start_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=5)).strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LATITUDE},{LONGITUDE}/{start_date}/{end_date}?unitGroup=metric&key={VC_API_KEY}&include=hours&elements=datetime,temp,humidity,windspeed,solarradiation,precip"
    response = requests.get(url)
    data = response.json()

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

    df_irrig = load_irrigation_dataframe().set_index("timestamp")
    df_moist = load_moisture_dataframe().set_index("timestamp")

    df = df_weather.join(df_irrig, how="left").fillna({"irrigation_mm": 0})
    df = df.sort_index()
    print("📅 Forecast dataframe shape:", df.shape)

    results = []
    last_pred = df_moist.iloc[-1]["moisture_mm"] if not df_moist.empty else 25.0
    sample_count = len(df_moist)

    print("🧪 Starting moisture prediction loop")
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

    print(f"📊 Returning {len(results)} predicted moisture points")
    return results

@app.get("/wilt-forecast")
def get_wilt_forecast(wilt_point: float = 18.0, upper_limit: float = 22.0):
    df_pred = get_predicted_moisture()

    if not isinstance(df_pred, list) or not df_pred:
        return {"error": "Unable to retrieve predicted moisture data."}

    for i, row in enumerate(df_pred):
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
