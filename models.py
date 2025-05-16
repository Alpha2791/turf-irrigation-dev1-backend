from sqlalchemy import Column, DateTime, Float, String
from database import Base

from sqlalchemy import Column, String
from database import Base  # Or however your Base is imported

class PredictionMeta(Base):
    __tablename__ = "prediction_meta"

    key = Column(String, primary_key=True)
    value = Column(String)

class WeatherHistory(Base):
    __tablename__ = "weather_history"
    timestamp = Column(DateTime, primary_key=True, index=True)
    et_mm_hour = Column(Float)
    rainfall_mm = Column(Float)
    solar_radiation = Column(Float)
    temp_c = Column(Float)
    humidity = Column(Float)
    windspeed = Column(Float)

class PredictionMeta(Base):
    __tablename__ = "prediction_meta"
    key = Column(String, primary_key=True)
    value = Column(String)


class MoistureLog(Base):
    __tablename__ = "moisture"

    timestamp = Column(DateTime, primary_key=True)
    moisture_mm = Column(Float, nullable=False)

class IrrigationLog(Base):
    __tablename__ = "irrigation"

    timestamp = Column(DateTime, primary_key=True)
    irrigation_mm = Column(Float, nullable=False)
