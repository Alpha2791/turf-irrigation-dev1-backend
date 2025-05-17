from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MoistureLog(Base):
    __tablename__ = "moisture"

    timestamp = Column(DateTime, primary_key=True)
    moisture_mm = Column(Float, nullable=False)

class IrrigationLog(Base):
    __tablename__ = "irrigation"

    timestamp = Column(DateTime, primary_key=True)
    irrigation_mm = Column(Float, nullable=False)

class WeatherHistory(Base):
    __tablename__ = "weather_history"
    __table_args__ = {'extend_existing': True}

    timestamp = Column(DateTime, primary_key=True)
    et_mm_hour = Column(Float, nullable=False)
    rainfall_mm = Column(Float, nullable=False)
    solar_radiation = Column(Float)
    temp_c = Column(Float)
    humidity = Column(Float)
    windspeed = Column(Float)

class PredictionMeta(Base):
    __tablename__ = "prediction_meta"
    __table_args__ = {'extend_existing': True}

    key = Column(String, primary_key=True)
    value = Column(String)
