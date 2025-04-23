from sqlalchemy import Column, Float, DateTime
from database import Base

class MoistureLog(Base):
    __tablename__ = "moisture"

    timestamp = Column(DateTime, primary_key=True)
    moisture_mm = Column(Float, nullable=False)

class IrrigationLog(Base):
    __tablename__ = "irrigation"

    timestamp = Column(DateTime, primary_key=True)
    irrigation_mm = Column(Float, nullable=False)
