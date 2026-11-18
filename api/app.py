
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import os
from typing import Literal
import pandas as pd

app = FastAPI(title="Kalamkari Fabric Drying Time API", description="Predict drying time for craft production", version="1.0.0")

FEATURE_COLUMNS = [
    "fabric_length_meters",
    "dye_coats",
    "humidity_percentage",
    "ambient_temperature",
    "airflow_speed",
    "drying_rack_occupancy",
    "moisture_load_index",
    "airflow_per_meter",
    "fabric_material_encoded",
]

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

class PredictionInput(BaseModel):
    fabric_material: Literal["Cotton", "Silk", "cotton", "silk"] = Field(..., description="Cotton or Silk")
    fabric_length_meters: float = Field(..., gt=0)
    dye_coats: int = Field(..., gt=0, le=10)
    humidity_percentage: float = Field(..., ge=0, le=100)
    ambient_temperature: float = Field(...)
    airflow_speed: float = Field(..., ge=0)
    drying_rack_occupancy: float = Field(..., ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    # Re-apply feature engineering
    mat_encoded = 0 if data.fabric_material.lower() == 'cotton' else 1
    moisture_load_index = data.fabric_length_meters * data.dye_coats * data.humidity_percentage / 100
    airflow_per_meter = data.airflow_speed / data.fabric_length_meters

    feature_row = [
        data.fabric_length_meters,
        data.dye_coats,
        data.humidity_percentage,
        data.ambient_temperature,
        data.airflow_speed,
        data.drying_rack_occupancy,
        moisture_load_index,
        airflow_per_meter,
        mat_encoded,
    ]

    input_frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_frame)[0]
    # Ensure prediction is non-negative
    prediction = max(0.0, float(prediction))
    return {"predicted_drying_duration_hours": round(prediction, 2)}
