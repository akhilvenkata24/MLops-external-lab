
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_valid():
    payload = {
        "fabric_material": "Cotton",
        "fabric_length_meters": 5.0,
        "dye_coats": 2,
        "humidity_percentage": 50.0,
        "ambient_temperature": 30.0,
        "airflow_speed": 2.0,
        "drying_rack_occupancy": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_drying_duration_hours" in response.json()
    assert response.json()["predicted_drying_duration_hours"] >= 0

def test_prediction_invalid():
    payload = {
        "fabric_material": "Cotton",
        "fabric_length_meters": -1.0,  # Invalid length
        "dye_coats": 2,
        "humidity_percentage": 120.0, # Invalid humidity
        "ambient_temperature": 30.0,
        "airflow_speed": 2.0,
        "drying_rack_occupancy": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
