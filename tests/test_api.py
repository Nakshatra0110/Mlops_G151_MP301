"""TestClient tests: health, a valid prediction and the two client rules."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID = {
    "students_registered": 500,
    "attendance_pct": 72.5,
    "menu_rating": 4.1,
    "temp_c": 24.0,
    "cooked_kg": 180.0,
    "plate_size_cm": 21.0,
    "meal": "lunch",
    "hostel": "Boys Hostel 1",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_number():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))


def test_cooked_kg_above_1500_is_422():
    bad = {**VALID, "cooked_kg": 1501}
    assert client.post("/predict", json=bad).status_code == 422


def test_plate_size_below_10_is_422():
    bad = {**VALID, "plate_size_cm": 9.9}
    assert client.post("/predict", json=bad).status_code == 422


def test_request_has_exactly_the_client_fields():
    assert set(VALID) == {
        "students_registered",
        "attendance_pct",
        "menu_rating",
        "temp_c",
        "cooked_kg",
        "plate_size_cm",
        "meal",
        "hostel",
    }
