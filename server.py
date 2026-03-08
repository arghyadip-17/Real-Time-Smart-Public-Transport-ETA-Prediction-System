# server.py
# Realistic cumulative ETA + occupancy + weather traffic + analytics (named stops)

import requests
import pickle
import math
from datetime import datetime
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# In-memory analytics
# -----------------------
trip_history: List[Dict] = []

# -----------------------
# Load occupancy model
# -----------------------
with open("models/occ_model.pkl", "rb") as f:
    occ_blob = pickle.load(f)
occ_model = occ_blob["model"]

# -----------------------
# Named stops (match frontend)
# -----------------------
STOPS = [
    {"name": "Shobhabazar", "coords": (22.5980, 88.3650)},
    {"name": "Shobhabazar Metro", "coords": (22.5960, 88.3680)},
    {"name": "Hatibagan", "coords": (22.5935, 88.3720)},
    {"name": "Maniktala", "coords": (22.5895, 88.3770)},
    {"name": "Vivekananda Road", "coords": (22.5855, 88.3810)},
    {"name": "Rajabazar", "coords": (22.5825, 88.3860)},
    {"name": "Sealdah", "coords": (22.5755, 88.3920)},
    {"name": "Moulali", "coords": (22.5725, 88.3970)},
    {"name": "CIT Road", "coords": (22.5690, 88.4010)},
    {"name": "Phoolbagan", "coords": (22.5660, 88.4050)},
    {"name": "Kankurgachi", "coords": (22.5625, 88.4100)},
    {"name": "Ultadanga", "coords": (22.5890, 88.4120)},
]

# -----------------------
# Utility functions
# -----------------------
def live_traffic_multiplier(lat: float, lon: float) -> float:
    """
    Query Open-Meteo for current weather and map to a traffic multiplier.
    (Simple heuristic: rain/snow => higher multiplier)
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        resp = requests.get(url, timeout=3)
        data = resp.json()
        weather_code = data.get("current_weather", {}).get("weathercode", None)
        if weather_code is None:
            return 1.2
        # weather codes mapping (rough)
        if 51 <= weather_code <= 67:  # drizzle / rain
            return 1.5
        if 71 <= weather_code <= 77:  # snow
            return 1.6
        return 1.1  # normal slight traffic
    except Exception:
        return 1.2  # fallback


def haversine_km(a: tuple, b: tuple) -> float:
    """Haversine distance in kilometers between (lat, lon) points a and b."""
    R = 6371.0
    lat1, lon1 = a
    lat2, lon2 = b
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def occupancy_speed_factor(occ_cat: int) -> float:
    """Map occupancy category to speed multiplier."""
    return {0: 1.0, 1: 0.9, 2: 0.7}.get(occ_cat, 1.0)


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -----------------------
# Request model
# -----------------------
class VehicleUpdate(BaseModel):
    lat: float
    lon: float
    speed_kmph: float
    sensor_estimate: float
    stop_index: int


# -----------------------
# Predict endpoint
# -----------------------
@app.post("/predict")
def predict(data: VehicleUpdate):
    now = datetime.utcnow()
    tod_seconds = now.hour * 3600 + now.minute * 60 + now.second
    tod_sin = math.sin(2 * math.pi * tod_seconds / 86400)
    # basic validation
    if data.stop_index < 0:
        raise HTTPException(status_code=400, detail="stop_index must be >= 0")

    if data.stop_index >= len(STOPS) - 1:
        return {
            "total_eta_minutes": 0.0,
            "per_stop_etas": [],
            "Occupancy": "Route Completed",
            "occupancy_confidence": None,
            "predicted_speed_kmph": None,
            "distance_to_next_km": 0.0,
            "used_fallback": False,
            "timestamp_iso": now.isoformat(),
        }

    # Occupancy prediction (sensor_estimate, time-of-day sin, stop_index)
    X_occ = np.array([[safe_float(data.sensor_estimate), tod_sin, float(data.stop_index)]], dtype=float).reshape(1, -1)
    try:
        occ_cat = int(occ_model.predict(X_occ)[0])
        occ_conf = None
        if hasattr(occ_model, "predict_proba"):
            occ_conf = float(np.max(occ_model.predict_proba(X_occ)[0]))
    except Exception:
        occ_cat = 1
        occ_conf = None

    # Distances: current -> next stop, then between subsequent stops
    distances: List[float] = []
    # next stop coords
    next_stop_coords = STOPS[data.stop_index + 1]["coords"]
    d0 = haversine_km((safe_float(data.lat), safe_float(data.lon)), next_stop_coords)
    distances.append(d0)
    for i in range(data.stop_index + 1, len(STOPS) - 1):
        a = STOPS[i]["coords"]
        b = STOPS[i + 1]["coords"]
        distances.append(haversine_km(a, b))

    total_distance_km = sum(distances)

    # Base speed and traffic multiplier
    base_speed = max(safe_float(data.speed_kmph), 3.0)  # minimum reasonable speed
    tmult = live_traffic_multiplier(data.lat, data.lon)

    # Apply occupancy penalty first, then traffic
    occ_factor = occupancy_speed_factor(occ_cat)
    predicted_speed_kmph = base_speed * occ_factor / tmult

    # City realism caps
    MAX_CITY_SPEED = 40.0
    MIN_CITY_SPEED = 8.0
    if predicted_speed_kmph > MAX_CITY_SPEED:
        predicted_speed_kmph = MAX_CITY_SPEED
    if predicted_speed_kmph < MIN_CITY_SPEED:
        predicted_speed_kmph = MIN_CITY_SPEED

    # Debug
    print(f"DEBUG: base_speed={base_speed}, occ_factor={occ_factor}, tmult={tmult}, final_speed={predicted_speed_kmph}")

    # Compute per-segment minutes and cumulative per-stop ETAs
    per_segment_minutes = [(d / predicted_speed_kmph) * 60.0 for d in distances]
    per_stop_etas = []
    cum = 0.0
    for idx, seg_min in enumerate(per_segment_minutes):
        cum += seg_min
        stop_idx = data.stop_index + 1 + idx
        per_stop_etas.append(
            {
                "stop_index": stop_idx,
                "stop_name": STOPS[stop_idx]["name"],
                "eta_minutes_from_now": round(cum, 3),
                "segment_distance_km": round(distances[idx], 4),
            }
        )

    total_eta_minutes = round(sum(per_segment_minutes), 3)

    # Save to analytics (in-memory)
    trip_history.append(
        {
            "timestamp": now.isoformat(),
            "start_stop_index": data.stop_index,
            "start_stop_name": STOPS[data.stop_index]["name"] if 0 <= data.stop_index < len(STOPS) else None,
            "predicted_speed_kmph": round(predicted_speed_kmph, 3),
            "eta_minutes": total_eta_minutes,
            "occupancy": int(occ_cat),
            "occupancy_confidence": None if occ_conf is None else round(occ_conf, 3),
            "traffic_multiplier": float(tmult),
            "distance_km": round(total_distance_km, 4),
        }
    )

    # Response
    return {
        "total_eta_minutes": total_eta_minutes,
        "per_stop_etas": per_stop_etas,
        "Occupancy": {0: "Low", 1: "Medium", 2: "High"}.get(occ_cat, "Unknown"),
        "occupancy_confidence": None if occ_conf is None else round(occ_conf, 3),
        "predicted_speed_kmph": round(predicted_speed_kmph, 3),
        "distance_to_next_km": round(distances[0], 4),
        "used_fallback": False,
        "timestamp_iso": now.isoformat(),
    }


# -----------------------
# Analytics endpoint
# -----------------------
@app.get("/analytics")
def analytics():
    if not trip_history:
        return {"message": "No trip data yet"}

    speeds = [t["predicted_speed_kmph"] for t in trip_history]
    etas = [t["eta_minutes"] for t in trip_history]
    traff = [t["traffic_multiplier"] for t in trip_history]
    dists = [t["distance_km"] for t in trip_history]
    occupancies = [t["occupancy"] for t in trip_history]

    avg_speed = sum(speeds) / len(speeds)
    avg_eta = sum(etas) / len(etas)
    avg_traffic = sum(traff) / len(traff)
    total_distance = sum(dists)

    occ_dist = {"Low": occupancies.count(0), "Medium": occupancies.count(1), "High": occupancies.count(2)}

    return {
        "total_trips": len(trip_history),
        "average_speed_kmph": round(avg_speed, 3),
        "average_eta_minutes": round(avg_eta, 3),
        "average_traffic_multiplier": round(avg_traffic, 3),
        "total_distance_km": round(total_distance, 3),
        "occupancy_distribution": occ_dist,
    }
