# 🚍 Real-Time Smart Public Transport ETA Prediction System

A **Machine Learning powered public transport prediction system** that estimates **bus arrival times (ETA)** using live vehicle location, traffic conditions, weather data, and occupancy prediction.

This project demonstrates how **AI + real-time APIs** can improve urban public transportation systems.

---

# 📌 Features

- ✅ Real-time **ETA prediction** for public transport
- ✅ **Machine Learning occupancy prediction**
- ✅ **Weather-based traffic adjustment**
- ✅ **Dynamic route selection (From Stop → To Stop)**
- ✅ **Interactive map for vehicle location**
- ✅ **Driver analytics dashboard**
- ✅ **Per-stop ETA breakdown**
- ✅ **Trip analytics tracking**

---

# 🛠️ Technologies Used

### Backend

* Python
* FastAPI
* NumPy
* Scikit-learn
* Requests API

### Frontend

* HTML
* JavaScript
* Leaflet.js Map

### Machine Learning

* Occupancy Prediction Model
* Time-of-day features
* Sensor input data

### External APIs

* Open-Meteo Weather API

---

# 📂 Project Structure

```
public-transport-ml/
│
├── server.py                 # FastAPI backend
├── index.html                # Frontend UI
├── models/
│   └── occ_model.pkl         # Trained occupancy ML model
│   └── eta_model.pkl
├── requirements.txt
├── README.md
```

---

# ▶️ Running the Project

Start the FastAPI server:

```bash
uvicorn server:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

Open `index.html` in your browser.

---

# 🧮 How ETA is Calculated

The system calculates ETA using:

### 1️⃣ Vehicle Speed

User provided real-time bus speed.

### 2️⃣ Occupancy Prediction

ML model predicts passenger load:

* Low
* Medium
* High

Higher occupancy → slower speed.

### 3️⃣ Weather Traffic Factor

Weather API adjusts traffic multiplier:

| Weather | Traffic Multiplier |
| ------- | ------------------ |
| Clear   | 1.1                |
| Rain    | 1.5                |
| Snow    | 1.6                |

### 4️⃣ Distance Calculation

Distance between stops is calculated using **Haversine formula**.

### Final Formula

```
Adjusted Speed =
Base Speed × Occupancy Factor ÷ Traffic Multiplier
```

ETA:

```
ETA = Distance / Adjusted Speed
```

---

# 📊 Driver Analytics

The system stores trip history and provides analytics such as:

* Total trips completed
* Average speed
* Average ETA
* Traffic multipliers
* Occupancy distribution
* Total distance covered

Endpoint:

```
GET /analytics
```

Example response:

```json
{
 "total_trips": 6,
 "average_speed_kmph": 40.0,
 "average_eta_minutes": 9.3,
 "average_traffic_multiplier": 1.1,
 "total_distance_km": 37.3
}
```

---

# 🗺️ Interactive Map

The UI includes an interactive map where users can:

* Select vehicle location
* Choose start stop
* Choose destination stop
* Run ETA prediction

---

# 🧪 Example Prediction Request

```
POST /predict
```

Example payload:

```json
{
 "lat": 22.594084,
 "lon": 88.395486,
 "speed_kmph": 18,
 "sensor_estimate": 6,
 "from_stop": 3,
 "to_stop": 8
}
```


---

# 👨‍💻 Author

**Arghyadip Ghosh**

Machine Learning / AI Project

---

# ⭐ If you like this project

Please consider **starring the repository** on GitHub.
