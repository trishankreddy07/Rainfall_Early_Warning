# 🌧️ AI/ML Heavy Rainfall & Inundation Risk Early Warning System

An intelligent, real-time weather risk assessment web application that uses machine learning to predict severe rainfall and inundation risks. Integrated with live satellite/station telemetry via Open-Meteo APIs and GPS auto-location.

🚀 **Live Demo:** [rainfall-early-warning.onrender.com](https://rainfall-early-warning.onrender.com/)

---

## ✨ Features

- **📍 One-Click GPS Auto-Location:** Uses browser geolocation (`navigator.geolocation`) to fetch immediate local weather parameters.
- **🌍 Worldwide City Search:** Built-in geocoding allows searching risk forecasts for any city globally.
- **🛰️ Live Weather Data Streaming:** Real-time atmospheric metrics (Humidity, Surface Pressure, Precipitation, Temperatures) fetched seamlessly via Open-Meteo API without requiring API key friction.
- **🤖 ML Risk Engine:** Driven by a trained `RandomForestClassifier` to compute risk status and probability scores.
- **🎛️ Interactive Manual Overrides:** Form fields auto-populate with live telemetry while allowing manual parameter tweaks for custom simulation testing.
- **🎨 Dark Glassmorphic UI:** Modern, responsive web interface complete with color-coded risk indicators (High Risk 🔴 / Low Risk 🟢) and visual loading states.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.14, Flask, Gunicorn
- **Machine Learning:** Scikit-Learn, NumPy, Pickle
- **External APIs:** Open-Meteo Weather & Geocoding APIs
- **Frontend:** HTML5, Modern CSS (Glassmorphism), JavaScript (Fetch API), FontAwesome Icons
- **Deployment:** Render

---

## 📂 Repository Structure

```text
Rainfall_Early_Warning/
├── app.py                 # Flask server, API endpoints & model inference
├── rainfall_model.pkl     # Trained ML classification model
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment instructions for Render / Heroku
└── templates/
    └── index.html         # Frontend interface with JS fetch & geolocation