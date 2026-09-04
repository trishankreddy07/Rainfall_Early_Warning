from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd
import urllib.request
import json
import os

app = Flask(__name__, template_folder='templates')

# Load the saved ML Model / Pipeline gracefully using joblib / pickle
pipeline = None
model_type = "pipeline"

if os.path.exists('rainfall_pipeline.pkl'):
    try:
        pipeline = joblib.load('rainfall_pipeline.pkl')
        model_type = "pipeline"
    except Exception as e:
        print(f"Error loading rainfall_pipeline.pkl with joblib: {e}")
        try:
            with open('rainfall_pipeline.pkl', 'rb') as f:
                pipeline = pickle.load(f)
                model_type = "pipeline"
        except Exception as e2:
            print(f"Error loading rainfall_pipeline.pkl with pickle: {e2}")

if pipeline is None and os.path.exists('rainfall_model.pkl'):
    try:
        pipeline = joblib.load('rainfall_model.pkl')
        model_type = "model"
    except Exception as e:
        try:
            with open('rainfall_model.pkl', 'rb') as f:
                pipeline = pickle.load(f)
                model_type = "model"
        except Exception as e2:
            print(f"Error loading rainfall_model.pkl: {e2}")


def parse_input(val, default=0.0):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def make_prediction(humidity, pressure, min_temp, max_temp, rainfall):
    """
    Computes prediction using either a 5-feature DataFrame (for pipeline)
    or a 3-feature array [humidity, pressure, rainfall] (for rainfall_model.pkl).
    """
    if pipeline is None:
        raise RuntimeError("No trained model found on server.")

    # Convert inputs to floats
    h = parse_input(humidity, 70.0)
    p = parse_input(pressure, 1013.0)
    mn = parse_input(min_temp, 20.0)
    mx = parse_input(max_temp, 30.0)
    rf = parse_input(rainfall, 0.0)

    # Check if pipeline expects 5 named features or 3 array features
    use_dataframe = False
    if hasattr(pipeline, 'feature_names_in_') and len(pipeline.feature_names_in_) == 5:
        use_dataframe = True
    elif model_type == "pipeline":
        use_dataframe = True

    if use_dataframe:
        input_data = pd.DataFrame([{
            'Humidity9am': h,
            'Pressure9am': p,
            'MinTemp': mn,
            'MaxTemp': mx,
            'Rainfall': rf
        }])
    else:
        input_data = np.array([[h, p, rf]])

    try:
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
    except Exception:
        # Fallback to alternative format if prediction failed
        if use_dataframe:
            input_data = np.array([[h, p, rf]])
        else:
            input_data = pd.DataFrame([{
                'Humidity9am': h,
                'Pressure9am': p,
                'MinTemp': mn,
                'MaxTemp': mx,
                'Rainfall': rf
            }])
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]

    rain_probability = round(float(probabilities[1]) * 100, 2)

    if prediction == 1:
        status = f"HIGH RISK: Heavy Rainfall Expected ({rain_probability}% probability)"
        level = "Red"
    else:
        status = f"LOW RISK: Clear Weather Expected ({round(100 - rain_probability, 2)}% safety score)"
        level = "Green"

    return status, level, rain_probability


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    humidity = data.get("humidity")
    pressure = data.get("pressure")
    min_temp = data.get("min_temp", 20.0)
    max_temp = data.get("max_temp", 30.0)
    rainfall = data.get("rainfall", 0.0)

    try:
        status, level, probability = make_prediction(humidity, pressure, min_temp, max_temp, rainfall)
        return jsonify({
            "status": status,
            "level": level,
            "probability": probability
        })
    except Exception as e:
        return jsonify({
            "status": f"Prediction Error: {str(e)}",
            "level": "Red",
            "probability": 0.0
        }), 500


@app.route("/fetch_weather", methods=["GET", "POST"])
def fetch_weather():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        lat = data.get("latitude")
        lon = data.get("longitude")
    else:
        lat = request.args.get("latitude")
        lon = request.args.get("longitude")

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude parameters are required"}), 400

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=relative_humidity_2m,surface_pressure,precipitation,temperature_2m"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=auto"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))

        current = res_data.get("current", {})
        daily = res_data.get("daily", {})

        humidity = current.get("relative_humidity_2m", 70.0)
        pressure = current.get("surface_pressure", 1013.25)
        rainfall = current.get("precipitation", 0.0)
        current_temp = current.get("temperature_2m", 25.0)

        max_temp_list = daily.get("temperature_2m_max", [])
        min_temp_list = daily.get("temperature_2m_min", [])

        max_temp = max_temp_list[0] if max_temp_list else current_temp + 5.0
        min_temp = min_temp_list[0] if min_temp_list else current_temp - 5.0

        status, level, probability = make_prediction(humidity, pressure, min_temp, max_temp, rainfall)

        weather_obs = {
            "humidity": round(float(humidity), 1),
            "pressure": round(float(pressure), 1),
            "rainfall": round(float(rainfall), 1),
            "min_temp": round(float(min_temp), 1),
            "max_temp": round(float(max_temp), 1),
            "current_temp": round(float(current_temp), 1),
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4)
        }

        return jsonify({
            "status": status,
            "level": level,
            "probability": probability,
            "weather": weather_obs
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch weather data: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)