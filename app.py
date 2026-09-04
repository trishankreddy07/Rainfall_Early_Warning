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


def safe_predict(input_data):
    """
    Defensive prediction function that handles array bounds checking
    and single-class probability returns gracefully.
    """
    if pipeline is None:
        raise RuntimeError("No trained model found on server.")

    # Execute prediction
    prediction = pipeline.predict(input_data)[0]

    # Default fallback probability thresholds if indexing fails or is incomplete
    rain_probability = 85.0 if prediction == 1 else 15.0

    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_data)
            # Unwrap 2D array shape (1, N) to 1D array shape (N,)
            if isinstance(probs, np.ndarray) and probs.ndim > 1:
                probs = probs[0]

            # Defensive array bounds check
            if len(probs) > 1:
                rain_probability = round(float(probs[1]) * 100, 2)
            elif len(probs) == 1:
                cls0_prob = float(probs[0])
                if prediction == 1:
                    rain_probability = round(cls0_prob * 100, 2)
                else:
                    rain_probability = round((1.0 - cls0_prob) * 100, 2)
    except Exception as e:
        print(f"Warning: safe_predict encountered probability indexing issue: {e}")

    if prediction == 1:
        status = f"HIGH RISK: Heavy Rainfall Expected ({rain_probability}% probability)"
        level = "Red"
    else:
        status = f"LOW RISK: Clear Weather Expected ({round(100 - rain_probability, 2)}% safety score)"
        level = "Green"

    return status, level, rain_probability


def process_and_predict(humidity, pressure, min_temp, max_temp, rainfall):
    """
    Formats input data for 5-feature DataFrames or 3-feature numpy arrays
    and routes inference through safe_predict.
    """
    h = parse_input(humidity, 70.0)
    p = parse_input(pressure, 1013.0)
    mn = parse_input(min_temp, 20.0)
    mx = parse_input(max_temp, 30.0)
    rf = parse_input(rainfall, 0.0)

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
        return safe_predict(input_data)
    except Exception:
        if use_dataframe:
            alt_data = np.array([[h, p, rf]])
        else:
            alt_data = pd.DataFrame([{
                'Humidity9am': h,
                'Pressure9am': p,
                'MinTemp': mn,
                'MaxTemp': mx,
                'Rainfall': rf
            }])
        return safe_predict(alt_data)


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
        status, level, probability = process_and_predict(humidity, pressure, min_temp, max_temp, rainfall)
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
            f"&current=relative_humidity_2m,surface_pressure,temperature_2m"
            f"&hourly=relative_humidity_2m,surface_pressure,precipitation,temperature_2m"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=auto"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))

        current = res_data.get("current", {})
        daily = res_data.get("daily", {})
        hourly = res_data.get("hourly", {})

        humidity = round(float(current.get("relative_humidity_2m", 70.0)), 1)
        pressure = round(float(current.get("surface_pressure", 1013.25)), 1)
        current_temp = round(float(current.get("temperature_2m", 25.0)), 1)

        max_temp_list = daily.get("temperature_2m_max", [])
        min_temp_list = daily.get("temperature_2m_min", [])
        precip_list = daily.get("precipitation_sum", [])

        max_temp = round(float(max_temp_list[0]), 1) if max_temp_list and max_temp_list[0] is not None else round(current_temp + 5.0, 1)
        min_temp = round(float(min_temp_list[0]), 1) if min_temp_list and min_temp_list[0] is not None else round(current_temp - 5.0, 1)

        if precip_list and len(precip_list) > 0 and precip_list[0] is not None:
            rainfall = round(float(precip_list[0]), 1)
        else:
            rainfall = 0.0

        status, level, probability = process_and_predict(humidity, pressure, min_temp, max_temp, rainfall)

        # Build 24-Hour Hourly Trajectory
        hourly_times = hourly.get("time", [])
        hourly_hum = hourly.get("relative_humidity_2m", [])
        hourly_pres = hourly.get("surface_pressure", [])
        hourly_precip = hourly.get("precipitation", [])

        current_time_str = current.get("time", "")
        start_idx = 0
        if current_time_str and hourly_times:
            curr_prefix = current_time_str.split(":")[0]  # e.g., "2026-09-04T19"
            for idx, t_str in enumerate(hourly_times):
                if t_str.startswith(curr_prefix):
                    start_idx = idx
                    break

        end_idx = min(start_idx + 24, len(hourly_times))
        indices = list(range(start_idx, end_idx))
        if len(indices) < 24 and len(hourly_times) >= 24:
            indices = list(range(max(0, len(hourly_times) - 24), len(hourly_times)))

        hourly_trajectory = []
        for i in indices:
            raw_time = hourly_times[i] if i < len(hourly_times) else f"H+{i}"
            time_display = raw_time.split("T")[1] if "T" in raw_time else raw_time
            
            h_hum = round(float(hourly_hum[i]), 1) if i < len(hourly_hum) and hourly_hum[i] is not None else humidity
            h_pres = round(float(hourly_pres[i]), 1) if i < len(hourly_pres) and hourly_pres[i] is not None else pressure
            h_precip = round(float(hourly_precip[i]), 1) if i < len(hourly_precip) and hourly_precip[i] is not None else 0.0

            h_status, h_level, h_prob = process_and_predict(h_hum, h_pres, min_temp, max_temp, h_precip)

            hourly_trajectory.append({
                "time": time_display,
                "humidity": h_hum,
                "pressure": h_pres,
                "rainfall": h_precip,
                "probability": h_prob,
                "level": h_level,
                "status": h_status
            })

        weather_obs = {
            "humidity": humidity,
            "pressure": pressure,
            "rainfall": rainfall,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "current_temp": current_temp,
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4)
        }

        return jsonify({
            "status": status,
            "level": level,
            "probability": probability,
            "humidity": humidity,
            "pressure": pressure,
            "rainfall": rainfall,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "weather": weather_obs,
            "hourly_trajectory": hourly_trajectory
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch weather data: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)