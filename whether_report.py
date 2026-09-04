from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Dummy AI model prediction function (replace with real ML model later)
def predict_inundation(humidity, pressure, radar_reflectivity):
    # Placeholder logic
    risk_score = (humidity * 0.4) + (radar_reflectivity * 0.6) - (pressure * 0.1)
    if risk_score > 50:
        return "HIGH RISK: Severe Heavy Rainfall & Inundation Expected", "Red"
    elif risk_score > 30:
        return "MEDIUM RISK: Moderate Rainfall", "Orange"
    else:
        return "LOW RISK: Safe Conditions", "Green"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Receive JSON data sent from HTML JS
    humidity = float(data.get("humidity", 0))
    pressure = float(data.get("pressure", 0))
    radar = float(data.get("radar", 0))

    status, level = predict_inundation(humidity, pressure, radar)
    return jsonify({"status": status, "level": level})

if __name__ == "__main__":
    app.run(debug=True)