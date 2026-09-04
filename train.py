from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained ML model
with open('rainfall_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Extract inputs from JavaScript fetch call
    humidity = float(data.get("humidity", 0))
    pressure = float(data.get("pressure", 0))
    radar = float(data.get("radar", 0))

    # Reformat inputs into a 2D array for scikit-learn
    input_features = np.array([[humidity, pressure, radar]])

    # Make Prediction (Returns 0 or 1)
    prediction = model.predict(input_features)[0]
    
    # Get Probability of High Risk Rainfall
    probabilities = model.predict_proba(input_features)[0]
    high_risk_prob = round(probabilities[1] * 100, 2)

    if prediction == 1:
        status = f"HIGH RISK: Heavy Rainfall Expected ({high_risk_prob}% probability)"
        level = "Red"
    else:
        status = f"LOW RISK: Normal Conditions ({100 - high_risk_prob}% safety score)"
        level = "Green"

    return jsonify({"status": status, "level": level})

if __name__ == "__main__":
    app.run(debug=True)