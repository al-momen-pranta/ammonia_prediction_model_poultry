from flask import Flask, request, jsonify
import numpy as np
import json
import os
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading LSTM model...")
model = tf.keras.models.load_model("lstm_model.h5", compile=False)
print("Model ready.")

# ── Load scalers ──────────────────────────────────────────────────────────────
with open("scaler_params.json") as f:
    scaler = json.load(f)

with open("nh3_scaler_params.json") as f:
    nh3_scaler = json.load(f)

DATA_SCALE = np.array(scaler["scale"], dtype=np.float32)
DATA_MINN  = np.array(scaler["min"],   dtype=np.float32)
NH3_SCALE  = float(nh3_scaler["scale"])
NH3_MINN   = float(nh3_scaler["min"])

LOOKBACK   = 30
N_FEATURES = 3
THRESHOLD  = 20.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(X):
    return X * DATA_SCALE + DATA_MINN

def inverse_nh3(y):
    return (y - NH3_MINN) / NH3_SCALE

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "model": "LSTM NH3 Predictor",
        "usage": "POST /predict"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "readings" not in data:
            return jsonify({"error": "Missing 'readings'"}), 400

        readings = data["readings"]

        if len(readings) != LOOKBACK:
            return jsonify({"error": f"Need {LOOKBACK} readings, got {len(readings)}"}), 400

        X = np.array(
            [[r["nh3"], r["temp"], r["hum"]] for r in readings],
            dtype=np.float32
        )

        X_scaled = normalize(X).reshape(1, LOOKBACK, N_FEATURES)
        y_scaled = model.predict(X_scaled, verbose=0)[0][0]

        predicted_ppm = float(inverse_nh3(y_scaled))
        predicted_ppm = max(0.0, round(predicted_ppm, 2))

        fan_action = "ON" if predicted_ppm > THRESHOLD else "OFF"

        current_nh3 = readings[-1]["nh3"]
        current_temp = readings[-1]["temp"]
        current_hum = readings[-1]["hum"]
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # ── Deploy log এ দেখা যাবে ───────────────────────────────────────────
        print(f"[{timestamp}] PREDICT | "
              f"NH3: {current_nh3:.1f} ppm | "
              f"Temp: {current_temp:.1f}C | "
              f"Hum: {current_hum:.1f}% | "
              f"Predicted: {predicted_ppm:.2f} ppm | "
              f"Threshold: {THRESHOLD} ppm | "
              f"Fan: {fan_action}")

        return jsonify({
            "predicted_nh3_ppm": predicted_ppm,
            "threshold_ppm": THRESHOLD,
            "fan_action": fan_action,
            "alert": predicted_ppm > THRESHOLD,
            "current_nh3": current_nh3,
            "message": f"NH3 will reach {predicted_ppm} ppm in 30 min — Fan {fan_action}"
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
