"""
NH3 Prediction API — Flask + Keras
Compatible with TF 2.13 saved models on TF 2.19
"""

from flask import Flask, request, jsonify
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import regularizers

app = Flask(__name__)

LOOKBACK   = 30
N_FEATURES = 3
THRESHOLD  = 10.0
L2_REG     = 1e-4

# ── Rebuild model architecture then load weights ─────────────────────
print("Building model architecture...")

model = Sequential([
    Input(shape=(LOOKBACK, N_FEATURES)),
    LSTM(64, return_sequences=True,
         kernel_regularizer=regularizers.l2(L2_REG),
         recurrent_regularizer=regularizers.l2(L2_REG)),
    Dropout(0.2),
    LSTM(32, return_sequences=False,
         kernel_regularizer=regularizers.l2(L2_REG),
         recurrent_regularizer=regularizers.l2(L2_REG)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse")

print("Loading weights from keras file...")
# Load weights only — avoids architecture compatibility issues
try:
    model.load_weights("lstm_nh3.keras")
    print("Weights loaded successfully (load_weights)")
except Exception as e1:
    print(f"load_weights failed: {e1}")
    try:
        # Try loading full model with compile=False
        loaded = tf.keras.models.load_model(
            "lstm_nh3.keras", compile=False)
        model.set_weights(loaded.get_weights())
        print("Weights transferred from loaded model")
    except Exception as e2:
        print(f"ERROR: {e2}")
        raise

# ── Load scaler params ───────────────────────────────────────────────
with open("scaler_params.json") as f:
    scaler = json.load(f)
with open("nh3_scaler_params.json") as f:
    nh3_scaler = json.load(f)

DATA_SCALE = np.array(scaler["scale"], dtype=np.float32)
DATA_MINN  = np.array(scaler["min"],   dtype=np.float32)
NH3_SCALE  = float(nh3_scaler["scale"])
NH3_MINN   = float(nh3_scaler["min"])

def normalize(X):
    return X * DATA_SCALE + DATA_MINN

def inverse_nh3(y):
    return (y - NH3_MINN) / NH3_SCALE

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "model":  "LSTM NH3 Predictor",
        "usage":  "POST /predict"
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
            return jsonify({
                "error": f"Need {LOOKBACK} readings, got {len(readings)}"
            }), 400

        X = np.array([[r["nh3"], r["temp"], r["hum"]]
                      for r in readings], dtype=np.float32)
        X_scaled = normalize(X).reshape(1, LOOKBACK, N_FEATURES)

        y_scaled = model.predict(X_scaled, verbose=0)[0][0]
        predicted_ppm = float(inverse_nh3(y_scaled))
        predicted_ppm = max(0.0, round(predicted_ppm, 2))

        fan_action = "ON" if predicted_ppm > THRESHOLD else "OFF"

        return jsonify({
            "predicted_nh3_ppm": predicted_ppm,
            "threshold_ppm":     THRESHOLD,
            "fan_action":        fan_action,
            "alert":             predicted_ppm > THRESHOLD,
            "current_nh3":       readings[-1]["nh3"],
            "message": (
                f"NH3 will reach {predicted_ppm} ppm in 30 min — "
                f"Fan {fan_action}"
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
