"""
Export model weights only — TF version independent
Run on your PC
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Loading model...")
# Load with custom objects to bypass time_major issue
try:
    model = tf.keras.models.load_model(
        "lstm_weights.weights.h5",
        compile=False,
        safe_mode=False
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Save weights only as H5
weights_path = "realistic_lstm_outputs/lstm_weights.weights.h5"
model.save_weights(weights_path)
print(f"Weights saved → {weights_path}")

# Verify
print(f"Number of layers: {len(model.layers)}")
for i, layer in enumerate(model.layers):
    print(f"  Layer {i}: {layer.name} — {layer.__class__.__name__}")

print("\nUpload lstm_weights.weights.h5 to GitHub")
