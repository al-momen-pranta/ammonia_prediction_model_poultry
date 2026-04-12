"""
Convert new .keras model to .h5 for Railway API
Run in F:\NEW\THESIS\
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model(
    "lstm_cleaned_outputs/lstm_nh3_combined.keras",
    compile=False
)
print("Model loaded!")

model.save("lstm_model.h5")
print("Saved as lstm_model.h5 — upload this to GitHub!")
