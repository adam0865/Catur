import numpy as np
import torch 
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten

# ========================
# 1. Set Path dan Konstanta
# ========================
MODEL_PATH = "chess_ppo_model_console.weights.h5"  # ← Perbaikan di sini

# ========================
# 2. Cek dan Load Model
# ========================
def build_model():
    model = Sequential([
        Flatten(input_shape=(8, 8, 12)),  # Contoh input papan catur
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')    # Output skor/aksi PPO
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if os.path.exists(MODEL_PATH):
    print("Model ditemukan. Memuat bobot dari file...")
    model = build_model()
    model.load_weights(MODEL_PATH)
else:
    print("File model tidak ditemukan. Membuat model baru dengan bobot acak...")
    model = build_model()

# ========================
# 3. Simulasi PPO Training Loop Sederhana
# ========================
for episode in range(10):  # Ganti ke jumlah episode sesuai kebutuhan
    dummy_board = np.random.random((8, 8, 12))  # Simulasi papan catur acak
    dummy_board = np.expand_dims(dummy_board, axis=0)
    
    prediction = model.predict(dummy_board)
    print(f"Episode {episode+1}, Prediction: {prediction}")

# ========================
# 4. Simpan Bobot Model
# ========================
model.save_weights(MODEL_PATH)
print(f"Model disimpan ke: {MODEL_PATH}")