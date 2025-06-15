import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.preprocessing import IoTDataPreprocessor
from client.model import IoTModel

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# ----------------- Config --------------------
WEIGHT_SAVE_DIR = "ae_data/weights"
AE_MODEL_DIR = "ae_data/models"
NUM_SYNTHETIC_MODELS = 100
EPOCHS_RANGE = (1, 3)
ENCODING_DIM = 128

os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
os.makedirs(AE_MODEL_DIR, exist_ok=True)

# ----------------- Data Load --------------------
preprocessor = IoTDataPreprocessor(random_state=42)
csv_path = "DATA/data_part_*.csv"
csv_files = [f for f in os.listdir(os.path.dirname(csv_path)) if f.startswith("data_part_")]
if not csv_files:
    raise FileNotFoundError("No dataset found for synthetic generation.")
csv_file = os.path.join(os.path.dirname(csv_path), csv_files[0])
X, y = preprocessor.load_data(csv_file)

(X_train, _, y_train, _, y_train_cat, _, stats) = preprocessor.preprocess_data(
    X, y, max_samples=5000, test_size=0.2
)

# ----------------- Synthetic Weight Generation --------------------
def generate_weights():
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    num_classes = y_train_cat.shape[1]
    print(f"Number of classes: {num_classes}")
    for i in range(NUM_SYNTHETIC_MODELS):
        model = IoTModel().create_mlp_model(input_dim, num_classes)
        indices = np.random.choice(len(X_train), size=1000)
        x_sample, y_sample = X_train.iloc[indices], y_train_cat[indices]
        model.fit(x_sample, y_sample, epochs=np.random.randint(*EPOCHS_RANGE), batch_size=64, verbose=0)

        weights = model.get_weights()
        flat = np.concatenate([w.flatten() for w in weights])
        np.save(os.path.join(WEIGHT_SAVE_DIR, f"weights_{i}.npy"), flat)

generate_weights()
print(f"Generated {NUM_SYNTHETIC_MODELS} synthetic weight files at '{WEIGHT_SAVE_DIR}'.")

# ----------------- Autoencoder Model --------------------
def build_autoencoder(input_dim, encoding_dim=ENCODING_DIM):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(512, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(512, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
    return autoencoder, encoder

# ----------------- Autoencoder Training --------------------
def train_autoencoder():
    weight_files = [os.path.join(WEIGHT_SAVE_DIR, f) for f in os.listdir(WEIGHT_SAVE_DIR) if f.endswith(".npy")]
    weight_data = np.array([np.load(f) for f in weight_files])
    X_train, X_test = train_test_split(weight_data, test_size=0.1, random_state=42)

    input_dim = X_train.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, ENCODING_DIM)

    autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, validation_data=(X_test, X_test), verbose=1)
    autoencoder.save(os.path.join(AE_MODEL_DIR, "autoencoder.h5"))
    encoder.save(os.path.join(AE_MODEL_DIR, "encoder.h5"))
    print("Autoencoder and encoder saved.")

train_autoencoder()
