import os
import time
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import threading
from statistics import mean
from itertools import product
import threading
from statistics import mean
import gc
import glob
import re
import logging
import pickle
import sys
import time, json, csv, psutil
from itertools import product
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import IoTDataPreprocessor
from model import IoTModel
from training import IoTModelTrainer
from azure.storage.blob import BlobServiceClient

from functions import wait_for_csv, load_model_weights, upload_file, save_run_info, find_csv_file

# ======== PATH SELECTION MECHANISM ========
def get_path_mode():
    mode = input("Select path mode ('colab' or 'local'): ").strip().lower()
    if mode not in ["colab", "local"]:
        print("Invalid input. Defaulting to 'colab'.")
        mode = "colab"
    return mode

PATH_MODE = get_path_mode()

if PATH_MODE == "colab":
    DATASET_PATH = "/content/drive/MyDrive/SecureFL/SecureFL_client/ML-EdgeIIoT-dataset.csv"
    DOTENV_PATH = "/content/drive/MyDrive/SecureFL/SecureFL_client/.env.client"
    SAVE_DIR = "/content/drive/MyDrive/SecureFL/mlp_models_for_ae"
else:
    DATASET_PATH = "./DATA/ML-EdgeIIoT-dataset.csv"
    DOTENV_PATH = ".env.client"
    script_directory = os.path.dirname(os.path.realpath(__file__))
    SAVE_DIR = os.path.join(script_directory, "models")

# ======== END PATH SELECTION ========

from dotenv import load_dotenv
load_dotenv(dotenv_path=DOTENV_PATH)

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

if not CLIENT_ACCOUNT_URL:
    raise ValueError("Missing required environment variable: Account url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
except Exception as e:
    print(f"Failed to initialize Azure Blob Service: {e}")
    raise

# path = "/home/umarb/SecureFL/client/DATA"
# script_directory = os.path.dirname(os.path.realpath(__file__))
# save_dir = os.path.join(script_directory, "models")
# save_dir = "/content/drive/MyDrive/SecureFL/mlp_models_for_ae"

from sklearn.model_selection import train_test_split
import random

# Define epoch categories
def get_training_epochs():
    p = random.random()
    if p < 0.2:
        return random.randint(20, 50)          # Undertrained
    elif p < 0.8:
        return random.randint(70, 100)         # Well-trained
    else:
        return random.randint(120, 150)        # Overtrained
    
def main(client_id, model_num, epochs, X_train, X_test, y_train, y_test, y_type, num_classes, l):
    config = {
        'data_path_pattern': DATASET_PATH,
        # 'data_path_pattern': f"{path}/data_part_*.csv",
        'test_size': 0.2,
        'epochs': epochs,
        'batch_size': 64,
        'random_state': 42,
        # 'model_architecture': [256, 128, 128]
        # 'model_architecture': [256, 128, 64],
        'model_architecture': [256, 128, 64], # 17 -> 0.84, 0.4
    }
    config['data_path'] = wait_for_csv(config['data_path_pattern'])
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"SecureFL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # preprocessing_start_time = time.time()
    # preprocessor = IoTDataPreprocessor()
    # X_train, X_test, y_train, y_test, y_type, num_classes, l = preprocessor.preprocess_data(config['data_path'])

    #######################################################################
    ############# Split data into train/test sets and sample x% of entire data
    ############# Only to generate dataset for autoencoder training
    #######################################################################
    # Concatenate full dataset
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)

    train_sizes = [0.5, 0.6, 0.7, 0.8]

    # Randomly select one
    selected_train_size = random.choice(train_sizes)
    print(f"📌 Randomly selected train size: {selected_train_size}")

    X_sampled, _, y_sampled, _ = train_test_split(
        X_full, y_full,
        train_size=selected_train_size,
        stratify=y_full,
        shuffle=True
    )
    # Re-split that sampled data into new train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled,
        test_size=0.01,
        stratify=y_sampled,
        shuffle=True
    )
    # Done
    print(f"Sampled Train shape: {X_train.shape}, Test shape: {X_test.shape}, Classes: {np.unique(y_sampled)}")
    #######################################################################

    # preprocessing_time = time.time() - preprocessing_start_time
    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        architecture=config['model_architecture']
    )
    print("@" * 50)
    print(X_train.shape[1])
    print("@" * 50)
    # model.save(os.path.join(save_dir, "architecture/model_arch.h5"))
    # print(f"Model architecture saved at {os.path.join(save_dir, 'architecture/model_arch.h5')}")
    print("\nTraining MLP model...")
    use_dp = True
    l2_norm_clip = 1.0
    noise_multiplier = 0.7
    microbatches = 1
    # if load_model_weights(model, save_dir):
    #     print("Weights loaded successfully.")
    # else:
    #     print("Failed to load weights. Training from scratch.")

    from tensorflow.keras.utils import to_categorical

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    num_examples = X_train.shape[0]

    history, training_time, noise_multiplier, l2_norm_clip, microbatches, epsilon_dict, delta, mem_start = trainer.train_model(
        X_train, y_train_cat, X_test, y_test_cat,
        model=model,
        epochs=config['epochs'],
        # epochs=2,
        batch_size=config['batch_size'],
        verbose=2,
        use_dp=use_dp,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        microbatches=microbatches
    )
    model = trainer.get_model()
    print("Model training complete.")
    model.summary()
    model.save(os.path.join(SAVE_DIR, f"ae/weights{model_num}.h5"))

# if __name__ == "__main__":
#     load_dotenv(dotenv_path='.env.client')
#     client_id = os.getenv("CLIENT_ID")
#     if not client_id:
#         print("Client ID environment variable is missing.")
#         raise ValueError("Missing required environment variable: CLIENT_ID")
#     else:
#         print(f"Client ID: {client_id}")
#         preprocessor = IoTDataPreprocessor()
#         X_train, X_test, y_train, y_test, y_type, num_classes, l = preprocessor.preprocess_data(f"/content/drive/MyDrive/SecureFL/AE_Dataset_Scripts/ML-EdgeIIoT-dataset.csv")
#         import os
#         import re

#         # Path to saved models
#         model_dir = "/content/drive/MyDrive/SecureFL/mlp_models_for_ae"

#         # Regex to match filenames like weights23.h5
#         pattern = re.compile(r"weights(\d+)\.h5")

#         # Get all valid weight files
#         existing_files = [f for f in os.listdir(model_dir) if pattern.match(f)]

#         # Extract iteration numbers
#         existing_nums = [int(pattern.match(f).group(1)) for f in existing_files]

#         # Determine start index
#         start_iter = max(existing_nums) + 1 if existing_nums else 1
#         for i in range(start_iter, 101):
#             epochs = get_training_epochs()
#             main(client_id, i, epochs, X_train, X_test, y_train, y_test, y_type, num_classes, l)
#             print("="*20)
#             print(f"Iteration {i} completed with {epochs} epochs.")



import os
import re
import argparse
from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser(description="Train models from a specific iteration range.")
    parser.add_argument("--start", type=int, help="Start iteration number")
    parser.add_argument("--end", type=int, help="End iteration number (inclusive)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    load_dotenv(dotenv_path=DOTENV_PATH)
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        raise ValueError("Missing required environment variable: CLIENT_ID")
    print(f"Client ID: {client_id}")

    preprocessor = IoTDataPreprocessor()
    X_train, X_test, y_train, y_test, y_type, num_classes, l = preprocessor.preprocess_data(
        DATASET_PATH
    )

    # Fallback logic if --start is not given
    if args.start is not None and args.end is not None:
        start_iter = args.start
        end_iter = args.end
        print(f"Using custom iteration range: {start_iter} to {end_iter}")
    else:
        # Path to saved models
        model_dir = SAVE_DIR
        pattern = re.compile(r"weights(\d+)\.h5")
        existing_files = [f for f in os.listdir(model_dir) if pattern.match(f)]
        existing_nums = [int(pattern.match(f).group(1)) for f in existing_files]
        start_iter = max(existing_nums) + 1 if existing_nums else 1
        end_iter = 100
        print(f"Auto-detected iteration range: {start_iter} to {end_iter}")

    # Run training loop
    for i in range(start_iter, end_iter + 1):
        epochs = get_training_epochs()
        main(client_id, i, epochs, X_train, X_test, y_train, y_test, y_type, num_classes, l)
        print("=" * 20)
        print(f"Iteration {i} completed with {epochs} epochs.")
