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

from functions import wait_for_csv, load_model_weights, upload_file, save_run_info, find_csv_file, save_weights


DATASET_PATH = "./DATA/train.csv"
TEST_DATASET_PATH = "./DATA/global_test.csv"
VAL_DATASET_PATH = "./DATA/val.csv"
DOTENV_PATH = ".env.client"
script_directory = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(script_directory, "models")


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
    # p = random.random()
    # if p < 0.2:
    #     return random.randint(20, 50)          # Undertrained
    # elif p < 0.8:
    #     return random.randint(70, 100)         # Well-trained
    # else:
    #     return random.randint(120, 150)        # Overtrained

    return 100
    

def evaluate_model(model, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat):
    """
    Evaluates the model on the training, validation, and test datasets.
    """
    print("\n" + "="*50)
    print("Evaluating model performance on all datasets...")
    
    # Evaluate on training data
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    print(f"✅ Training Set - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Evaluate on validation data
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"✅ Validation Set - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"✅ Test Set - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    print("="*50 + "\n")


def main(client_id, epochs):
    config = {
        'data_path_pattern': DATASET_PATH,
        'test_data_path_pattern': TEST_DATASET_PATH,
        'val_data_path_pattern': VAL_DATASET_PATH,
        'epochs': epochs,
        'batch_size': 512,
        'random_state': 42,
        'model_architecture': [1024, 512, 256],
    }
    config['data_path'] = wait_for_csv(config['data_path_pattern'])
    config['test_data_path'] = wait_for_csv(config['test_data_path_pattern'])
    config['val_data_path'] = wait_for_csv(config['val_data_path_pattern'])
    
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)
        
    print(f"\n{'='*70}")
    print(f"SecureFL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    preprocessing_start_time = time.time()
    preprocessor = IoTDataPreprocessor()

    ## only for feature selection
    # X_processed, y_processed, num_classes, feature_list = preprocessor.preprocess_data(
    #     config['data_path'],
    #     correlation_threshold=0.1,  # Higher threshold for large dataset
    #     multicollinearity_threshold=0.95,
    #     final_feature_count=25  # Optimal for 150K rows
    #     )

    ## after feature selection - Process training data
    X_train, y_train, num_classes = preprocessor.preprocess_data(
        config['data_path'],
        apply_smote=True  # Apply SMOTE only on training data
        )
    
    # Process test data separately
    print("Loading and preprocessing test data...")
    X_test, y_test, _ = preprocessor.preprocess_data(
        config['test_data_path'],
        apply_smote=False  # Ensure no SMOTE on test data
        )
    
    # Process val data separately
    print("Loading and preprocessing test data...")
    X_val, y_val, _ = preprocessor.preprocess_data(
        config['val_data_path'],
        apply_smote=False  # Ensure no SMOTE on test data
        )

    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        architecture=config['model_architecture']
    )
    print("@" * 50)
    print(f"Training data features: {X_train.shape[1]}")
    print(f"Test data features: {X_test.shape[1]}")
    print("@" * 50)

    use_dp = False
    l2_norm_clip = 1.0
    noise_multiplier = 0.7
    microbatches = 1

    # model.save(os.path.join(SAVE_DIR, "architecture/model_arch.h5"))
    # print(f"Model architecture saved at {os.path.join(save_dir, 'architecture/model_arch.h5')}")

    # if load_model_weights(model, save_dir):
    #     print("Weights loaded successfully.")
    # else:
    #     print("Failed to load weights. Training from scratch.")

    from tensorflow.keras.utils import to_categorical

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    history, training_time, noise_multiplier, l2_norm_clip, microbatches, epsilon_dict, delta, mem_start = trainer.train_model(
        X_train, y_train_cat, X_val, y_val_cat, # Pass val set for validation
        model=model,
        architecture=config['model_architecture'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=2,
        use_dp=False,
        l2_norm_clip=1.0,
        noise_multiplier=0.7,
        microbatches=1
    )
    
    model = trainer.get_model()
    print("Model training complete.")
    model.summary()
    
    # Evaluate model on all sets (train, val from training data + separate test data)
    evaluate_model(model, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat)

    weights_path, timestamp = save_weights(client_id, model, SAVE_DIR)

    # Prepare metadata for Azure upload
    metadata = {
        'timestamp': timestamp,
    }
    
    # Upload model and weights to Azure
    upload_file(weights_path, CLIENT_CONTAINER_NAME, metadata)

import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(dotenv_path=DOTENV_PATH)
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        raise ValueError("Missing required environment variable: CLIENT_ID")
    print(f"Client ID: {client_id}")

    try:
        epochs = int(input("Enter the number of epochs for training (Default=100): ").strip())
    except ValueError:
        print("Invalid input. Using default value: 100")
        epochs = get_training_epochs()

    main(client_id, epochs)