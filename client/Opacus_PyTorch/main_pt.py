import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime
import threading
from statistics import mean
from itertools import product
import gc
import glob
import re
import logging
import pickle
import sys
import time, json, csv, psutil
from itertools import product
import pandas as pd

# PyTorch DP imports
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants.utils import get_noise_multiplier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import IoTDataPreprocessor  # Reuse your TF preprocessor
from model_pt import IoTModelPyTorch
from training_pt import IoTModelTrainerPyTorch
from azure.storage.blob import BlobServiceClient

from functions import wait_for_csv, upload_file, save_run_info, find_csv_file

# ======== PATH SELECTION MECHANISM ========
def get_path_mode():
    mode = input("Select path mode ('colab' or 'local'): ").strip().lower()
    if mode not in ["colab", "local"]:
        print("Invalid input. Defaulting to 'colab'.")
        mode = "colab"
    return mode

def get_dp_mechanism():
    """Get user's choice for DP mechanism"""
    print("\nDifferential Privacy Mechanisms:")
    print("1. Laplace DP (Default)")
    print("2. Gaussian DP") 
    print("3. No DP")
    
    choice = input("Select DP mechanism (1/2/3, default=1): ").strip()
    
    if choice == "2":
        return "gaussian"
    elif choice == "3":
        return "none"
    else:
        return "laplace"

PATH_MODE = get_path_mode()

if PATH_MODE == "colab":
    DATASET_PATH = "/content/drive/MyDrive/SecureFL/SecureFL_client/ML-EdgeIIoT-dataset.csv"
    DOTENV_PATH = "/content/drive/MyDrive/SecureFL/SecureFL_client/.env.client"
    SAVE_DIR = "/content/drive/MyDrive/SecureFL/mlp_models_for_ae_pytorch"
else:
    DATASET_PATH = "./DATA/ML-EdgeIIoT-dataset.csv"
    DOTENV_PATH = ".env.client"
    script_directory = os.path.dirname(os.path.realpath(__file__))
    SAVE_DIR = os.path.join(script_directory, "pytorch_models")

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

from sklearn.model_selection import train_test_split
import random

# Define epoch categories
def get_training_epochs():
    return 75 

def convert_tf_to_pytorch_data(X_train, X_test, y_train, y_test):
    """
    Convert TensorFlow/NumPy data to PyTorch tensors
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and test features
    y_train, y_test : numpy.ndarray
        Training and test labels
        
    Returns:
    --------
    PyTorch DataLoaders for training and testing
    """
    print("Converting data from TensorFlow format to PyTorch...")
    
    # Convert to PyTorch tensors
    X_train_pt = torch.FloatTensor(X_train)
    X_test_pt = torch.FloatTensor(X_test) 
    y_train_pt = torch.LongTensor(y_train)
    y_test_pt = torch.LongTensor(y_test)
    
    print(f"PyTorch - Train shape: {X_train_pt.shape}, Test shape: {X_test_pt.shape}")
    print(f"PyTorch - Train labels: {y_train_pt.shape}, Test labels: {y_test_pt.shape}")
    
    return X_train_pt, X_test_pt, y_train_pt, y_test_pt

def main(client_id, epochs, X_train, X_test, y_train, y_test, y_type, num_classes, l):
    config = {
        'data_path_pattern': DATASET_PATH,
        'test_size': 0.2,
        'epochs': epochs,
        'batch_size': 64,
        'random_state': 42,
        'model_architecture': [256, 128, 64], # Same as TensorFlow version
    }
    
    config['data_path'] = wait_for_csv(config['data_path_pattern'])
    
    # Set random seeds for reproducibility (PyTorch version)
    torch.manual_seed(config['random_state'])
    np.random.seed(config['random_state'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_state'])
        torch.cuda.manual_seed_all(config['random_state'])
    
    for directory in ['pytorch_models', 'pytorch_logs', 'pytorch_plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)
        
    print(f"\n{'='*70}")
    print(f"SecureFL - PyTorch Implementation with Laplace DP")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

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

    # Convert data to PyTorch format
    X_train_pt, X_test_pt, y_train_pt, y_test_pt = convert_tf_to_pytorch_data(
        X_train, X_test, y_train, y_test
    )

    trainer = IoTModelTrainerPyTorch(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        architecture=config['model_architecture']
    )
    print("@" * 50)
    print(X_train.shape[1])
    print("@" * 50)

    print("\nTraining MLP model with PyTorch...")
    
    # Get DP mechanism choice from user
    dp_mechanism = get_dp_mechanism()
    use_dp = dp_mechanism in ['gaussian', 'laplace']
    
    # DP Configuration
    max_grad_norm = 1.0  # L2 clipping for Gaussian, L1 clipping for Laplace
    target_epsilon = 1.0
    target_delta = 1e-5
    
    # Laplace DP specific parameters
    laplace_sensitivity = 1.0
    
    # Gaussian DP specific parameters  
    noise_multiplier = 0.7

    # Create data loaders
    train_dataset = TensorDataset(X_train_pt, y_train_pt)
    test_dataset = TensorDataset(X_test_pt, y_test_pt)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    if dp_mechanism == 'laplace':
        print(f"Using Laplace DP with target_epsilon={target_epsilon}, sensitivity={laplace_sensitivity}")
        history, training_time, epsilon_spent, delta_spent, final_accuracy = trainer.train_model(
            train_loader, test_loader,
            model=model,
            epochs=config['epochs'],
            verbose=2,
            use_dp=use_dp,
            dp_mechanism='laplace',
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            laplace_sensitivity=laplace_sensitivity
        )
    elif dp_mechanism == 'gaussian':
        print(f"Using Gaussian DP with target_epsilon={target_epsilon}, noise_multiplier={noise_multiplier}")
        history, training_time, epsilon_spent, delta_spent, final_accuracy = trainer.train_model(
            train_loader, test_loader,
            model=model,
            epochs=config['epochs'],
            verbose=2,
            use_dp=use_dp,
            dp_mechanism='gaussian',
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            noise_multiplier=noise_multiplier
        )
    else:
        print("Training without differential privacy")
        history, training_time, epsilon_spent, delta_spent, final_accuracy = trainer.train_model(
            train_loader, test_loader,
            model=model,
            epochs=config['epochs'],
            verbose=2,
            use_dp=False
        )

    model = trainer.get_model()
    print("Model training complete.")
    
    # Save model (PyTorch format)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"weights.pth"))
    print(f"Model weights saved to {os.path.join(SAVE_DIR, 'weights.pth')}")
    
    # Print final privacy guarantees
    if use_dp:
        print(f"\n🔒 Final Privacy Guarantee:")
        print(f"   Mechanism: {dp_mechanism.upper()}")
        print(f"   Epsilon (ε): {epsilon_spent:.4f}")
        print(f"   Delta (δ): {delta_spent:.2e}")
        print(f"   Final Accuracy: {final_accuracy:.4f}")
        if dp_mechanism == 'laplace':
            print(f"   Sensitivity: {laplace_sensitivity}")
        elif dp_mechanism == 'gaussian':
            print(f"   Max grad norm: {max_grad_norm}")
    
    # Save results for comparison with TensorFlow version
    results = {
        'mechanism': dp_mechanism,
        'epsilon': float(epsilon_spent) if use_dp else float('inf'),
        'delta': float(delta_spent) if use_dp else 0.0,
        'final_accuracy': float(final_accuracy),
        'training_time': float(training_time),
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'architecture': config['model_architecture'],
        'dataset_info': {
            'train_samples': len(X_train_pt),
            'test_samples': len(X_test_pt),
            'num_classes': num_classes,
            'input_dim': X_train.shape[1]
        }
    }
    
    results_file = os.path.join(SAVE_DIR, f"pytorch_{dp_mechanism}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

import os
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv(dotenv_path=DOTENV_PATH)
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        raise ValueError("Missing required environment variable: CLIENT_ID")
    print(f"Client ID: {client_id}")

    # Reuse your existing TensorFlow preprocessor
    preprocessor = IoTDataPreprocessor()
    X_train, X_test, y_train, y_test, y_type, num_classes, l = preprocessor.preprocess_data(
        DATASET_PATH
    )

    try:
        inp = int(input("Enter the number of epochs for training (Default=75): ").strip())
        epochs = inp
    except ValueError:
        print("Invalid input. Using default value: 75")
        epochs = get_training_epochs()

    main(client_id, epochs, X_train, X_test, y_train, y_test, y_type, num_classes, l)