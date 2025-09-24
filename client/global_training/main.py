import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.utils import to_categorical
import sys
import json
import time
from azure.storage.blob import BlobServiceClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import IoTDataPreprocessor
from training import IoTModelTrainer
from functions import upload_file, save_weights, wait_for_csv, find_csv_file
from evaluate import evaluate_model

from dotenv import load_dotenv
DOTENV_PATH = ".env.server"
load_dotenv(dotenv_path=DOTENV_PATH)

DATASET_PATH = "./DATA/global_train.csv"
TEST_DATASET_PATH = "./DATA/global_test.csv"
script_directory = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(script_directory, "models")

SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")

if not SERVER_ACCOUNT_URL:
    raise ValueError("Missing required environment variable: Account url")

try:
    BLOB_SERVICE_SERVER = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
except Exception as e:
    print(f"Failed to initialize Azure Blob Service: {e}")
    raise


def main(epochs=100):
    config = {
        'data_path_pattern': DATASET_PATH,
        'test_data_path_pattern': TEST_DATASET_PATH,
        'epochs': epochs,
        'batch_size': 512,
        'random_state': 42,
        'model_architecture': [1024, 512, 256],
    }
    config['data_path'] = wait_for_csv(config['data_path_pattern'])
    config['test_data_path'] = wait_for_csv(config['test_data_path_pattern'])
    
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)
        
    print(f"\n{'='*70}")
    print(f"SecureFL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    preprocessor = IoTDataPreprocessor()

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

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    history, training_time = trainer.train_model_without_validation(
        X_train, y_train_cat,
        model=model,
        architecture=config['model_architecture'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=2,
    )
    
    model = trainer.get_model()
    print("Model training complete.")
    model.summary()

    le = preprocessor.le_dict.get('Attack_type', None)
    class_names = le.classes_.tolist() if le else None
    
    eval_results = evaluate_model(model, X_train, y_train_cat, X_test, y_test_cat, class_names=class_names)
    train_metrics = eval_results['train']
    test_metrics = eval_results['test']

    weights_path = save_weights(model, SAVE_DIR, "g0.h5")

    # Prepare metadata for Azure upload
    metadata = {
        "final_train_loss": str(train_metrics['loss']),
        "final_train_accuracy": str(train_metrics['accuracy']),
        "final_train_precision": str(train_metrics['macro_precision']),
        "final_train_recall": str(train_metrics['macro_recall']),
        "final_train_f1": str(train_metrics['macro_f1']),
        
        "final_test_loss": str(test_metrics['loss']),
        "final_test_accuracy": str(test_metrics['accuracy']),
        "final_test_precision": str(test_metrics['macro_precision']),
        "final_test_recall": str(test_metrics['macro_recall']),
        "final_test_f1": str(test_metrics['macro_f1']),
        
        "epochs": str(epochs),
        "batch_size": str(config['batch_size']),
        "model_architecture": json.dumps(config['model_architecture'])
    }

    complete_metadata = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_architecture": config['model_architecture'],  # List is JSON serializable
        "training_params": {
            "epochs": epochs,
            "batch_size": config['batch_size'],
        }
    }

    # Save JSON metadata to file
    metadata_json_str = json.dumps(complete_metadata, indent=2)
    metadata_file_path = os.path.join(SAVE_DIR, "g0_metadata.json")
    with open(metadata_file_path, "w") as f:
        f.write(metadata_json_str)
    
    # Upload model and weights to Azure
    upload_file(weights_path, SERVER_CONTAINER_NAME, metadata)

    # Upload JSON metadata file as a separate blob
    upload_file(metadata_file_path, SERVER_CONTAINER_NAME, metadata={})

import os
from dotenv import load_dotenv

if __name__ == "__main__":
    main(epochs=100)