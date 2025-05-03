import os
import time
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import glob
import re

# Import custom modules
from preprocessing import IoTDataPreprocessor
from model import IoTModel
from training import IoTModelTrainer
from evaluation import IoTModelEvaluator

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

if not CLIENT_ACCOUNT_URL:
    print("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
except Exception as e:
    print(f"Failed to initialize Azure Blob Service: {e}")
    raise

path = "D:\FL\client\DATA"
script_directory = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(script_directory, "models")

def upload_file(file_path, container_name, metadata):
    """
    Upload a file to Azure Blob Storage with versioned naming.

    Args:
        client_id (str): ID of the client.
        file_path (str): Path to the file to upload.
        container_name (str): Azure container name.
    """
    filename = os.path.basename(file_path)
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=filename)
        with open(file_path, "rb") as file:
            blob_client.upload_blob(file.read(), overwrite=True, metadata=metadata)
        print(f"File {filename} uploaded successfully to Azure Blob Storage.")
        print(f"File {filename} uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        print(f"Error uploading file {filename}: {e}")
        print(f"Error uploading file {filename}: {e}")

def save_run_info(config, stats, model_info, eval_results):
    """Save run information to JSON file"""
    run_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': config,
        'preprocessing_stats': stats,
        'model_info': model_info,
        'evaluation_results': {k: v for k, v in eval_results.items()
                              if k != 'confusion_matrix' and k != 'per_class_metrics'}
    }

    # Save to file
    os.makedirs('logs', exist_ok=True)
    with open('logs/run_summary.json', 'w') as f:
        json.dump(run_info, f, indent=2)

    print("Run information saved to 'logs/run_summary.json'")

def find_csv_file(file_pattern):
    """
    Find a CSV file matching the given pattern.

    Args:
        file_pattern (str): The file pattern to search for (e.g., "data_part_*.csv").

    Returns:
        str: The path to the first matching file, or None if no file is found.
    """
    matching_files = glob.glob(file_pattern)
    if matching_files:
        print(f"Found dataset: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"No dataset found matching pattern: {file_pattern}")
        return None

def wait_for_csv(file_pattern, wait_time=300):
    """
    Wait for a CSV file matching the given pattern to appear.

    Args:
        file_pattern (str): The file pattern to search for (e.g., "data_part_*.csv").
        wait_time (int): Time to wait (in seconds) before rechecking.
    """
    print(f"Checking for dataset matching pattern: {file_pattern}")
    while True:
        csv_file = find_csv_file(file_pattern)
        if csv_file:
            return csv_file
        print(f"Dataset not found. Waiting for {wait_time // 60} minutes...")
        time.sleep(wait_time)
        print(f"Rechecking for dataset matching pattern: {file_pattern}")

def load_model_weights(model, directory_path):
    """
    Load the first .h5 weights file found in the specified directory.

    Args:
        model: Keras model instance.
        directory_path: Path to directory containing weights file.

    Returns:
        bool: True if weights loaded successfully, False otherwise.
    """
    try:
        # Search for .h5 files in the directory
        keras_file = next((file for file in glob.glob(os.path.join(directory_path, "weights.h5"))), None)
        
        if keras_file:
            model.load_weights(keras_file)
            print(f"Successfully loaded weights from {keras_file}")
            return True
        
        print(f"No .h5 files found in {directory_path}")
        return False

    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return False

def get_versioned_filename(client_id, save_dir, extension="keras"):
    """
    Generate a versioned filename with timestamp for saving models or weights.

    Args:
        client_id (str): ID of the client.
        save_dir (str): Directory to save files.
        prefix (str): Prefix for the file (e.g., 'weights', 'model').
        extension (str): File extension (e.g., 'keras').

    Returns:
        str: Full path to the versioned filename.
        int: Next version number.
        str: Timestamp for the file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_pattern = re.compile(rf"client{client_id}_v(\d+).*\.{extension}")
    existing_versions = [
        int(version_pattern.match(f).group(1))
        for f in os.listdir(save_dir)
        if version_pattern.match(f)
    ]
    next_version = max(existing_versions, default=0) + 1
    filename = f"client{client_id}_v{next_version}_{timestamp}.{extension}"
    return os.path.join(save_dir, filename), next_version, timestamp

def save_weights(client_id, model, save_dir):
    """
    Save model weights with versioning.
    """
    os.makedirs(save_dir, exist_ok=True)

    weights_path, next_version, timestamp = get_versioned_filename(client_id, save_dir)
    try:
        model.save_weights(weights_path)
        print(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        print(f"Failed to save weights for {client_id}: {e}")
    return weights_path, timestamp


def save_model(client_id, model, save_dir):
    """
    Save the trained model with versioning.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"weights.h5")
    try:
        model.save(model_path)
        print(f"Model for {client_id} saved at {model_path}")
    except Exception as e:
        print(f"Failed to save model for {client_id}: {e}")
    return model_path
    
def main(client_id):
    """Main function to run the entire pipeline"""
    # Configuration settings
    config = {
        'data_path_pattern': f"{path}/data_part_*.csv",  # Path to dataset
        'max_samples': 30000,                                # Set to a number to limit samples
        'test_size': 0.2,                                    # Test split proportion
        'epochs': 50,                                        # Max training epochs
        'batch_size': 32,                                   # Training batch size
        'random_state': 42,                                  # For reproducibility
        'model_architecture':  [128, 64],                    # Units per hidden layer
    }

    # Check if the dataset exists, wait if not
    config['data_path'] = wait_for_csv(config['data_path_pattern'])

    # Set random seeds for reproducibility
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])

    # Create directory structure
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SecureFL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = IoTDataPreprocessor(random_state=config['random_state'])

    # Load and preprocess data
    X, y = preprocessor.load_data(config['data_path'])
    num_examples = len(X)
    print("Data loaded successfully")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X columns: {X.columns.tolist()}")

    # Preprocess data
    preprocessing_start_time = time.time()
    (X_train, X_test, y_train, y_test,
     y_train_cat, y_test_cat, stats) = preprocessor.preprocess_data(
        X, y, max_samples=config['max_samples'], test_size=config['test_size']
    )
    preprocessing_time = time.time() - preprocessing_start_time

    # Initialize trainer
    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],
        num_classes=len(preprocessor.attack_type_map),
        architecture=config['model_architecture']
    )

    print("\nTraining MLP model...")
    
    if load_model_weights(model, save_dir):
        print("Weights loaded successfully.")
    else:
        print("Failed to load weights. Training from scratch.")
    history, training_time = trainer.train_model(
        X_train, y_train_cat, X_test, y_test_cat,
        model=model,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=2
    )
    model = trainer.get_model()
    try:
        final_loss = history.history['loss'][-1]
    except Exception as e:
        print(f"Error getting final loss: {e}")
        final_loss = 0

    # Prepare metadata
    metadata = {
        'num_examples': str(num_examples),
        'loss': str(final_loss),
    }
    
    # Save weights
    model.save_weights(os.path.join(save_dir, "weights.h5"))
    weights_path, timestamp = save_weights(client_id, model, save_dir)
    upload_file(weights_path, CLIENT_CONTAINER_NAME, metadata)


    # Evaluate model
    print("\nEvaluating model...")
    evaluator = IoTModelEvaluator(preprocessor.attack_type_map)
    eval_results = evaluator.evaluate_model(model, X_test, y_test, y_test_cat)

    # Generate predictions for visualization
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Prepare model information
    model_info = {
        'training_time': training_time,
        'parameters': model.count_params(),
        'layers': len(model.layers)
    }

    # Save run information
    save_run_info(config, stats, model_info, eval_results)

    print(f"\n{'='*70}")
    print(f"SecureFL - Run Complete")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model accuracy: {eval_results['accuracy']:.4f}")
    print(f"Preprocessing time: {preprocessing_time:.2f} seconds")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env')
    client_id = os.getenv("CLIENT_ID")
    main(client_id)