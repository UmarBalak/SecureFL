import os
import time
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import glob
import re
import logging
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.preprocessing import IoTDataPreprocessor
from client.model import IoTModel
from client.training import IoTModelTrainer
from client.evaluation import IoTModelEvaluator
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.client')

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

if not CLIENT_ACCOUNT_URL:
    raise ValueError("Missing required environment variable: Account url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
except Exception as e:
    print(f"Failed to initialize Azure Blob Service: {e}")
    raise

path = "/home/umarb/SecureFL/client/DATA"
script_directory = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(script_directory, "models")

from he.phe_provider import PHEProvider
from he.fhe_provider import FHEProvider

phe_provider = PHEProvider.generate()
fhe_provider = FHEProvider.generate()

def quantize_to_float(weights):
    """
    Quantizes weights to 6-bit floating-point precision (float-to-float, 6 decimal digits).
    """
    quantized_weights = []
    for w in weights:
        # Round to 6 decimal places to simulate 6-bit float precision
        quantized = np.round(w, decimals=7).astype(np.float32)
        quantized_weights.append(quantized)
    return quantized_weights

def count_decimal_places(weight_array):
    """
    Counts decimal places in each weight and returns max decimal places.
    """
    decimals = []
    for w in np.nditer(weight_array):
        s = f"{float(w):.20f}".rstrip('0')
        if '.' in s:
            decimals.append(len(s.split('.')[1]))
        else:
            decimals.append(0)
    return max(decimals, default=0)

def upload_file(file_path, container_name, metadata):
    filename = os.path.basename(file_path)
    print(f"Uploading weights ({filename}) to Azure Blob Storage...")
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=filename)
        with open(file_path, "rb") as file:
            blob_client.upload_blob(file.read(), overwrite=True, metadata=metadata)
        print(f"Weights ({filename}) uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        print(f"Error uploading weights ({filename}): {e}")

def save_run_info(config, stats, model_info, eval_results):
    run_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': config,
        'preprocessing_stats': stats,
        'model_info': model_info,
        'evaluation_results': {k: v for k, v in eval_results.items()
                              if k != 'confusion_matrix' and k != 'per_class_metrics'}
    }
    os.makedirs('logs', exist_ok=True)
    with open('logs/run_summary.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    print("Run information saved to 'logs/run_summary.json'")

def find_csv_file(file_pattern):
    matching_files = glob.glob(file_pattern)
    if matching_files:
        print(f"Found dataset: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"No dataset found matching pattern: {file_pattern}")
        return None

def wait_for_csv(file_pattern, wait_time=300):
    print(f"Checking for dataset matching pattern: {file_pattern}")
    while True:
        csv_file = find_csv_file(file_pattern)
        if csv_file:
            return csv_file
        print(f"Dataset not found. Waiting for {wait_time // 60} minutes...")
        time.sleep(wait_time)
        print(f"Rechecking for dataset matching pattern: {file_pattern}")

def load_model_weights(model, directory_path):
    try:
        h5_weights_file = next((file for file in glob.glob(os.path.join(directory_path, "weights.h5"))), None)
        if h5_weights_file:
            model.load_weights(h5_weights_file)
            print(f"Successfully loaded weights from {h5_weights_file}")
            return True
        print(f"No .h5 files found in {directory_path}")
        return False
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return False

def get_versioned_filename(client_id, save_dir, extension=".pkl"):
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
    os.makedirs(save_dir, exist_ok=True)
    weights_path, next_version, timestamp = get_versioned_filename(client_id, save_dir, extension="h5")
    try:
        model.save_weights(weights_path)
        print(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        print(f"Failed to save weights for {client_id}: {e}")
    return weights_path, timestamp

def encrypt_weights(weights, provider, provider_type="fhe"):
    logging.info(f"Encrypting weights with {provider_type.upper()}Provider...")
    result = []
    for i, weight_tensor in enumerate(weights):
        logging.info(f"Encrypting tensor {i+1}/{len(weights)}")
        encrypted_tensor, original_shape = provider.encrypt_tensor(weight_tensor)
        logging.info(f"Encrypted tensor {i+1}: shape={original_shape}")
        result.append((encrypted_tensor, original_shape))
    return result

def save_weights_with_encryption(client_id, model, save_dir, provider, encryption_type="fhe"):
    logging.info("Initializing encryption...")
    os.makedirs(save_dir, exist_ok=True)
    weights_path, next_version, timestamp = get_versioned_filename(client_id, save_dir, extension="pkl")
    try:
        logging.info("Getting model weights...")
        model_weights = model.get_weights()
        logging.info("Model weights retrieved successfully.")
        logging.info(f"Model weights shape: {[w.shape for w in model_weights]}")
        logging.info(f"Encrypting weights using {encryption_type}...")
        encrypted_weights = encrypt_weights(model_weights, provider, encryption_type)
        logging.info("Weights encrypted successfully.")
        logging.info(f"Saving encrypted weights to {weights_path}...")
        with open(weights_path, "wb") as f:
            pickle.dump(encrypted_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Encrypted weights ({encryption_type}) for {client_id} saved at {weights_path}")
        metadata_path = weights_path.replace(".pkl", "_metadata.json")
        metadata = {
            "client_id": client_id,
            "version": next_version,
            "timestamp": timestamp,
            "encryption_type": encryption_type,
            "model_architecture": [w.shape for w in model_weights]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save encrypted weights for {client_id}: {e}")
        raise
    return weights_path, timestamp

def load_encrypted_weights(weights_path, provider, encryption_type="fhe"):
    logging.info(f"Loading encrypted weights from {weights_path}...")
    try:
        with open(weights_path, "rb") as f:
            encrypted_weights = pickle.load(f)
        decrypted_weights = []
        for i, (enc_tensor, original_shape) in enumerate(encrypted_weights):
            logging.info(f"Decrypting tensor {i+1}/{len(encrypted_weights)}")
            decrypted_tensor = provider.decrypt_tensor(enc_tensor, original_shape)
            decrypted_weights.append(decrypted_tensor)
        logging.info("Weights decrypted successfully.")
        return decrypted_weights
    except Exception as e:
        logging.error(f"Failed to load encrypted weights: {e}")
        raise

def save_model(client_id, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"weights.h5")
    try:
        model.save(model_path)
        print(f"Model for {client_id} saved at {model_path}")
    except Exception as e:
        print(f"Failed to save model for {client_id}: {e}")
    return model_path

def main(client_id):
    config = {
        'data_path_pattern': f"{path}/data_part_*.csv",
        'max_samples': 5000,
        'test_size': 0.2,
        'epochs': 1,
        'batch_size': 64,
        'random_state': 42,
        'model_architecture': [256, 128, 128, 64],
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
    preprocessor = IoTDataPreprocessor(random_state=config['random_state'])
    X, y = preprocessor.load_data(config['data_path'])
    num_examples = len(X)
    print("Data loaded successfully")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X columns: {X.columns.tolist()}")
    preprocessing_start_time = time.time()
    (X_train, X_test, y_train, y_test,
     y_train_cat, y_test_cat, stats) = preprocessor.preprocess_data(
        X, y, max_samples=config['max_samples'], test_size=config['test_size']
    )
    preprocessing_time = time.time() - preprocessing_start_time
    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],
        num_classes=len(preprocessor.attack_type_map),
        architecture=config['model_architecture']
    )
    model.save(os.path.join(save_dir, "model_arch.h5"))
    print(f"Model architecture saved at {os.path.join(save_dir, 'model_arch.h5')}")
    print("\nTraining MLP model...")
    use_dp = True
    l2_norm_clip = 1.5
    noise_multiplier = 0.7
    microbatches = 1
    if load_model_weights(model, save_dir):
        print("Weights loaded successfully.")
    else:
        print("Failed to load weights. Training from scratch.")
    history, training_time = trainer.train_model(
        X_train, y_train_cat, X_test, y_test_cat,
        model=model,
        epochs=config['epochs'],
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
    try:
        if history and hasattr(history, 'history') and 'loss' in history.history:
            final_loss = history.history['loss'][-1]
        else:
            final_loss = None
    except Exception as e:
        print(f"Error getting final loss: {e}")
        final_loss = None
    metadata = {
        'num_examples': str(num_examples),
        'loss': str(final_loss) if final_loss is not None else "unknown",
    }
    print("\nEvaluating model on original weights...")
    evaluator = IoTModelEvaluator(preprocessor.attack_type_map)
    eval_results = evaluator.evaluate_model(model, X_test, y_test, y_test_cat)
    original_loss, original_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Original Weights - Loss: {original_loss:.4f}, Accuracy: {original_accuracy:.4f}")

    # Quantize to 6-bit float
    print("\nQuantizing weights to 6-decimal float...")
    original_weights = model.get_weights()
    quantized_weights = quantize_to_float(original_weights)
    model.set_weights(quantized_weights)
    print("Evaluating model on quantized weights...")
    eval_results_quantized = evaluator.evaluate_model(model, X_test, y_test, y_test_cat)
    quantized_loss, quantized_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Quantized Weights - Loss: {quantized_loss:.4f}, Accuracy: {quantized_accuracy:.4f}")

    # Encrypt and save quantized weights
    encryption_type = "fhe"
    print("\nEncrypting quantized weights...")
    weights_path, timestamp = save_weights_with_encryption(client_id, model, save_dir, fhe_provider, encryption_type)
    # upload_file(weights_path, CLIENT_CONTAINER_NAME, metadata)

    # Decrypt and compare
    print("\nDecrypting weights...")
    decrypted_weights = load_encrypted_weights(weights_path=weights_path, provider=fhe_provider, encryption_type="fhe")

    # Compare original (quantized) and decrypted weights
    print("\nComparing quantized and decrypted weights...")
    for i, (quantized, decrypted) in enumerate(zip(quantized_weights, decrypted_weights)):
        quantized_np = quantized.numpy() if isinstance(quantized, tf.Tensor) else quantized
        decrypted_np = decrypted.numpy() if isinstance(decrypted, tf.Tensor) else decrypted
        if np.allclose(quantized_np, decrypted_np, rtol=1e-5, atol=1e-6):
            print(f"Tensor {i+1} decrypted successfully and matches quantized weights.")
        else:
            diff = np.abs(quantized_np - decrypted_np)
            max_diff = np.max(diff)
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Tensor {i+1} mismatch after decryption.")
            print(f"Max difference: {max_diff}")
            print(f"Quantized value at max difference: {quantized_np[idx]}")
            print(f"Decrypted value at max difference: {decrypted_np[idx]}")

    model_info = {
        'training_time': training_time,
        'parameters': model.count_params(),
        'layers': len(model.layers)
    }
    save_run_info(config, stats, model_info, eval_results)
    print(f"\n{'='*70}")
    print(f"SecureFL - Run Complete")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Original accuracy: {eval_results['accuracy']:.4f}")
    print(f"Quantized accuracy: {eval_results_quantized['accuracy']:.4f}")
    print(f"Preprocessing time: {preprocessing_time:.2f} seconds")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env.client')
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        print("Client ID environment variable is missing.")
        raise ValueError("Missing required environment variable: CLIENT_ID")
    else:
        print(f"Client ID: {client_id}")
        main(client_id)