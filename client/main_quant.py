import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
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
        'max_samples': 20000,
        'test_size': 0.2,
        'epochs': 50,
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
    model = trainer.create_quantized_model(
        input_dim=X_train.shape[1],
        num_classes=len(preprocessor.attack_type_map),
        architecture=config['model_architecture']
    )
    print("@" * 50)
    print(X_train.shape[1])
    print("@" * 50)
    model.save(os.path.join(save_dir, "model_arch.h5"))
    print(f"Model architecture saved at {os.path.join(save_dir, 'model_arch.h5')}")

    ####################################################################################
    print("\nApplying quantization-aware training...")
    
    # FIXED: Use the correct TensorFlow Model Optimization API
    quantize_model = tfmot.quantization.keras.quantize_model
    
    # Apply quantization-aware training to the model
    qat_model = quantize_model(model)
    
    # Compile the quantized model
    qat_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    
    print("\nTraining quantization-aware MLP model...")
    use_dp = True
    l2_norm_clip = 1.5
    noise_multiplier = 0.7
    microbatches = 1

    if load_model_weights(qat_model, save_dir):
        print("Weights loaded successfully for QAT model.")
    else:
        print("Failed to load weights. Training QAT model from scratch.")

    history, training_time = trainer.train_model(
    X_train, y_train_cat, X_test, y_test_cat,
    model=qat_model,
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    verbose=2,
    use_dp=use_dp,
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    microbatches=microbatches
    )

    print("Quantization-aware training complete.")
    qat_model.summary()

    ######################################################################################

    def evaluate_tflite_model(tflite_model, X_test, y_test_cat):
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct_predictions = 0
        total_loss = 0.0
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        for i in range(len(X_test)):
            input_data = X_test[i:i+1].astype(np.float32)  # Shape: (1, input_dim)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            true_class = np.argmax(y_test_cat[i])
            if predicted_class == true_class:
                correct_predictions += 1
            true_label = y_test_cat[i:i+1]
            loss = loss_fn(true_label, output_data).numpy()
            total_loss += loss

        accuracy = correct_predictions / len(X_test)
        average_loss = total_loss / len(X_test)
        return {'accuracy': accuracy, 'loss': average_loss}

    qat_loss, qat_accuracy = qat_model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Quantization-Aware Model - Loss: {qat_loss:.4f}, Accuracy: {qat_accuracy:.4f}")

    # Step 5: Convert to TFLite with full integer quantization
    print("\nConverting to TFLite with full integer quantization...")
    def representative_dataset():
        for i in range(100):  # Use a small subset of data
            yield [X_test[i:i+1].astype(np.float32)]  # Shape: (1, input_dim)

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # INT8 inputs
    converter.inference_output_type = tf.int8  # INT8 outputs
    converter.representative_dataset = representative_dataset

    try:
        tflite_model = converter.convert()
        with open(os.path.join(save_dir, 'quantized_qat_model.tflite'), 'wb') as f:
            f.write(tflite_model)
        print(f"Quantized TFLite model saved at {os.path.join(save_dir, 'quantized_qat_model.tflite')}")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")

    # Step 6: Evaluate the quantized TFLite model
    tflite_eval_results = evaluate_tflite_model(tflite_model, X_test, y_test_cat)
    print(f"Quantized TFLite Model - Accuracy: {tflite_eval_results['accuracy']:.4f}")
    print(f"Quantized TFLite Model - Loss: {tflite_eval_results['loss']:.4f}")

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env.client')
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        print("Client ID environment variable is missing.")
        raise ValueError("Missing required environment variable: CLIENT_ID")
    else:
        print(f"Client ID: {client_id}")
        main(client_id)