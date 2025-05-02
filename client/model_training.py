import tensorflow as tf
import warnings
# from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# import tensorflow_privacy as tfp
tf.get_logger().setLevel('ERROR')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import re
import glob
import numpy as np
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.public')

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

if not CLIENT_ACCOUNT_URL:
    logging.error("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise

# Set up logger
def setup_logger():
    """Set up logging for client."""
    script_directory = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(script_directory, "logs","training.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Load preprocessed data
def load_preprocessed_data(data_path):
    """Load preprocessed data for training from a .npz file."""
    data_file = os.path.join(data_path, "preprocessed_data.npz")

    # Load the .npz file
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found.")
    
    data = np.load(data_file)

    # Convert npz data to a dictionary
    loaded_data = {key: data[key] for key in data.keys()}

    return loaded_data

import time
import psutil

import pandas as pd
import os

def store_results_in_csv(history, csv_file='training_results.csv'):
    """
    Function to store training results (loss, metrics, and learning rate) into a CSV file.
    Appends results with an incremental index.

    Parameters:
    - history: The history object from model training (it contains loss and metrics).
    - csv_file: The CSV file to store the results.
    """
    
    # Get the current index by counting the existing rows in the CSV
    if os.path.isfile(csv_file):
        existing_data = pd.read_csv(csv_file)
        next_index = len(existing_data) + 1
    else:
        next_index = 1

    # Extracting the necessary metrics from the history
    results = {
        'index': next_index,
        'loss': history.history['loss'][-1],  # Last epoch loss
        'throttle_loss': history.history['throttle_loss'][-1],
        'steering_loss': history.history['steering_loss'][-1],
        'brake_loss': history.history['brake_loss'][-1],
        'throttle_mean_absolute_error': history.history['throttle_mean_absolute_error'][-1],
        'steering_mean_absolute_error': history.history['steering_mean_absolute_error'][-1],
        'brake_mean_absolute_error': history.history['brake_mean_absolute_error'][-1],
    }

    # Convert to DataFrame
    results_df = pd.DataFrame([results])

    # Append to CSV, create file with headers if it doesn't exist
    results_df.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)

    print(f"Results saved to {csv_file}")


def train_model(model, data, epochs, batch_size, learning_rate):
    """Train the model with differential privacy applied to the optimizer."""
    # Start time tracking
    start_time = time.time()

    # Initial resource usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB

    rgb_data = data["rgb"]
    segmentation_data = data["segmentation"]
    hlc_data = data["hlc"]
    light_data = data["light"]
    measurements_data = data["measurements"]
    controls_data = data["controls"]  # Target variable

    # Start model training
    print("Training started...")

    # Warmup and decay schedule
    initial_lr = learning_rate
    warmup_epochs = 5
    
    def custom_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * ((epoch + 1) / warmup_epochs)
        else:
            decay_rate = 0.95
            return initial_lr * decay_rate ** (epoch - warmup_epochs)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(custom_schedule),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=2
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss=['mean_squared_error'] * 3,
        metrics=['mean_absolute_error']
    )

    history = model.fit(
        [rgb_data, segmentation_data, hlc_data, light_data, measurements_data],
        controls_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2,
        callbacks=callbacks
    )
    store_results_in_csv(history)

    # End time tracking
    end_time = time.time()
    training_time = end_time - start_time

    # Resource usage after training
    final_memory = process.memory_info().rss / 1024 / 1024  # in MB

    # Print resource usage statistics
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Initial Memory usage: {initial_memory:.2f} MB")
    print(f"Final Memory usage: {final_memory:.2f} MB")

    return history


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
        logging.info(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        logging.error(f"Failed to save weights for {client_id}: {e}")
    return weights_path, timestamp


def save_model(client_id, model, save_dir):
    """
    Save the trained model with versioning.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"weights.keras")
    try:
        model.save(model_path)
        logging.info(f"Model for {client_id} saved at {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model for {client_id}: {e}")
    return model_path


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
        logging.info(f"File {filename} uploaded successfully to Azure Blob Storage.")
        print(f"File {filename} uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        logging.error(f"Error uploading file {filename}: {e}")
        print(f"Error uploading file {filename}: {e}")

def load_model_weights(model, directory_path):
    """
    Load the first .keras weights file found in the specified directory.

    Args:
        model: Keras model instance.
        directory_path: Path to directory containing weights file.

    Returns:
        bool: True if weights loaded successfully, False otherwise.
    """
    try:
        # Search for .keras files in the directory
        keras_file = next((file for file in glob.glob(os.path.join(directory_path, "weights.keras"))), None)
        
        if keras_file:
            model.load_weights(keras_file)
            logging.info(f"Successfully loaded weights from {keras_file}")
            return True
        
        logging.error(f"No .keras files found in {directory_path}")
        return False

    except Exception as e:
        logging.error(f"Error loading weights: {str(e)}")
        return False

def add_laplace_noise(weights, epsilon, sensitivity):
    """Adds Laplace noise to the model weights for differential privacy."""
    noise = np.random.laplace(0, sensitivity / epsilon, size=weights.shape)
    return weights + noise

def save_weights_with_dp(model, weights_path, epsilon=0.1, sensitivity=1.0):
    """Saves the model weights with added differential privacy noise."""
    # Get the model weights
    weights = model.get_weights()
    
    # Add Laplace noise to each layer's weights
    perturbed_weights = []
    for weight in weights:
        perturbed_weights.append(add_laplace_noise(weight, epsilon, sensitivity))
    
    # Set the model weights to the perturbed weights
    model.set_weights(perturbed_weights)
    
    # Save the perturbed weights to the specified file
    model.save_weights(weights_path)
    print(f"Weights saved with DP noise to {weights_path}")

def add_laplace_noise_to_weights(weights, epsilon, sensitivity):
    """
    Add Laplace noise to model weights for differential privacy.
    - weights: model weights to perturb.
    - epsilon: privacy budget.
    - sensitivity: sensitivity of the weights.
    """
    noise = np.random.laplace(0, sensitivity / epsilon, size=weights.shape)
    return weights + noise
def save_weights_with_dp(client_id, model, save_dir, epsilon, sensitivity):
    """
    Save model weights with versioning and added DP noise.
    - epsilon: privacy budget.
    - sensitivity: sensitivity of the weights.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get versioned filename
    weights_path, next_version, timestamp = get_versioned_filename(client_id, save_dir)

    try:
        # Get model weights and add DP noise
        model_weights = model.get_weights()
        noisy_weights = [add_laplace_noise_to_weights(w, epsilon, sensitivity) for w in model_weights]

        # Set noisy weights back to the model
        model.set_weights(noisy_weights)

        # Save noisy weights to the specified path
        model.save_weights(weights_path)

        model.save_weights(os.path.join(save_dir, "weights.keras"))

        logging.info(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        logging.error(f"Failed to save weights for {client_id}: {e}")

    return weights_path, timestamp

# Updated Main Function to Reflect Unified Versioning
def main(client_id, data_path, save_dir, build_model):
    setup_logger()
    logging.info(f"Starting training...")

    try:
        # Load preprocessed data
        data = load_preprocessed_data(data_path)
        # Count total number of examples
        controls_data = data["controls"] # Target
        num_examples = controls_data.shape[0]
        print(num_examples)
        logging.info("Data loaded successfully.")
        print("Data loaded successfully")

        # Define the input shapes
        input_shapes = {
            "rgb": (128, 128, 3),
            "segmentation": (128, 128, 3),
            "hlc": (1,),
            "light": (1,),
            "measurements": (1,)
        }

        # Build and train the model
        model = build_model(input_shapes)

        if load_model_weights(model, save_dir):
            logging.info("Weights loaded successfully.")
            print("Weights loaded successfully.")
        else:
            logging.info("Failed to load weights. Training from scratch.")
            print("Failed to load weights. Training from scratch.")

        epochs = 30
        batch_size = 32
        learning_rate = 0.001

        history = train_model(model, data, epochs=epochs, batch_size=batch_size, 
                    learning_rate=learning_rate)
        
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
        model.save_weights(os.path.join(save_dir, "weights.keras"))
        weights_path, timestamp = save_weights(client_id, model, save_dir)


        # save weights with dp
        # epsilon = 0.5  # Privacy budget (0.1 == 0.5 inc in loss, lower value == high noise)
        # sensitivity = 1.0  # Sensitivity of the weights
        # weights_path, timestamp = save_weights_with_dp(client_id, model, save_dir, epsilon, sensitivity)
        


        # upload weights to blob
        upload_file(weights_path, CLIENT_CONTAINER_NAME, metadata)



        # Save and upload the full model
        # model_path = save_model(client_id, model, save_dir)

        logging.info(f"Training completed successfully.")
        print("Training and upload completed successfully.")
    
    except Exception as e:
        logging.error(f"Error during training: {e}")
        print(f"Error during training: {e}")