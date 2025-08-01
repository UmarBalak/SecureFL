
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
from client.preprocessing import IoTDataPreprocessor
from client.model import IoTModel
from client.training import IoTModelTrainer
from client.evaluation import IoTModelEvaluator
from azure.storage.blob import BlobServiceClient

from client.functions import wait_for_csv, load_model_weights, upload_file, save_run_info, find_csv_file

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

def main(client_id):

    config = {
        'data_path_pattern': f"./DATA/ML-EdgeIIoT-dataset.csv",
        # 'data_path_pattern': f"{path}/data_part_*.csv",
        'max_samples': 200000,
        'test_size': 0.2,
        'epochs': 50,
        'batch_size': 256,
        'random_state': 42,
        # 'model_architecture': [256, 128, 128]
        'model_architecture': [64, 128, 64, 32],
        # 'model_architecture': [128, 64, 32], # 1.0, 1.0, bs 512, ep 50 -> e 2.75, ac 40
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

    preprocessing_start_time = time.time()
    preprocessor = IoTDataPreprocessor()
    X_train, X_test, y_train_type, y_test_type, y_type, num_classes, l = preprocessor.preprocess_data(config['data_path'])
    preprocessing_time = time.time() - preprocessing_start_time
    trainer = IoTModelTrainer(random_state=config['random_state'])
    # model = trainer.create_model(
    #     input_dim=X_train.shape[1],
    #     num_classes=num_classes,
    #     architecture=config['model_architecture']
    # )
    print("@" * 50)
    print(X_train.shape[1])
    print("@" * 50)
    # model.save(os.path.join(save_dir, "model_arch.h5"))
    print(f"Model architecture saved at {os.path.join(save_dir, 'model_arch.h5')}")
    print("\nTraining MLP model...")
    use_dp = True
    l2_norm_clip = 1.0
    noise_multiplier = 0.9
    microbatches = 1
    # if load_model_weights(model, save_dir):
    #     print("Weights loaded successfully.")
    # else:
    #     print("Failed to load weights. Training from scratch.")

    from tensorflow.keras.utils import to_categorical

    y_train_cat = to_categorical(y_train_type, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_type, num_classes=num_classes)
    num_examples = X_train.shape[0]

    class CPUMonitor:
        def __init__(self, sample_interval=0.05):
            """
            Initialize CPU usage monitor for the current process.

            Parameters:
            -----------
            sample_interval : float
                Time interval between CPU samples (seconds).
            """
            self.sample_interval = sample_interval
            self.cpu_samples = []
            self.timestamps = []
            self.running = False
            self.thread = None
            self.start_time = None
            self.process = psutil.Process()
            self.num_cores = psutil.cpu_count(logical=True)  # Total logical cores

        def start(self):
            """Start CPU sampling in a separate thread."""
            self.cpu_samples = []
            self.timestamps = []
            self.running = True
            self.start_time = time.time()
            self.process.cpu_percent()  # Initialize cpu_percent
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()

        def _monitor(self):
            """Sample process-specific CPU usage until stopped."""
            while self.running:
                current_time = time.time()
                # Normalize by number of cores
                cpu_usage = self.process.cpu_percent(interval=0) / self.num_cores
                self.cpu_samples.append(cpu_usage)
                self.timestamps.append(current_time - self.start_time)
                time.sleep(self.sample_interval)

        def stop(self):
            """Stop CPU sampling."""
            self.running = False
            if self.thread:
                self.thread.join()
            # Debug: Save samples to file
            with open("cpu_samples.txt", "a") as f:
                f.write(f"Total cores: {self.num_cores}\n")
                for t, cpu in zip(self.timestamps, self.cpu_samples):
                    f.write(f"Time: {t:.2f}s, CPU: {cpu:.2f}%\n")

        def get_average_cpu_usage(self, up_to_time):
            """
            Compute average CPU usage up to a given time.

            Parameters:
            -----------
            up_to_time : float
                Time (seconds) up to which to average CPU samples.

            Returns:
            --------
            float
                Average CPU usage percentage (0-100%).
            """
            if not self.cpu_samples:
                return 0.0
            valid_samples = [cpu for t, cpu in zip(self.timestamps, self.cpu_samples) if t <= up_to_time]
            return mean(valid_samples) if valid_samples else 0.0

    def run_all_combinations(trainer, X_train, y_train_cat, X_test, y_test_cat, config, use_dp=True):
        """
        Run training experiments for all combinations of hyperparameters and save results.

        Parameters:
        -----------
        trainer : IoTModelTrainer
            Trainer object with create_model and train_model methods.
        X_train : numpy.ndarray
            Training features.
        y_train_cat : numpy.ndarray
            One-hot encoded training labels.
        X_test : numpy.ndarray
            Validation features.
        y_test_cat : numpy.ndarray
            One-hot encoded validation labels.
        config : dict
            Configuration dictionary with model_architecture, batch_size, epochs.
        use_dp : bool
            Whether to use differential privacy.

        Returns:
        --------
        results : list
            List of dictionaries containing experiment results.
        """
        results = []
        results_file_csv = "mlp_experiments.csv"
        results_file_json = "mlp_experiments.json"

        # Load existing results if available
        if os.path.exists(results_file_json):
            with open(results_file_json, "r") as jf:
                results = json.load(jf)

        combinations = product(
            [[128, 64, 32]],  # model_architecture
            [1.0],       # l2_norm_clip
            [0.2],  # noise_multiplier
            [256, 512],           # batch_size (updated to match your latest JSON)
            [[0.3, 0.25]]     # dropout_rate_for_1, dropout_rate_for_all
        )

        epoch_checkpoints = [30, 50, 80, 100]  # Updated to match your latest JSON
        process = psutil.Process()

        for arch, l2_clip, noise_mul, batch_size, [d1, d2] in combinations:
            config['model_architecture'] = arch
            config['batch_size'] = batch_size
            config['epochs'] = max(epoch_checkpoints)

            model = trainer.create_model(
                input_dim=X_train.shape[1],
                num_classes=y_train_cat.shape[1],
                architecture=arch,
                dropout_rate_for_1=d1,
                dropout_rate_for_all=d2
            )

            start_time = time.time()
            cpu_monitor = CPUMonitor(sample_interval=0.05)
            cpu_monitor.start()

            train_start = time.time()
            history, training_time, noise_multiplier, l2_norm_clip, microbatches, epsilon_dict, delta, epoch_times, memory_samples, mem_start = trainer.train_model(
                X_train, y_train_cat, X_test, y_test_cat,
                model=model,
                epochs=config['epochs'],
                batch_size=batch_size,
                verbose=0,
                use_dp=use_dp,
                l2_norm_clip=l2_clip,
                noise_multiplier=noise_mul,
                microbatches=1,
                epoch_checkpoints=epoch_checkpoints
            )
            train_end = time.time()
            print(f"train_model took {train_end - train_start:.2f} seconds")
            print(f"epoch_times length: {len(epoch_times)}, memory_samples length: {len(memory_samples)}")

            cpu_monitor.stop()
            elapsed_time = time.time() - start_time

            post_metrics_start = time.time()
            for e in epoch_checkpoints:
                if e > len(history.history['accuracy']):
                    print(f"Warning: Checkpoint epoch {e} not reached (possibly due to early stopping).")
                    continue

                train_acc = history.history['accuracy'][e-1]
                val_acc = history.history['val_accuracy'][e-1]
                train_loss = history.history['loss'][e-1]
                val_loss = history.history['val_loss'][e-1]
                checkpoint_time = epoch_times[e-1] if e-1 < len(epoch_times) else training_time
                epsilon = epsilon_dict.get(e, float("inf"))
                cpu_usage = cpu_monitor.get_average_cpu_usage(checkpoint_time)
                memory_usage_mb = (memory_samples[e-1] - mem_start) / (1024 * 1024) if e-1 < len(memory_samples) else 0.0

                result = {
                    'architecture': arch,
                    'dropout_rate_1': d1,
                    'dropout_rate_all': d2,
                    'l2_norm_clip': l2_clip,
                    'noise_multiplier': noise_mul,
                    'batch_size': batch_size,
                    'epochs': e,
                    'train_accuracy': round(train_acc, 4),
                    'val_accuracy': round(val_acc, 4),
                    'train_loss': round(train_loss, 4),
                    'val_loss': round(val_loss, 4),
                    'dp_epsilon': round(epsilon, 4),
                    'dp_delta': delta,
                    'checkpoint_time_sec': round(checkpoint_time, 2),
                    'wall_clock_time_sec': round(elapsed_time, 2),
                    'cpu_usage_percent': round(cpu_usage, 2),
                    'memory_used_mb': round(memory_usage_mb, 2)
                }
                results.append(result)
            post_metrics_end = time.time()
            print(f"Post-train metrics and loop took {post_metrics_end - post_metrics_start:.2f} seconds")

            clear_start = time.time()
            K.clear_session()
            gc.collect()  # Additional cleanup
            clear_end = time.time()
            print(f"K.clear_session took {clear_end - clear_start:.2f} seconds")

        # Save results after all combinations
        pd.DataFrame(results).to_csv(results_file_csv, index=False)
        with open(results_file_json, "w") as jf:
            json.dump(results, jf, indent=4)

        return results

    run_all_combinations(
        trainer=trainer,
        X_train=X_train,
        y_train_cat=y_train_cat,
        X_test=X_test,
        y_test_cat=y_test_cat,
        config=config,
        use_dp=False
    )

    model = trainer.get_model()
    print("Model training complete.")
    model.summary()
    model.save('baseline_model.h5')


    # try:
    #     if history and hasattr(history, 'history') and 'loss' in history.history:
    #         final_loss = history.history['loss'][-1]
    #     else:
    #         final_loss = None
    # except Exception as e:
    #     print(f"Error getting final loss: {e}")
    #     final_loss = None
    # metadata = {
    #     'num_examples': str(num_examples),
    #     'loss': str(final_loss) if final_loss is not None else "unknown",
    # }
    # print("\nEvaluating model on original weights...")
    # evaluator = IoTModelEvaluator(preprocessor.attack_type_map)
    # eval_results = evaluator.evaluate_model(model, X_test, y_test, y_test_cat)
    # print(f"Original accuracy: {eval_results['accuracy']:.4f}")
    # original_loss, original_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    # print(f"Original Weights - Loss: {original_loss:.4f}, Accuracy: {original_accuracy:.4f}")

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env.client')
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        print("Client ID environment variable is missing.")
        raise ValueError("Missing required environment variable: CLIENT_ID")
    else:
        print(f"Client ID: {client_id}")
        main(client_id)