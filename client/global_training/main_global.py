# main.py

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv

# local imports (adjust paths if needed)
from preprocessing_global import IoTDataPreprocessor
from training import IoTModelTrainer
from functions import upload_file, save_weights, wait_for_csv, find_csv_file
from evaluate import evaluate_model

load_dotenv(dotenv_path=".env.server")

TRAIN_DATASET_PATH = "./DATA/DATA_VARIABLE/global_train.csv"
TEST_DATASET_PATH = "./DATA/DATA_VARIABLE/global_test.csv"

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main(epochs=100):
    # Research-grade configuration matching successful notebook for your 25 features
    config = {
        'train_data_path_pattern': TRAIN_DATASET_PATH,
        'test_data_path_pattern': TEST_DATASET_PATH,
        'epochs': epochs,
        'batch_size': 128,  # Same as successful notebook
        'random_state': SEED,
        'model_architecture': [256, 256],  # Research-proven architecture
        'learning_rate': 5e-5,  # Research-proven learning rate
        'num_classes': 15,
        'class_names': [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ],
        'artifacts_dir': "artifacts"
    }
    
    # Keep your existing file waiting and directory creation logic
    config['train_data_path'] = wait_for_csv(config['train_data_path_pattern'])
    config['test_data_path'] = wait_for_csv(config['test_data_path_pattern'])
    
    # Create directories
    for d in ['models', 'logs', 'plots', 'data', 'federated_models', config['artifacts_dir']]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print("SecureFL - Server: Enhanced preprocessing and training for 25-feature ML-Edge-IIoT")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architecture: {config['model_architecture']}, Learning Rate: {config['learning_rate']}")
    print(f"{'='*60}\n")

    # --------------------------
    # 1) Enhanced preprocessing for your 25 features
    # --------------------------
    preprocessor = IoTDataPreprocessor(global_class_names=config['class_names'],
                                     artifacts_dir=config['artifacts_dir'])
    
    # Fit enhanced preprocessor with dual-scaling pipeline
    X_all, y_all, actual_classes, artifacts = preprocessor.fit_preprocessor(
        config['train_data_path'], target_num_classes=config['num_classes']
    )
    
    print(f"Enhanced preprocessing expanded your 25 features to {X_all.shape[1]} dimensions")
    
    # Keep your existing artifact saving and uploading logic
    preprocessor_path = os.path.join(config['artifacts_dir'], "preprocessor.pkl")
    global_le_path = os.path.join(config['artifacts_dir'], "global_label_encoder.pkl")
    feature_info_path = os.path.join(config['artifacts_dir'], "feature_info.pkl")
    
    joblib.dump(artifacts['preprocessor'], preprocessor_path)
    joblib.dump(artifacts['global_label_encoder'], global_le_path)
    joblib.dump(artifacts['feature_info'], feature_info_path)
    
    # Upload preprocessing artifacts for clients to download
    try:
        upload_file(preprocessor_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
        upload_file(global_le_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
        upload_file(feature_info_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
        print("Uploaded enhanced preprocessing artifacts to server blob container.")
    except Exception as e:
        print(f"Warning: failed to upload artifacts to cloud: {e}")

    # --------------------------
    # 2) Enhanced train/val split with stratification
    # --------------------------
    min_class_count = int(np.min(np.bincount(y_all)))
    if min_class_count >= 2:
        stratify_label = y_all
    else:
        stratify_label = None

    # Use 20% validation split for better generalization (like notebook)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=config['random_state'], 
        stratify=stratify_label
    )

    print(f"Enhanced split -> Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Feature dimensions: {X_train.shape[1]} (enhanced from your 25 selected features)")

    # --------------------------
    # 3) Transform test set using enhanced preprocessing
    # --------------------------
    X_test, y_test, test_num_classes = preprocessor.transform(config['test_data_path'],
                                                            target_num_classes=config['num_classes'])
    print(f"Test shape: {X_test.shape}, classes in test: {test_num_classes}")

    # --------------------------
    # 4) Build and train model with research-grade settings
    # --------------------------
    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],  # Enhanced feature dimensions
        num_classes=config['num_classes'],
        architecture=config['model_architecture']  # Research-proven architecture
    )

    print(f"Enhanced model -> Input dim: {X_train.shape[1]}, Output classes: {config['num_classes']}")
    print(f"Architecture: {config['model_architecture']}")

    # Categorical labels (fixed num_classes)
    y_train_cat = to_categorical(y_train, num_classes=config['num_classes'])
    y_val_cat = to_categorical(y_val, num_classes=config['num_classes'])
    y_test_cat = to_categorical(y_test, num_classes=config['num_classes'])

    print(f"y shapes -> train: {y_train_cat.shape}, val: {y_val_cat.shape}, test: {y_test_cat.shape}")

    # Train with research-grade settings
    history, training_time, num_samples = trainer.train_model(
        X_train, y_train_cat, X_val, y_val_cat,
        model=model,
        architecture=config['model_architecture'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],  # Research-proven learning rate
        verbose=2,
    )

    model = trainer.get_model()
    print("Enhanced global model training complete.")
    model.summary()

    # Keep your existing model saving logic
    weights_path = save_weights(model, "models", "g0.h5")
    arch_file = os.path.join("models", "model_architecture.json")
    with open(arch_file, "w") as f:
        f.write(model.to_json())

    # Evaluate on the enhanced preprocessed test set
    eval_results = evaluate_model(model, X_test, y_test_cat, class_names=config['class_names'])
    test_metrics = eval_results['test']
    print(f"Enhanced model test results: {test_metrics}")

    # Keep your existing metadata saving and uploading logic
    metadata = {
        "final_test_loss": str(test_metrics.get('loss', 'nan')),
        "final_test_accuracy": str(test_metrics.get('accuracy', 'nan')),
        "final_test_precision": str(test_metrics.get('macro_precision', 'nan')),
        "final_test_recall": str(test_metrics.get('macro_recall', 'nan')),
        "final_test_f1": str(test_metrics.get('macro_f1', 'nan')),
    }

    complete_metadata = {
        "test_metrics": test_metrics,
        "model_architecture": config['model_architecture'],
        "learning_rate": config['learning_rate'],
        "enhanced_features": True,
        "selected_features_count": 25,
        "feature_dimensions": X_train.shape[1],
        "preprocessing_pipeline": "RobustScaler + MinMaxScaler + OneHotEncoder",
        "epochs": epochs,
        "batch_size": config['batch_size'],
        "num_training_samples": str(num_samples),
        "global_classes": config['class_names'],
        "num_classes_fixed": config['num_classes'],
        "data_classes_present": int(actual_classes),
    }

    metadata_file_path = os.path.join("models", "g0_metadata.json")
    with open(metadata_file_path, "w") as f:
        json.dump(complete_metadata, f, indent=2)

    # Upload weights and metadata
    try:
        upload_file(weights_path, os.getenv("SERVER_CONTAINER_NAME"), metadata)
        upload_file(metadata_file_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
        print("Uploaded enhanced model weights and metadata to server blob container.")
    except Exception as e:
        print(f"Warning: failed to upload model artifacts: {e}")

    print(f"\nTraining completed for 25-feature ML-Edge-IIoT setup!")
    print("Done.")

if __name__ == "__main__":
    main(epochs=100)
