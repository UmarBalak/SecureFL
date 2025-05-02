import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import time
from datetime import datetime

# Import custom modules
from preprocessing import IoTDataPreprocessor
from model import IoTModel
from training import IoTModelTrainer
from evaluation import IoTModelEvaluator
from visualization import IoTVisualizer

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

def main():
    """Main function to run the entire pipeline"""
    # Configuration settings
    config = {
        'data_path': f"{path}/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",  # Path to dataset
        'max_samples': 30000,                                # Set to a number to limit samples
        'test_size': 0.2,                                    # Test split proportion
        'epochs': 50,                                        # Max training epochs
        'batch_size': 128,                                    # Training batch size
        'random_state': 42,                                  # For reproducibility
        'model_architecture':  [128, 64],      # Units per hidden layer
        'skip_training': False,                              # Skip training and load model
        'use_lightweight_model': False                       # Use lightweight model for edge devices
    }

    # Set random seeds for reproducibility
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])

    # Create directory structure
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"IoT Network Attack Detection System")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = IoTDataPreprocessor(random_state=config['random_state'])

    # Load and preprocess data
    X, y = preprocessor.load_data(config['data_path'])

    # Display initial class distribution
    visualizer = IoTVisualizer(preprocessor.attack_type_map, random_state=config['random_state'])
    visualizer.plot_class_distribution(y, title_prefix="Original")

    # Preprocess data
    preprocessing_start_time = time.time()
    (X_train, X_test, y_train, y_test,
     y_train_cat, y_test_cat, stats) = preprocessor.preprocess_data(
        X, y, max_samples=config['max_samples'], test_size=config['test_size']
    )
    preprocessing_time = time.time() - preprocessing_start_time

    # Display balanced class distribution
    visualizer.plot_class_distribution(y_train, title_prefix="Balanced Training")

    # Feature importance visualization
    feature_names = X_train.columns.tolist()
    visualizer.plot_feature_importance(X_train, y_train, feature_names)

    # Data embeddings visualization (on a sample for performance)
    print("Generating data embeddings visualization...")
    if len(X_train) > 5000:
        sample_idx = np.random.choice(len(X_train), 5000, replace=False)
        X_sample = X_train.iloc[sample_idx].values
        y_sample = y_train[sample_idx]
    else:
        X_sample = X_train.values
        y_sample = y_train

    visualizer.visualize_data_embeddings(X_sample, y_sample, method='tsne')

    # Initialize model
    model_builder = IoTModel()

    # Check if we should skip training
    if config['skip_training']:
        print("\nSkipping training, loading saved model...")
        try:
            model = tf.keras.models.load_model('models/mlp_model.keras')
            print("Model loaded successfully from 'models/mlp_model.keras'")
            training_time = 0
            history = None
        except:
            print("Failed to load model. Training new model...")
            config['skip_training'] = False

    # Train model if not skipping
    if not config['skip_training']:
        # Initialize trainer
        trainer = IoTModelTrainer(random_state=config['random_state'])

        # Train model
        if config['use_lightweight_model']:
            print("\nTraining lightweight model for edge devices...")
            input_dim = X_train.shape[1]
            num_classes = y_train_cat.shape[1]
            model = model_builder.create_lightweight_model(input_dim, num_classes)

            # Train with early stopping and checkpoints
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='models/best_lightweight_model.keras',
                monitor='val_loss', save_best_only=True)

            training_start_time = time.time()
            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            training_time = time.time() - training_start_time

            # Save model
            model.save('models/lightweight_model.keras')
            print("Lightweight model saved to 'models/lightweight_model.keras'")
        else:
            print("\nTraining MLP model...")
            history, training_time = trainer.train_model(
                X_train, y_train_cat, X_test, y_test_cat,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                architecture=config['model_architecture'],
                verbose=1
            )
            model = trainer.get_model()

    # Plot training history if available
    if not config['skip_training'] and history:
        visualizer.plot_training_history(history)

    # Evaluate model
    print("\nEvaluating model...")
    evaluator = IoTModelEvaluator(preprocessor.attack_type_map)
    eval_results = evaluator.evaluate_model(model, X_test, y_test, y_test_cat)

    # Generate predictions for visualization
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Generate visualizations
    visualizer.plot_confusion_matrix(y_test, y_pred)
    visualizer.plot_roc_curves(X_test, y_test_cat, model)
    visualizer.plot_precision_recall_curves(X_test, y_test_cat, model)

    # Plot per-class metrics
    visualizer.plot_per_class_metrics(eval_results['per_class_metrics'])

    # Calculate advanced metrics
    advanced_metrics = evaluator.calculate_advanced_metrics(y_test, y_pred, y_pred_proba)

    # Prepare model information
    model_info = {
        'architecture': config['model_architecture'] if not config['use_lightweight_model'] else 'lightweight',
        'training_time': training_time,
        'parameters': model.count_params(),
        'layers': len(model.layers)
    }

    # Save run information
    save_run_info(config, stats, model_info, eval_results)

    print(f"\n{'='*70}")
    print(f"IoT Network Attack Detection System - Run Complete")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model accuracy: {eval_results['accuracy']:.4f}")
    print(f"Preprocessing time: {preprocessing_time:.2f} seconds")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()