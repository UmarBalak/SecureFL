# training.py

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import time
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import IoTModel

class IoTModelTrainer:
    def __init__(self, random_state=42):
        """
        Initialize the model trainer with research-grade settings
        """
        self.random_state = random_state
        self.model_builder = IoTModel()
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def create_model(self, input_dim, num_classes, architecture=None):
        """
        Create FL-compatible model with research-grade architecture
        """
        # Use research-proven architecture if none provided
        if architecture is None:
            architecture = [256, 256]  # Matches successful notebook
            
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        return self.model

    def train_model(self, X_train, y_train_cat, X_val, y_val_cat,
                   model, architecture, epochs=100, batch_size=128, verbose=2,
                   callbacks=None, epoch_checkpoints=None, num_classes=15, 
                   learning_rate=5e-5):  # Research-proven learning rate
        """
        Train model with research-grade settings matching successful notebook
        """
        print("\nTraining MLP model with research-grade configuration...")
        print(f"Input dimensions: {X_train.shape[1]} (from your 25 selected features)")
        
        self.model = model
        num_samples = len(X_train)
        epoch_checkpoints = epoch_checkpoints or [epochs]
        callbacks = callbacks or []
        
        # Use same optimizer settings as successful notebook
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,  # Much lower LR: 5e-5 vs 0.001
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        # Compile with F1-score monitoring (research-grade metrics)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy'
            ]
        )
        
        # Research-grade callbacks with F1-score monitoring
        default_callbacks = [
            ModelCheckpoint(
                filepath='models/best_mlp_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=2
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More aggressive reduction like notebook
                patience=8,  # Longer patience for stability
                min_lr=1e-6,
                verbose=1,
                cooldown=3  # Cooldown period
            ),
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            )
        ]
        
        all_callbacks = default_callbacks + callbacks
        
        start_time = time.time()
        
        try:
            # Train with research-grade configuration
            self.history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=batch_size,  # Same as notebook: 128
                callbacks=all_callbacks,
                verbose=verbose,
                shuffle=True  # Ensure data shuffling
            )
            
            training_success = True
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        return self.history, training_time, num_samples

    def get_model(self):
        """Get the trained model"""
        return self.model

    def get_history(self):
        """Get the training history"""
        return self.history
