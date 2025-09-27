# training_global_complete.py - Training module for global model

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import time
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import IoTModel

class GlobalModelTrainer:
    """
    Complete trainer for initial global model with WandB integration
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_builder = IoTModel()
        self.model = None
        self.history = None
        
        # Set reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def create_model(self, input_dim, num_classes, architecture=None):
        """
        Create global model architecture
        """
        if architecture is None:
            architecture = [256, 256]  # Research-proven architecture
        
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        return self.model
    
    def train_global_model(self, X_train, y_train_cat, X_val, y_val_cat,
                          model, architecture, epochs=100, batch_size=128,
                          learning_rate=5e-5, num_classes=15, verbose=2,
                          wandb_tracker=None):
        """
        Train initial global model with comprehensive WandB logging
        """
        print("\n" + "="*60)
        print("🎯 GLOBAL MODEL TRAINING - Initial FL Baseline")
        print("="*60)
        print(f"Input dimensions: {X_train.shape[1]} features")
        print(f"Architecture: {architecture}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        
        self.model = model
        num_samples = len(X_train)
        
        # Research-grade optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        
        # Compile with comprehensive metrics
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tfa.metrics.F1Score(
                    num_classes=num_classes,
                    average='macro',
                    threshold=0.5,
                    name='macro_f1'
                ),
                tfa.metrics.F1Score(
                    num_classes=num_classes,
                    average='weighted',
                    threshold=0.5,
                    name='weighted_f1'
                )
            ]
        )
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/global_checkpoints", exist_ok=True)
        
        # Standard callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='models/best_global_model.h5',
                monitor='val_macro_f1',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-6,
                verbose=1,
                cooldown=3
            ),
            EarlyStopping(
                monitor='val_macro_f1',
                mode='max',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            )
        ]
        
        # WandB callbacks
        if wandb_tracker and wandb_tracker.current_run:
            print("✅ Adding WandB callbacks for global training...")
            
            wandb_callbacks = [
                WandbMetricsLogger(
                    log_freq='epoch',
                    initial_global_step=0
                )
            ]
            
            callbacks.extend(wandb_callbacks)
            print(f"   Total callbacks: {len(callbacks)}")
        
        # Custom WandB logging callback
        class GlobalTrainingLogger(Callback):
            def __init__(self, wandb_tracker):
                super().__init__()
                self.wandb_tracker = wandb_tracker
            
            def on_epoch_end(self, epoch, logs=None):
                if self.wandb_tracker and logs:
                    train_metrics = {
                        "loss": logs.get("loss", 0),
                        "accuracy": logs.get("accuracy", 0),
                        "macro_f1": logs.get("macro_f1", 0),
                        "weighted_f1": logs.get("weighted_f1", 0)
                    }
                    
                    val_metrics = {
                        "loss": logs.get("val_loss", 0),
                        "accuracy": logs.get("val_accuracy", 0),
                        "macro_f1": logs.get("val_macro_f1", 0),
                        "weighted_f1": logs.get("val_weighted_f1", 0)
                    }
                    
                    self.wandb_tracker.log_global_training_metrics(
                        epoch + 1, train_metrics, val_metrics
                    )
        
        if wandb_tracker:
            callbacks.append(GlobalTrainingLogger(wandb_tracker))
        
        print(f"📊 Starting global model training...")
        print(f"   Callbacks: {[cb.__class__.__name__ for cb in callbacks]}")
        
        start_time = time.time()
        
        try:
            # Train the global model
            self.history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )
            
            training_success = True
            print("✅ Global model training completed successfully!")
            
        except Exception as e:
            print(f"❌ Global training failed: {e}")
            training_success = False
            raise
        
        training_time = time.time() - start_time
        print(f"⏱️  Global training completed in {training_time:.2f} seconds")
        
        # Log final training summary to WandB
        if wandb_tracker and wandb_tracker.current_run:
            final_summary = {
                "training_duration_seconds": training_time,
                "training_samples": num_samples,
                "epochs_completed": len(self.history.history.get('loss', [])),
                "training_success": training_success
            }
            
            if self.history.history:
                final_summary.update({
                    "final_train_loss": self.history.history.get('loss', [0])[-1],
                    "final_val_loss": self.history.history.get('val_loss', [0])[-1],
                    "final_train_accuracy": self.history.history.get('accuracy', [0])[-1],
                    "final_val_accuracy": self.history.history.get('val_accuracy', [0])[-1],
                    "final_train_f1": self.history.history.get('macro_f1', [0])[-1],
                    "final_val_f1": self.history.history.get('val_macro_f1', [0])[-1]
                })
            
            wandb.log({"global_training_summary": final_summary})
        
        return self.history, training_time, num_samples
    
    def get_model(self):
        return self.model
    
    def get_history(self):
        return self.history
