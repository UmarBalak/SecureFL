import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import IoTModel

class IoTModelTrainer:
    def __init__(self, random_state=42):
        """
        Initialize the model trainer

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
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
        Create a new MLP model

        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        architecture : list of int, optional
            List specifying the number of units in each hidden layer

        Returns:
        --------
        model : tf.keras.models.Sequential
            Compiled MLP model
        """
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        return self.model

    def train_model(self, X_train, y_train_cat, X_test, y_test_cat,
                   model, epochs=50, batch_size=64, verbose=2,):
        """
        Train the MLP model with early stopping

        Parameters:
        -----------
        X_train : numpy.ndarray or pandas.DataFrame
            Training features
        y_train_cat : numpy.ndarray
            One-hot encoded training labels
        X_test : numpy.ndarray or pandas.DataFrame
            Validation features
        y_test_cat : numpy.ndarray
            One-hot encoded validation labels
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        architecture : list of int, optional
            List specifying the number of units in each hidden layer
        verbose : int
            Verbosity level (0, 1, or 2)

        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        training_time : float
            Time taken for training in seconds
        """
        print("\nTraining MLP model...")

        # Create model
        self.model = model

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model_checkpoint = ModelCheckpoint(
            filepath='models/best_mlp_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train model
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=verbose
        )
        training_time = time.time() - start_time

        print(f"Model training completed in {training_time:.2f} seconds")

        # Save model
        self.model.save('models/model.h5')
        print("Model saved to 'models/model.h5'")

        # Save weights separately for federated learning
        self.model.save_weights('federated_models/weights.h5')
        print("Model weights saved to 'federated_models/weights.h5'")

        return self.history, training_time

    def get_model(self):
        """
        Get the trained model

        Returns:
        --------
        model : tf.keras.models.Sequential
            Trained model
        """
        return self.model

    def get_history(self):
        """
        Get the training history

        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        return self.history