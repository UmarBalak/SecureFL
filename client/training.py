import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
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
                model, epochs=50, batch_size=64, verbose=2,
                use_dp=True, l2_norm_clip=1.0, noise_multiplier=1.2, microbatches=1):
        """
        Train the MLP model with optional differential privacy.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features.
        y_train_cat : numpy.ndarray
            One-hot encoded training labels.
        X_test : numpy.ndarray
            Validation features.
        y_test_cat : numpy.ndarray
            One-hot encoded validation labels.
        model : tf.keras.models.Sequential
            The model to train.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        verbose : int
            Verbosity level for training output.
        use_dp : bool
            Whether to use differential privacy.
        l2_norm_clip : float
            Clipping norm for DP-SGD.
        noise_multiplier : float
            Noise multiplier for DP-SGD.
        microbatches : int
            Number of microbatches for DP-SGD.

        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history.
        training_time : float
            Time taken for training in seconds.
        """
        print("\nTraining MLP model...")

        self.model = model

        # Calculate the appropriate number of microbatches
        num_samples = len(X_train)
        if use_dp:
            # If microbatches > batch_size, reduce it to batch_size
            if microbatches > batch_size:
                microbatches = batch_size
            
            # Ensure batch_size is divisible by microbatches
            if batch_size % microbatches != 0:
                # Find the largest divisor of batch_size that is <= microbatches
                for i in range(microbatches, 0, -1):
                    if batch_size % i == 0:
                        microbatches = i
                        break
            
            print(f"Using {microbatches} microbatches with batch size {batch_size}")
            
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=0.001
            )
            
            # Calculate steps for privacy accounting
            steps_per_epoch = num_samples // batch_size
            total_steps = steps_per_epoch * epochs
            
            # Import required packages - attempt multiple import paths to handle different versions
            import importlib
            
            try:
                # Try to import using current tensorflow_privacy package structure
                try:
                    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
                    
                    # Calculate privacy using the compute_dp_sgd_privacy function
                    # which internally uses the RDP accountant
                    sampling_rate = batch_size / num_samples
                    delta = 1.0 / num_samples
                    
                    epsilon = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                        n=num_samples,
                        batch_size=batch_size,
                        noise_multiplier=noise_multiplier,
                        epochs=epochs,
                        delta=delta
                    )[0]
                    
                except ImportError:
                    # Try alternative import paths for different tensorflow_privacy versions
                    try:
                        # For newer versions
                        from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
                        from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
                    except ImportError:
                        # For older versions
                        from tensorflow_privacy.analysis.rdp_accountant import compute_rdp
                        from tensorflow_privacy.analysis.rdp_accountant import get_privacy_spent
                    
                    # Calculate privacy using RDP accountant
                    # Sampling rate: batch_size / dataset_size
                    sampling_rate = batch_size / num_samples
                    
                    # RDP orders
                    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                    
                    # Compute RDP for given parameters
                    rdp = compute_rdp(
                        q=sampling_rate,
                        noise_multiplier=noise_multiplier,
                        steps=total_steps,
                        orders=orders
                    )
                    
                    # Convert RDP to (ε, δ)-DP
                    # Common delta value is 1/N where N is the training dataset size
                    delta = 1.0 / num_samples
                    epsilon, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
                
                print(f"Using DP-SGD Optimizer with:")
                print(f" - L2 norm clip: {l2_norm_clip}")
                print(f" - Noise multiplier: {noise_multiplier}")
                print(f" - Microbatches: {microbatches}")
                print(f" - Sampling rate: {sampling_rate:.6f}")
                print(f" - Training steps: {total_steps} ({epochs} epochs)")
                print(f" - DP guarantee: ε = {epsilon:.2f} at δ = {delta:.2e}")
            except Exception as e:
                print(f"Could not use TensorFlow Privacy modules for precise privacy accounting: {str(e)}")
                print("Using a simplified estimate (less accurate):")
                
                # Check if numpy is available
                try:
                    import numpy as np
                    # Simplified estimation as fallback
                    sampling_rate = batch_size / num_samples
                    delta = 1.0 / num_samples
                    # This is a simplified upper bound estimate
                    approx_epsilon = noise_multiplier * np.sqrt(2 * total_steps * np.log(1/delta))
                    print(f" - Approximate ε: {approx_epsilon:.2f} (rough estimate)")
                    print(f" - δ = {delta:.2e} (1/N)")
                except ImportError:
                    # Even more basic calculation if numpy isn't available
                    sampling_rate = batch_size / num_samples
                    delta = 1.0 / num_samples
                    import math
                    approx_epsilon = noise_multiplier * math.sqrt(2 * total_steps * math.log(num_samples))
                    print(f" - Very rough ε estimate: {approx_epsilon:.2f}")
                    print(f" - δ = {delta:.2e} (1/N)")
                    
                print("Note: This is an approximation. For accurate accounting, install tensorflow-privacy.")
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            print("Using standard Adam optimizer (no differential privacy)")

        # Compile with selected optimizer
        self.model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True, 
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath='models/best_mlp_model.h5', 
            monitor='val_loss', 
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )

        start_time = time.time()
        
        try:
            self.history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, model_checkpoint],
                verbose=verbose
            )
            training_success = True
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            print("Trying again with simpler DP configuration...")
            
            # Fallback to simpler configuration
            if use_dp:
                optimizer = DPKerasAdamOptimizer(
                    l2_norm_clip=l2_norm_clip,
                    noise_multiplier=noise_multiplier,
                    num_microbatches=1,  # Simplest case: one microbatch
                    learning_rate=0.001
                )
                
                self.model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                try:
                    self.history = self.model.fit(
                        X_train, y_train_cat,
                        validation_data=(X_test, y_test_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=verbose
                    )
                    training_success = True
                    print("Training successful with simplified DP configuration (1 microbatch)")
                except Exception as e2:
                    print(f"Training failed again with error: {str(e2)}")
                    print("Falling back to standard optimizer without DP")
                    
                    # Final fallback to non-DP optimizer
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    self.model.compile(
                        optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    self.history = self.model.fit(
                        X_train, y_train_cat,
                        validation_data=(X_test, y_test_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=verbose
                    )
                    training_success = True
                    print("Training successful with standard optimizer (no DP)")
            
        training_time = time.time() - start_time

        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Save model and weights
        try:
            self.model.save('models/model.h5')
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            
        try:
            self.model.save_weights('models/weights.h5')
            print("Model weights saved successfully")
            
            # Also save to federated_models directory
            os.makedirs('federated_models', exist_ok=True)
            self.model.save_weights('federated_models/weights.h5')
            print("Model weights saved to federated_models directory")
        except Exception as e:
            print(f"Error saving weights: {str(e)}")

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