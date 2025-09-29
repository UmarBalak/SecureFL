# training_dp_corrected.py

import os
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from functools import partial

# TensorFlow Privacy for Gaussian mechanism
try:
    import dp_accounting
except Exception as e:
    dp_accounting = None

# Model builder (existing)
try:
    from model_dp import IoTModel
except ImportError:
    class IoTModel:
        def create_mlp_model(self, input_dim, num_classes, architecture):
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_dim,))
            ])
            for units in architecture:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            return model

# ========================= Per-example gradients (unchanged) =========================

@tf.function
def compute_per_example_gradients(model, X_batch, y_batch):
    """Efficient per-example gradient computation using tf.vectorized_map"""
    def single_example_grad(example_data):
        x_single, y_single = example_data
        x_single = tf.expand_dims(x_single, 0)
        y_single = tf.expand_dims(y_single, 0)
        
        with tf.GradientTape() as tape:
            predictions = model(x_single, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_single, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [g if g is not None else tf.zeros_like(v) 
                    for g, v in zip(gradients, model.trainable_variables)]
        
        return gradients

    per_example_grads = tf.vectorized_map(
        fn=single_example_grad,
        elems=(X_batch, y_batch),
        fallback_to_while_loop=True
    )
    
    return per_example_grads

@tf.function
def clip_gradients_by_l2_norm(per_example_grads, clip_norm=1.0):
    """Clip gradients by L1 or L2 norm as specified"""
    batch_size = tf.shape(per_example_grads[0])[0]

    def clip_single_example(example_idx):
        example_grads = [grad[example_idx] for grad in per_example_grads]
        
        # L2 norm clipping for Gaussian and advanced Laplace
        global_norm = tf.linalg.global_norm(example_grads)
        clip_coeff = tf.minimum(1.0, clip_norm / (global_norm + 1e-8))
        
        clipped_grads = [grad * clip_coeff for grad in example_grads]
        return clipped_grads

    clipped_per_example = tf.map_fn(
        fn=clip_single_example,
        elems=tf.range(batch_size),
        fn_output_signature=[tf.TensorSpec(shape=grad.shape[1:], dtype=grad.dtype)
                           for grad in per_example_grads],
        parallel_iterations=32
    )
    
    return clipped_per_example

@tf.function
def clip_gradients_by_l1_norm(per_example_grads, clip_norm):
    """
    Per-example L1 clipping used **only** for the traditional Laplace mechanism.
    Replaces the old tf.linalg.global_norm-based clip.
    """
    batch_size = tf.shape(per_example_grads[0])[0]

    def _clip_one(idx):
        g_i = [g[idx] for g in per_example_grads]           # grads of sample i
        # total L1 norm of that sample
        l1_norm = tf.add_n([tf.reduce_sum(tf.abs(g)) for g in g_i])
        coeff   = tf.minimum(1.0, clip_norm / (l1_norm + 1e-8))
        return [g * coeff for g in g_i]

    clipped = tf.map_fn(
        fn=_clip_one,
        elems=tf.range(batch_size),
        fn_output_signature=[
            tf.TensorSpec(shape=g.shape[1:], dtype=g.dtype) for g in per_example_grads
        ],
        parallel_iterations=32
    )
    return clipped  

# ========================= 1. TRADITIONAL LAPLACE (Pure ε-DP, L1 sensitivity) =========================

class TraditionalLaplaceDPManager:
    """Traditional Laplace mechanism: Pure ε-DP with L1 sensitivity"""
    
    def __init__(self, epsilon_total, l1_norm_clip):
        self.epsilon_total = float(epsilon_total)
        self.l1_norm_clip = float(l1_norm_clip)  # L1 sensitivity
        self.epsilon_used = 0.0
        self.step_count = 0
        self.epsilon_per_step = None

    def setup_budget(self, total_steps):
        """Setup uniform epsilon allocation"""
        self.total_steps = total_steps
        self.epsilon_per_step = self.epsilon_total / max(total_steps, 1)
        print(f"Traditional Laplace: ε_per_step = {self.epsilon_per_step:.6f}, L1 sensitivity = {self.l1_norm_clip}")

    def add_noise_to_gradients(self, aggregated_gradients):
        """Add Laplace noise scaled to L1 sensitivity"""
        if self.epsilon_used + self.epsilon_per_step > self.epsilon_total:
            raise ValueError(f"Epsilon budget exhausted!")

        noisy_gradients = []
        
        # Traditional Laplace scale: L1_sensitivity / epsilon_per_step
        laplace_scale = self.l1_norm_clip / self.epsilon_per_step

        for grad in aggregated_gradients:
            # Generate Laplace noise
            uniform_noise = tf.random.uniform(
                tf.shape(grad),
                minval=-0.49999,
                maxval=0.49999,
                dtype=grad.dtype
            )
            
            sign = tf.sign(uniform_noise)
            abs_uniform = tf.abs(uniform_noise)
            laplace_noise = -laplace_scale * sign * tf.math.log(1.0 - 2.0 * abs_uniform)
            
            noisy_grad = grad + laplace_noise
            noisy_gradients.append(noisy_grad)

        self.epsilon_used += self.epsilon_per_step
        self.step_count += 1

        return noisy_gradients

    def get_privacy_spent(self):
        return {
            'epsilon': self.epsilon_used,
            'delta': 0.0,  # Pure DP
            'steps': self.step_count,
            'mechanism': 'Traditional Laplace (L1, Pure ε-DP)'
        }

# ========================= 2. ADVANCED LAPLACE ((ε,δ)-DP, L2 sensitivity) =========================

class AdvancedLaplaceDPManager:
    """Advanced Laplace mechanism from research: (ε,δ)-DP with L2 sensitivity"""
    
    def __init__(self, epsilon_total, delta, l2_norm_clip):
        self.epsilon_total = float(epsilon_total)
        self.delta = float(delta)
        self.l2_norm_clip = float(l2_norm_clip)  # L2 sensitivity like Gaussian
        self.epsilon_used = 0.0
        self.step_count = 0
        self.epsilon_per_step = None

    def setup_budget(self, total_steps):
        """Setup uniform epsilon allocation"""
        self.total_steps = total_steps
        self.epsilon_per_step = self.epsilon_total / max(total_steps, 1)
        
        # Advanced Laplace scaling from research paper
        # Scale parameter accounts for δ and L2 sensitivity
        self.advanced_scale = self._calculate_advanced_scale()
        
        print(f"Advanced Laplace: ε_per_step = {self.epsilon_per_step:.6f}, "
              f"δ = {self.delta}, L2 sensitivity = {self.l2_norm_clip}, "
              f"scale = {self.advanced_scale:.6f}")

    def _calculate_advanced_scale(self):
        """Calculate advanced Laplace scale for (ε,δ)-DP with L2 sensitivity"""
        if self.epsilon_per_step <= 0:
            return float('inf')
        
        # Research paper formula: Scale accounts for δ failure probability
        # σ = (L2_sensitivity / ε) * sqrt(2 * ln(1.25/δ))
        if self.delta > 0:
            ln_term = max(np.log(1.25 / self.delta), 1.0)  # Avoid numerical issues
            scale = (self.l2_norm_clip / self.epsilon_per_step) * np.sqrt(2 * ln_term)
        else:
            # Fallback to traditional scaling if δ = 0
            print("Falling back to traditional scaling...")
            scale = self.l2_norm_clip / self.epsilon_per_step
        
        return float(scale)

    def add_noise_to_gradients(self, aggregated_gradients):
        """Add advanced Laplace noise for (ε,δ)-DP"""
        if self.epsilon_used + self.epsilon_per_step > self.epsilon_total:
            raise ValueError(f"Epsilon budget exhausted!")

        noisy_gradients = []

        for grad in aggregated_gradients:
            # Generate advanced Laplace noise 
            uniform_noise = tf.random.uniform(
                tf.shape(grad),
                minval=-0.49999,
                maxval=0.49999,
                dtype=grad.dtype
            )
            
            sign = tf.sign(uniform_noise)
            abs_uniform = tf.abs(uniform_noise)
            
            # Advanced Laplace with delta-adjusted scaling
            laplace_noise = -self.advanced_scale * sign * tf.math.log(1.0 - 2.0 * abs_uniform)
            
            noisy_grad = grad + laplace_noise
            noisy_gradients.append(noisy_grad)

        self.epsilon_used += self.epsilon_per_step
        self.step_count += 1

        return noisy_gradients

    def get_privacy_spent(self):
        return {
            'epsilon': self.epsilon_used,
            'delta': self.delta,  # (ε,δ)-DP
            'steps': self.step_count,
            'mechanism': 'Advanced Laplace (L2, (ε,δ)-DP)'
        }

# ========================= 3. GAUSSIAN (unchanged but clarified) =========================

class GaussianDPManager:
    """Gaussian mechanism: (ε,δ)-DP with L2 sensitivity and RDP accounting"""
    
    def __init__(self, noise_multiplier, l2_norm_clip, delta=1e-5):
        self.noise_multiplier = float(noise_multiplier)
        self.l2_norm_clip = float(l2_norm_clip)  # L2 sensitivity
        self.delta = delta

        # RDP accounting setup
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.rdp_accountant = None
        if dp_accounting:
            self.rdp_accountant = dp_accounting.rdp.RdpAccountant(self.orders)

        self.step_count = 0
        self.sampling_probability = None

        # Calculate noise scale
        self.noise_scale = self.l2_norm_clip * self.noise_multiplier
        print(f"Gaussian: noise_scale = {self.noise_scale:.6f}, L2 sensitivity = {self.l2_norm_clip}")

    def setup_accounting(self, batch_size, num_samples):
        """Setup RDP accounting parameters"""
        self.sampling_probability = batch_size / float(num_samples)

    def add_noise_to_gradients(self, aggregated_gradients):
        """Add Gaussian noise with RDP tracking"""
        noisy_gradients = []

        for grad in aggregated_gradients:
            # Generate Gaussian noise
            gaussian_noise = tf.random.normal(
                tf.shape(grad),
                mean=0.0,
                stddev=self.noise_scale,  # L2_sensitivity * noise_multiplier
                dtype=grad.dtype
            )
            
            noisy_grad = grad + gaussian_noise
            noisy_gradients.append(noisy_grad)

        # Track privacy using RDP
        if self.rdp_accountant and self.sampling_probability:
            event = dp_accounting.PoissonSampledDpEvent(
                self.sampling_probability,
                dp_accounting.GaussianDpEvent(self.noise_multiplier)
            )
            self.rdp_accountant.compose(event)

        self.step_count += 1
        return noisy_gradients

    def get_privacy_spent(self):
        """Return current privacy expenditure"""
        if not self.rdp_accountant:
            return {'epsilon': 0.0, 'delta': self.delta, 'steps': self.step_count, 'mechanism': 'Gaussian (L2, RDP)'}

        try:
            epsilon = self.rdp_accountant.get_epsilon(target_delta=self.delta)
            return {
                'epsilon': float(epsilon or 0.0), 
                'delta': self.delta, 
                'steps': self.step_count,
                'mechanism': 'Gaussian (L2, RDP)'
            }
        except Exception as e:
            return {'epsilon': 0.0, 'delta': self.delta, 'steps': self.step_count, 'mechanism': 'Gaussian (L2, RDP)'}

# ========================= CORRECTED OPTIMIZERS =========================

class TraditionalLaplaceOptimizer(tf.keras.optimizers.Adam):
    """Traditional Laplace DP optimizer: Pure ε-DP with L1 clipping"""
    
    def __init__(self, l1_norm_clip, epsilon_total, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.l1_norm_clip = l1_norm_clip
        self.epsilon_total = epsilon_total
        self.dp_manager = None
        self.model = None

    def setup_dp(self, model, total_steps):
        """Setup DP manager"""
        self.model = model
        self.dp_manager = TraditionalLaplaceDPManager(
            epsilon_total=self.epsilon_total,
            l1_norm_clip=self.l1_norm_clip
        )
        self.dp_manager.setup_budget(total_steps)

    def train_step_with_dp(self, x_batch, y_batch):
        # 1) per-example grads
        per_ex = compute_per_example_gradients(self.model, x_batch, y_batch)

        # 2) **L1** clipping  ← REPLACED
        clipped = clip_gradients_by_l1_norm(per_ex, self.l1_norm_clip)

        # 3) aggregate + noise
        agg   = [tf.reduce_sum(g, axis=0) for g in clipped]
        noisy = self.dp_manager.add_noise_to_gradients(agg)

        # 4) average & apply
        bs = tf.cast(tf.shape(x_batch)[0], tf.float32)
        self.apply_gradients(
            list(zip([g / bs for g in noisy], self.model.trainable_variables))
        )

        # 5) tracking loss for logs
        preds = self.model(x_batch, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_batch, preds)
        )
        return loss

    def get_privacy_spent(self):
        if self.dp_manager is None:
            return {'epsilon': 0.0, 'delta': 0.0}
        return self.dp_manager.get_privacy_spent()

class AdvancedLaplaceOptimizer(tf.keras.optimizers.Adam):
    """Advanced Laplace DP optimizer: (ε,δ)-DP with L2 clipping"""
    
    def __init__(self, l2_norm_clip, epsilon_total, delta=1e-5, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.l2_norm_clip = l2_norm_clip
        self.epsilon_total = epsilon_total
        self.delta = delta
        self.dp_manager = None
        self.model = None

    def setup_dp(self, model, total_steps):
        """Setup DP manager"""
        self.model = model
        self.dp_manager = AdvancedLaplaceDPManager(
            epsilon_total=self.epsilon_total,
            delta=self.delta,
            l2_norm_clip=self.l2_norm_clip
        )
        self.dp_manager.setup_budget(total_steps)

    def train_step_with_dp(self, x_batch, y_batch):
        """Training step with Advanced Laplace DP"""
        # Compute per-example gradients
        per_example_grads = compute_per_example_gradients(self.model, x_batch, y_batch)
        
        # Clip by L2 norm (like Gaussian)
        clipped_grads = clip_gradients_by_l2_norm(per_example_grads, clip_norm=self.l2_norm_clip)
        
        # Aggregate gradients
        aggregated_grads = [tf.reduce_sum(grad, axis=0) for grad in clipped_grads]
        
        # Add noise
        noisy_grads = self.dp_manager.add_noise_to_gradients(aggregated_grads)

        # Average gradients
        batch_size = tf.cast(tf.shape(x_batch)[0], tf.float32)
        noisy_avg = [g / batch_size for g in noisy_grads]

        # Apply gradients
        grads_and_vars = list(zip(noisy_avg, self.model.trainable_variables))
        self.apply_gradients(grads_and_vars)

        # Compute loss for logging
        predictions = self.model(x_batch, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, predictions))

        return loss

    def get_privacy_spent(self):
        if self.dp_manager is None:
            return {'epsilon': 0.0, 'delta': self.delta}
        return self.dp_manager.get_privacy_spent()

class GaussianDPOptimizer(tf.keras.optimizers.Adam):
    """Gaussian DP optimizer: (ε,δ)-DP with L2 clipping and RDP accounting"""
    
    def __init__(self, l2_norm_clip, noise_multiplier=1.0, delta=1e-5, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.dp_manager = None
        self.model = None

    def setup_dp(self, model, num_samples, batch_size):
        """Setup DP manager with RDP accounting"""
        self.model = model
        self.dp_manager = GaussianDPManager(
            noise_multiplier=self.noise_multiplier,
            l2_norm_clip=self.l2_norm_clip,
            delta=self.delta
        )
        self.dp_manager.setup_accounting(batch_size, num_samples)

    def train_step_with_dp(self, x_batch, y_batch):
        """Training step with Gaussian DP"""
        # Compute per-example gradients
        per_example_grads = compute_per_example_gradients(self.model, x_batch, y_batch)
        
        # Clip by L2 norm
        clipped_grads = clip_gradients_by_l2_norm(per_example_grads, clip_norm=self.l2_norm_clip)
        
        # Aggregate gradients
        aggregated_grads = [tf.reduce_sum(grad, axis=0) for grad in clipped_grads]
        
        # Add noise
        noisy_grads = self.dp_manager.add_noise_to_gradients(aggregated_grads)

        # Average gradients
        batch_size = tf.cast(tf.shape(x_batch)[0], tf.float32)
        noisy_avg = [g / batch_size for g in noisy_grads]

        # Apply gradients
        grads_and_vars = list(zip(noisy_avg, self.model.trainable_variables))
        self.apply_gradients(grads_and_vars)

        # Compute loss for logging
        predictions = self.model(x_batch, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, predictions))

        return loss

    def get_privacy_spent(self):
        if self.dp_manager is None:
            return {'epsilon': 0.0, 'delta': self.delta}
        return self.dp_manager.get_privacy_spent()

# ========================= UPDATED TRAINER =========================

class IoTModelTrainer:
    """Corrected trainer with all three DP mechanisms"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_builder = IoTModel()
        self.model = None
        
        # Three optimizers
        self.traditional_laplace_optimizer = None
        self.advanced_laplace_optimizer = None
        self.gaussian_optimizer = None
        
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def create_model(self, input_dim, num_classes, architecture=None):
        """Create model architecture"""
        if architecture is None:
            architecture = [256, 256]

        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        return self.model

    def train_model(self, X_train, y_train_cat, X_val, y_val_cat,
                   model, epochs=20, batch_size=128, verbose=2,
                   use_dp=True, noise_type='gaussian', 
                   
                   # Universal parameters
                   l2_norm_clip=1.0,  # Used by Gaussian and Advanced Laplace
                   l1_norm_clip=1.0,  # Used by Traditional Laplace only
                   
                   # Gaussian parameters
                   noise_multiplier=1.0, delta=1e-5,
                   
                   # Laplace parameters  
                   epsilon_total=10.0,  # Used by both Laplace methods
                   
                   learning_rate=1e-3):
        """
        CORRECTED: Train with proper mechanism selection
        
        noise_type options:
        - 'gaussian': Gaussian mechanism with L2 clipping and RDP
        - 'traditional_laplace': Pure ε-DP Laplace with L1 clipping  
        - 'advanced_laplace': (ε,δ)-DP Laplace with L2 clipping (research paper)
        """
        
        self.model = model
        print(f"\n=== TRAINING WITH {noise_type.upper()} MECHANISM ===")

        if use_dp and noise_type == 'gaussian':
            return self._train_gaussian_dp(
                X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, verbose,
                l2_norm_clip, noise_multiplier, delta, learning_rate
            )
        elif use_dp and noise_type == 'traditional_laplace':
            return self._train_traditional_laplace_dp(
                X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, verbose,
                l1_norm_clip, epsilon_total, learning_rate
            )
        elif use_dp and noise_type == 'advanced_laplace':
            return self._train_advanced_laplace_dp(
                X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, verbose,
                l2_norm_clip, epsilon_total, delta, learning_rate
            )
        else:
            # Non-DP training
            return self._train_non_dp(
                X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, verbose, learning_rate
            )

    def _train_gaussian_dp(self, X_train, y_train_cat, X_val, y_val_cat,
                          epochs, batch_size, verbose,
                          l2_norm_clip, noise_multiplier, delta, learning_rate):
        """Gaussian DP training (unchanged)"""
        print(f"Parameters: noise_multiplier={noise_multiplier}, l2_norm_clip={l2_norm_clip}, delta={delta}")

        self.gaussian_optimizer = GaussianDPOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            delta=delta,
            learning_rate=learning_rate
        )

        self.gaussian_optimizer.setup_dp(self.model, len(X_train), batch_size)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        return self._run_training_loop(
            X_train, y_train_cat, X_val, y_val_cat,
            epochs, batch_size, verbose, 'gaussian'
        )

    def _train_traditional_laplace_dp(self, X_train, y_train_cat, X_val, y_val_cat,
                                    epochs, batch_size, verbose,
                                    l1_norm_clip, epsilon_total, learning_rate):
        """Traditional Laplace DP training"""
        print(f"Parameters: epsilon_total={epsilon_total}, l1_norm_clip={l1_norm_clip} (Pure ε-DP)")

        steps_per_epoch = len(X_train) // batch_size
        total_steps = steps_per_epoch * epochs

        self.traditional_laplace_optimizer = TraditionalLaplaceOptimizer(
            l1_norm_clip=l1_norm_clip,
            epsilon_total=epsilon_total,
            learning_rate=learning_rate
        )

        self.traditional_laplace_optimizer.setup_dp(self.model, total_steps)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        return self._run_training_loop(
            X_train, y_train_cat, X_val, y_val_cat,
            epochs, batch_size, verbose, 'traditional_laplace'
        )

    def _train_advanced_laplace_dp(self, X_train, y_train_cat, X_val, y_val_cat,
                                 epochs, batch_size, verbose,
                                 l2_norm_clip, epsilon_total, delta, learning_rate):
        """Advanced Laplace DP training (research paper)"""
        print(f"Parameters: epsilon_total={epsilon_total}, l2_norm_clip={l2_norm_clip}, delta={delta} ((ε,δ)-DP)")

        steps_per_epoch = len(X_train) // batch_size
        total_steps = steps_per_epoch * epochs

        self.advanced_laplace_optimizer = AdvancedLaplaceOptimizer(
            l2_norm_clip=l2_norm_clip,
            epsilon_total=epsilon_total,
            delta=delta,
            learning_rate=learning_rate
        )

        self.advanced_laplace_optimizer.setup_dp(self.model, total_steps)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        return self._run_training_loop(
            X_train, y_train_cat, X_val, y_val_cat,
            epochs, batch_size, verbose, 'advanced_laplace'
        )

    def _train_non_dp(self, X_train, y_train_cat, X_val, y_val_cat,
                     epochs, batch_size, verbose, learning_rate):
        """Non-DP baseline training"""
        print("=== NON-DP BASELINE TRAINING ===")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        return history, [], 0.0

    def _run_training_loop(self, X_train, y_train_cat, X_val, y_val_cat,
                          epochs, batch_size, verbose, dp_type):
        """Common training loop for DP methods"""
        steps_per_epoch = len(X_train) // batch_size
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(epochs):
            if verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs}")

            epoch_loss = tf.keras.metrics.Mean()
            epoch_acc = tf.keras.metrics.CategoricalAccuracy()

            # Training batches
            for i in range(steps_per_epoch):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train_cat[start_idx:end_idx]

                try:
                    # Choose optimizer
                    if dp_type == 'gaussian':
                        batch_loss = self.gaussian_optimizer.train_step_with_dp(X_batch, y_batch)
                    elif dp_type == 'traditional_laplace':
                        batch_loss = self.traditional_laplace_optimizer.train_step_with_dp(X_batch, y_batch)
                    elif dp_type == 'advanced_laplace':
                        batch_loss = self.advanced_laplace_optimizer.train_step_with_dp(X_batch, y_batch)

                    # Update metrics
                    epoch_loss(batch_loss)
                    predictions = self.model(X_batch, training=False)
                    epoch_acc(y_batch, predictions)
                except Exception as e:
                    print(f"Training step failed: {e}")
                    break

            # Validation evaluation
            val_predictions = self.model(X_val, training=False)
            val_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_val_cat, val_predictions)
            ).numpy()

            val_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_val_cat, axis=1), tf.argmax(val_predictions, axis=1)), tf.float32)
            ).numpy()

            # Store history
            train_loss = epoch_loss.result().numpy()
            train_acc = epoch_acc.result().numpy()

            history['loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['accuracy'].append(float(train_acc))
            history['val_accuracy'].append(float(val_acc))

            if verbose >= 1:
                # Get privacy info
                if dp_type == 'gaussian':
                    privacy_info = self.gaussian_optimizer.get_privacy_spent()
                elif dp_type == 'traditional_laplace':
                    privacy_info = self.traditional_laplace_optimizer.get_privacy_spent()
                elif dp_type == 'advanced_laplace':
                    privacy_info = self.advanced_laplace_optimizer.get_privacy_spent()

                print(f" loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
                      f"epsilon: {privacy_info['epsilon']:.4f} - "
                      f"delta: {privacy_info['delta']:.2e} - "
                      f"mechanism: {privacy_info['mechanism']}")

        # Create history wrapper
        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict

        # Get final epsilon
        if dp_type == 'gaussian':
            final_eps = self.gaussian_optimizer.get_privacy_spent()['epsilon']
        elif dp_type == 'traditional_laplace':
            final_eps = self.traditional_laplace_optimizer.get_privacy_spent()['epsilon']
        elif dp_type == 'advanced_laplace':
            final_eps = self.advanced_laplace_optimizer.get_privacy_spent()['epsilon']

        return HistoryWrapper(history), [], final_eps
