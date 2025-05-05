import tensorflow as tf
from phe_provider import PHEProvider
from fhe_provider import FHEProvider
import dill
import numpy as np

# Dummy TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(2,))
])
weights = model.get_weights()[0]  # Shape: (2, 2)

print("\n🎯 Original Model Weights:")
print(weights)

global original_weights, transmitted_data, phe_keys, fhe_keys  # Declare as global to make accessible in the server script
# Save original weights for comparison
original_weights = weights

# ======== Choose Encryption Method ========
USE_PHE = True  # Set to False for FHE

if USE_PHE:
    crypto = PHEProvider.generate()
else:
    crypto = FHEProvider.generate()

# ======== Encrypt Weights ========
encrypted_weights, shape = crypto.encrypt_tensor(tf.convert_to_tensor(weights, dtype=tf.float32))

# Simulate transmission by storing in memory
transmitted_data = (encrypted_weights, shape)  # Store as a tuple directly in memory

if USE_PHE:
    phe_keys = crypto.save_keys_to_memory()
else:
    fhe_keys = crypto.save_keys_to_memory()

print("\n🔐 Encrypted weights and keys prepared for transmission.")

def get_simulated_data():
    """Return the simulated data and keys for transmission."""
    return original_weights, transmitted_data, phe_keys if USE_PHE else fhe_keys