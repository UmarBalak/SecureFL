import tensorflow as tf
from phe_provider import PHEProvider
from fhe_provider import FHEProvider
import numpy as np
from client_test import get_simulated_data  # Import the function to retrieve simulated data and keys

# Simulate receiving encrypted weights and keys
original_weights, transmitted_data, keys = get_simulated_data()
encrypted_weights, shape = transmitted_data  # Unpack the tuple directly

# ======== Choose Decryption Method ========
USE_PHE = True  # Match client setting

if USE_PHE:
    crypto = PHEProvider.load_keys_from_memory(keys)
else:
    crypto = FHEProvider.load_keys_from_memory(keys)

# ======== Decrypt ========
decrypted_tensor = crypto.decrypt_tensor(encrypted_weights, shape)

print("\n✅ Decrypted Tensor Received at Server:")
print(decrypted_tensor.numpy())

# Compare with original (for testing purposes)
if original_weights is not None:
    print("\n🔍 Comparison with original:")
    print(f"Mean absolute error: {np.mean(np.abs(original_weights - decrypted_tensor.numpy()))}")