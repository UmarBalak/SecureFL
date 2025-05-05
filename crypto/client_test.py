import tensorflow as tf
from phe_provider import PHEProvider
from fhe_provider import FHEProvider

# Dummy TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(2,))
])
weights = model.get_weights()[0]  # Shape: (2, 2)

print("\n🎯 Original Model Weights:")
print(weights)

# ======== Choose Encryption Method ========
USE_PHE = True  # Set to False for FHE

if USE_PHE:
    crypto = PHEProvider.generate()
    crypto.save_keys("phe_keys.json")
else:
    crypto = FHEProvider()

# ======== Encrypt Weights ========
encrypted_weights, shape = crypto.encrypt_tensor(tf.convert_to_tensor(weights, dtype=tf.float32))

# Simulate transmission
with open("encrypted_weights.bin", "wb") as f:
    import dill
    dill.dump((encrypted_weights, shape), f)

print("\n🔐 Encrypted weights sent to server.")
