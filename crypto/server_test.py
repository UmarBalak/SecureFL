import tensorflow as tf
from crypto.phe_provider import PHEProvider
from crypto.fhe_provider import FHEProvider

# Simulate receiving encrypted weights
with open("encrypted_weights.bin", "rb") as f:
    import dill
    encrypted_weights, shape = dill.load(f)

# ======== Choose Decryption Method ========
USE_PHE = True  # Match client setting

if USE_PHE:
    crypto = PHEProvider.load_keys("phe_keys.json")
else:
    crypto = FHEProvider()

# ======== Decrypt ========
decrypted_tensor = crypto.decrypt_tensor(encrypted_weights, shape)

print("\n✅ Decrypted Tensor Received at Server:")
print(decrypted_tensor.numpy())
