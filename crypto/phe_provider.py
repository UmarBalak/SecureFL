import phe
import json
import numpy as np
import tensorflow as tf
from base import BaseCryptoProvider

class PHEProvider(BaseCryptoProvider):
    def __init__(self, public_key, private_key=None):
        self.public_key = public_key
        self.private_key = private_key

    @classmethod
    def generate(cls):
        pub, priv = phe.paillier.generate_paillier_keypair()
        return cls(pub, priv)

    def encrypt(self, value: float):
        return self.public_key.encrypt(value)

    def decrypt(self, encrypted):
        if not self.private_key:
            raise ValueError("No private key provided")
        return self.private_key.decrypt(encrypted)

    def save_keys(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'n': str(self.public_key.n),
                'p': str(self.private_key.p) if self.private_key else None,
                'q': str(self.private_key.q) if self.private_key else None
            }, f)

    def save_keys_to_memory(self):
        """Save keys to memory."""
        return {
            'n': str(self.public_key.n),
            'p': str(self.private_key.p) if self.private_key else None,
            'q': str(self.private_key.q) if self.private_key else None
        }

    @classmethod
    def load_keys(cls, path: str):
        with open(path) as f:
            keys = json.load(f)
            pub = phe.paillier.PaillierPublicKey(n=int(keys['n']))
            priv = None
            if keys['p'] and keys['q']:
                priv = phe.paillier.PaillierPrivateKey(pub, int(keys['p']), int(keys['q']))
            return cls(pub, priv)

    @classmethod
    def load_keys_from_memory(cls, keys):
        """Load keys from memory."""
        pub = phe.paillier.PaillierPublicKey(n=int(keys['n']))
        priv = None
        if keys['p'] and keys['q']:
            priv = phe.paillier.PaillierPrivateKey(pub, int(keys['p']), int(keys['q']))
        return cls(pub, priv)

    def encrypt_tensor(self, tensor):
        # Convert the tensor to a numpy array
        numpy_array = tensor.numpy() if isinstance(tensor, tf.Tensor) else np.array(tensor)

        # Flatten the numpy array
        flat_array = numpy_array.flatten()

        # Convert numpy.float32 to Python float and encrypt each element
        encrypted_array = [self.encrypt(float(x)) for x in flat_array]

        # Return the encrypted array and the original shape
        return encrypted_array, numpy_array.shape

    def decrypt_tensor(self, encrypted_weights, shape):
        # Decrypt each element in the encrypted array
        decrypted_flat_array = [self.decrypt(x) for x in encrypted_weights]

        # Reshape the decrypted array back to the original shape
        decrypted_tensor = np.array(decrypted_flat_array).reshape(shape)

        # Convert to TensorFlow tensor
        return tf.convert_to_tensor(decrypted_tensor, dtype=tf.float32)
