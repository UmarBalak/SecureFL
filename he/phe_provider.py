import concurrent.futures
import json
import logging
import time
import os
import hmac
import hashlib
from typing import List, Tuple, Dict, Any, Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required")

try:
    import phe
except ImportError:
    raise ImportError("Install python-paillier for PHE support.")

from .base import BaseCryptoProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PHEProvider")

class PHEProvider(BaseCryptoProvider):
    def __init__(self, public_key: Optional[phe.paillier.PaillierPublicKey] = None,
                 private_key: Optional[phe.paillier.PaillierPrivateKey] = None,
                 key_size: int = 2048, precision: int = 32,
                 scaling_factor: int = 1000, max_workers: int = None):
        self.public_key = public_key
        self.private_key = private_key
        self.key_size = key_size
        self.precision = precision
        self.scaling_factor = scaling_factor
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        self.hmac_key = os.urandom(32)  # For integrity checks

        if self.public_key is None:
            logger.info(f"Generating new Paillier keypair with key_size={key_size}")
            self.public_key, self.private_key = phe.paillier.generate_paillier_keypair(n_length=key_size)

        phe.paillier.EncodedNumber.DEFAULT_PRECISION = precision
        logger.info(f"PHE Provider initialized with precision={precision}, scaling_factor={scaling_factor}, max_workers={self.max_workers}")

    def encrypt(self, value: float) -> Dict[str, Any]:
        if self.public_key is None:
            raise ValueError("Public key not available")
        scaled_value = int(value * self.scaling_factor)
        try:
            encrypted = self.public_key.encrypt(scaled_value)
            hmac_value = hmac.new(self.hmac_key, str(encrypted.ciphertext()).encode(), hashlib.sha256).hexdigest()
            return {"encrypted": encrypted, "hmac": hmac_value}
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError(f"Encryption failed: {str(e)}")

    def decrypt(self, encrypted_dict: Dict[str, Any]) -> float:
        if self.private_key is None:
            raise ValueError("No private key provided")
        encrypted = encrypted_dict["encrypted"]
        received_hmac = encrypted_dict["hmac"]
        expected_hmac = hmac.new(self.hmac_key, str(encrypted.ciphertext()).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(received_hmac.encode(), expected_hmac.encode()):
            raise ValueError("Integrity check failed")
        try:
            scaled_value = self.private_key.decrypt(encrypted)
            return scaled_value / self.scaling_factor
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Decryption failed: {str(e)}")

    def _encrypt_batch(self, values: List[Union[float, int]]) -> List[Dict[str, Any]]:
        return [self.encrypt(float(v)) for v in values]

    def _decrypt_batch(self, encrypted_values: List[Dict[str, Any]]) -> List[float]:
        return [self.decrypt(e) for e in encrypted_values]

    def encrypt_tensor(self, tensor: Union[tf.Tensor, np.ndarray]) -> Tuple[List[Dict[str, Any]], Tuple]:
        logger.info("Starting encryption of tensor")
        numpy_array = tensor.numpy() if isinstance(tensor, tf.Tensor) else np.array(tensor)
        shape = numpy_array.shape
        flat_array = numpy_array.flatten()
        total_len = len(flat_array)
        logger.info(f"Tensor shape: {shape}, total elements: {total_len}")

        batch_size = max(1, total_len // self.max_workers)
        batches = [flat_array[i:i + batch_size] for i in range(0, total_len, batch_size)]
        logger.info(f"Split into {len(batches)} batches, batch size: {batch_size}")

        encrypted_batches = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._encrypt_batch, batch) for batch in batches]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()
                    encrypted_batches.append(result)
                    logger.info(f"Batch {i}/{len(batches)} encrypted successfully")
                except Exception as e:
                    logger.error(f"Error in batch {i}: {e}")
                    raise

        encrypted_array = [item for batch in encrypted_batches for item in batch]
        logger.info(f"Encryption complete. Total encrypted elements: {len(encrypted_array)}")
        return encrypted_array, shape

    def decrypt_tensor(self, encrypted_weights: List[Dict[str, Any]], shape: Tuple) -> tf.Tensor:
        batch_size = max(1, len(encrypted_weights) // self.max_workers)
        batches = [encrypted_weights[i:i + batch_size] for i in range(0, len(encrypted_weights), batch_size)]

        decrypted_batches = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._decrypt_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                decrypted_batches.append(future.result())

        decrypted_flat_array = [item for batch in decrypted_batches for item in batch]
        decrypted_tensor = np.array(decrypted_flat_array).reshape(shape)
        return tf.convert_to_tensor(decrypted_tensor, dtype=tf.float32)

    def secure_add(self, encrypted_a: Dict[str, Any], encrypted_b: Dict[str, Any]) -> Dict[str, Any]:
        result = encrypted_a["encrypted"] + encrypted_b["encrypted"]
        hmac_value = hmac.new(self.hmac_key, str(result.ciphertext()).encode(), hashlib.sha256).hexdigest()
        return {"encrypted": result, "hmac": hmac_value}

    def secure_weighted_add(self, encrypted_a: Dict[str, Any], weight: float) -> Dict[str, Any]:
        scaled_weight = int(weight * 1000)  # Scale weight to integer
        result = encrypted_a["encrypted"] * scaled_weight
        hmac_value = hmac.new(self.hmac_key, str(result.ciphertext()).encode(), hashlib.sha256).hexdigest()
        return {"encrypted": result, "hmac": hmac_value}

    def secure_weighted_sum(self, encrypted_tensors: List[List[Dict[str, Any]]], weights: List[float]) -> List[Dict[str, Any]]:
        if not encrypted_tensors or not weights or len(encrypted_tensors) != len(weights):
            raise ValueError("Invalid inputs for weighted sum")
        
        result_length = len(encrypted_tensors[0])
        for tensor in encrypted_tensors:
            if len(tensor) != result_length:
                raise ValueError("All tensors must have the same length")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for pos in range(result_length):
                futures.append(executor.submit(
                    self._compute_position_weighted_sum,
                    [tensor[pos] for tensor in encrypted_tensors],
                    weights
                ))
            result = [future.result() for future in futures]
        
        return result

    def _compute_position_weighted_sum(self, position_values: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        result = self.secure_weighted_add(position_values[0], weights[0])["encrypted"]
        for i in range(1, len(position_values)):
            weighted_val = self.secure_weighted_add(position_values[i], weights[i])["encrypted"]
            result += weighted_val
        hmac_value = hmac.new(self.hmac_key, str(result.ciphertext()).encode(), hashlib.sha256).hexdigest()
        return {"encrypted": result, "hmac": hmac_value}

    def save_keys(self, path: str, password: str = None) -> None:
        keys_dict = self.save_keys_to_memory()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if password:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'salt_', iterations=100000)
            key = kdf.derive(password.encode())
            fernet = Fernet(Fernet.generate_key())
            encrypted_data = fernet.encrypt(json.dumps(keys_dict).encode())
            with open(path, 'wb') as f:
                f.write(encrypted_data)
        else:
            with open(path, 'w') as f:
                json.dump(keys_dict, f)
        logger.info(f"PHE keys saved to {path}")

    def save_keys_to_memory(self) -> Dict[str, Any]:
        return {
            'n': str(self.public_key.n) if self.public_key else None,
            'p': str(self.private_key.p) if self.private_key else None,
            'q': str(self.private_key.q) if self.private_key else None,
            'precision': self.precision,
            'key_size': self.key_size,
            'scaling_factor': self.scaling_factor,
            'hmac_key': self.hmac_key.hex()
        }

    @classmethod
    def generate(cls, key_size: int = 2048, precision: int = 24, scaling_factor: int = 1000,
                 max_workers: int = None) -> 'PHEProvider':
        return cls(key_size=key_size, precision=precision, scaling_factor=scaling_factor, max_workers=max_workers)

    @classmethod
    def load_keys(cls, path: str, password: str = None) -> 'PHEProvider':
        if not os.path.exists(path):
            logger.warning(f"PHE keys file {path} not found. Generating new keys.")
            return cls.generate()
        
        with open(path, 'rb' if password else 'r') as f:
            if password:
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'salt_', iterations=100000)
                key = kdf.derive(password.encode())
                fernet = Fernet(Fernet.generate_key())
                keys = json.loads(fernet.decrypt(f.read()).decode())
            else:
                keys = json.load(f)
            return cls.load_keys_from_memory(keys)

    @classmethod
    def load_keys_from_memory(cls, keys: Dict[str, Any]) -> 'PHEProvider':
        if not keys.get('n'):
            logger.warning("Missing key information. Generating new keys.")
            return cls.generate(
                key_size=keys.get('key_size', 2048),
                precision=keys.get('precision', 24),
                scaling_factor=keys.get('scaling_factor', 1000)
            )
        
        pub = phe.paillier.PaillierPublicKey(n=int(keys['n']))
        priv = None
        if keys.get('p') and keys.get('q'):
            priv = phe.paillier.PaillierPrivateKey(pub, int(keys['p']), int(keys['q']))
        
        provider = cls(
            public_key=pub,
            private_key=priv,
            key_size=keys.get('key_size', 2048),
            precision=keys.get('precision', 24),
            scaling_factor=keys.get('scaling_factor', 1000)
        )
        provider.hmac_key = bytes.fromhex(keys.get('hmac_key', os.urandom(32).hex()))
        return provider