import os
import json
import pickle
import numpy as np
import concurrent.futures
from typing import List, Tuple, Dict, Any, Union
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import hmac
import hashlib
import pkg_resources
from decimal import Decimal, getcontext, ROUND_HALF_UP

getcontext().prec = 32

try:
    import concrete.numpy as cnp
    from concrete.numpy import EncryptionStatus
except ImportError:
    raise ImportError("Install concrete-numpy for FHE support.")

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required")

from .base import BaseCryptoProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FHEProvider")

try:
    concrete_version = pkg_resources.get_distribution("concrete-numpy").version
    logger.info(f"Using Concrete-Numpy version {concrete_version}")
except pkg_resources.DistributionNotFound:
    logger.warning("Concrete-Numpy version could not be determined.")

class FHEProvider(BaseCryptoProvider):
    def __init__(self, min_range: int = -100000000, max_range: int = 100000000, 
                 scaling_factor: int = 10000000, adaptive_scaling: bool = False,
                 max_workers: int = None):
        self.min_range = min_range
        self.max_range = max_range
        self.base_scaling_factor = scaling_factor
        self.scaling_factor = scaling_factor
        self.adaptive_scaling = adaptive_scaling
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        self.hmac_key = os.urandom(32)
        self._circuit_cache = {}
        self._compile_circuits()
        logger.info(f"FHE Provider initialized with scaling_factor={scaling_factor}, adaptive_scaling={adaptive_scaling}")

    def _compile_circuits(self):
        def identity(x):
            return x
        logger.info("Compiling FHE identity circuit...")
        compiler = cnp.Compiler(identity, {"x": EncryptionStatus.ENCRYPTED})
        step = 10
        inputset = [np.array([i]) for i in range(self.min_range, self.max_range + 1, step)]
        self._circuit_cache['identity'] = compiler.compile(inputset)
        logger.info("FHE identity circuit compiled.")

        def weighted_sum(x, w):
            return np.sum(x * w)
        logger.info("Compiling FHE weighted sum circuit...")
        compiler = cnp.Compiler(weighted_sum, {
            "x": EncryptionStatus.ENCRYPTED,
            "w": EncryptionStatus.CLEAR
        })
        x_range = range(-10000, 10001, 100)
        w_range = range(0, 1001, 50)
        inputset = [(np.array([x] * 5), np.array([w] * 5)) for x in x_range for w in w_range]
        self._circuit_cache['weighted_sum'] = compiler.compile(inputset)
        logger.info("FHE weighted sum circuit compiled.")

    def _adjust_scaling_factor(self, data: np.ndarray) -> float:
        if not self.adaptive_scaling:
            return self.base_scaling_factor
        abs_max = np.max(np.abs(data))
        if abs_max == 0:
            return self.base_scaling_factor
        circuit_range = self.max_range / 2
        optimal_scale = circuit_range / abs_max
        min_scale = 100000
        max_scale = 10000000
        scaling = max(min(optimal_scale, max_scale), min_scale)
        logger.debug(f"Adjusted scaling factor to {scaling} (data range: {abs_max})")
        return scaling

    def quantize_tensor(self, tensor: Union[np.ndarray, tf.Tensor], scaling_factor: float = None) -> np.ndarray:
        scaling_factor = scaling_factor or self.scaling_factor
        return np.round(np.array(tensor) * scaling_factor).astype(np.int64)

    def dequantize_tensor(self, tensor: np.ndarray, scaling_factor: float = None) -> np.ndarray:
        scaling_factor = scaling_factor or self.scaling_factor
        return tensor.astype(np.float32) / scaling_factor

    def encrypt(self, value: float) -> Dict[str, Any]:
        scaled_value = int(Decimal(str(value)) * Decimal(str(self.scaling_factor)).to_integral_value(rounding=ROUND_HALF_UP))
        scaled_value = max(min(scaled_value, self.max_range), self.min_range)
        logger.debug(f"Encrypting value={value}, scaled_value={scaled_value}, scaling_factor={self.scaling_factor}")
        # logger.info(f"Encrypting value={value}, scaled_value={scaled_value}, scaling_factor={self.scaling_factor}")

        try:
            circuit = self._circuit_cache['identity']
            encrypted_input = circuit.encrypt(np.array([scaled_value]))
            encrypted_result = circuit.run(encrypted_input)
            encrypted_data = {
                'value': scaled_value,
                'circuit_id': 'identity',
                'scaling_factor': self.scaling_factor
            }
            serialized_result = pickle.dumps(encrypted_data)
            hmac_value = hmac.new(self.hmac_key, serialized_result, hashlib.sha256).hexdigest()
            return {"encrypted": serialized_result, "hmac": hmac_value}
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError(f"Encryption failed: {str(e)}")

    def decrypt(self, encrypted_dict: Dict[str, Any]) -> float:
        serialized_encrypted = encrypted_dict["encrypted"]
        received_hmac = encrypted_dict["hmac"]
        expected_hmac = hmac.new(self.hmac_key, serialized_encrypted, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(received_hmac.encode(), expected_hmac.encode()):
            raise ValueError("Integrity check failed")
        try:
            encrypted_data = pickle.loads(serialized_encrypted)
            if encrypted_data['circuit_id'] != 'identity':
                raise ValueError("Invalid circuit ID")
            scaled_value = encrypted_data['value']
            scaling_factor = encrypted_data['scaling_factor']
            logger.debug(f"Decrypting scaled_value={scaled_value}, scaling_factor={scaling_factor}")
            decrypted_value = float(Decimal(str(scaled_value)) / Decimal(str(scaling_factor)))
            logger.debug(f"Decrypted value={decrypted_value}")
            return decrypted_value
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Decryption failed: {str(e)}")

    def _encrypt_batch(self, values: List[float]) -> List[Dict[str, Any]]:
        return [self.encrypt(val) for val in values]

    def _decrypt_batch(self, encrypted_values: List[Dict[str, Any]]) -> List[float]:
        return [self.decrypt(enc) for enc in encrypted_values]

    def encrypt_tensor(self, tensor: Union[tf.Tensor, np.ndarray]) -> Tuple[List[Dict[str, Any]], Tuple]:
        numpy_array = tensor.numpy() if isinstance(tensor, tf.Tensor) else np.array(tensor)
        self.scaling_factor = self._adjust_scaling_factor(numpy_array)
        logger.info(f"Encrypting tensor with scaling_factor={self.scaling_factor}")
        
        flat_array = numpy_array.astype(np.float64).flatten()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            encrypted_flat = list(executor.map(self.encrypt, flat_array))
        
        original_shape = numpy_array.shape
        return encrypted_flat, original_shape
    
    def decrypt_tensor(self, encrypted_flat: List[Dict[str, Any]], original_shape: Tuple) -> np.ndarray:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            decrypted_flat = list(executor.map(self.decrypt, encrypted_flat))
        
        decrypted_array = np.array(decrypted_flat).reshape(original_shape)
        return decrypted_array

    def secure_weighted_sum(self, encrypted_tensors: List[List[Dict[str, Any]]], weights: List[float]) -> List[Dict[str, Any]]:
        if not encrypted_tensors or not weights or len(encrypted_tensors) != len(weights):
            raise ValueError("Invalid inputs for weighted sum")
        
        result_length = len(encrypted_tensors[0])
        for tensor in encrypted_tensors:
            if len(tensor) != result_length:
                raise ValueError("All tensors must have the same length")

        scaled_weights = [int(w * 1000) for w in weights]
        circuit = self._circuit_cache['weighted_sum']
        result = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for pos in range(result_length):
                serialized_vals = [tensor[pos]["encrypted"] for tensor in encrypted_tensors]
                futures.append(executor.submit(
                    self._compute_position_weighted_sum,
                    serialized_vals,
                    scaled_weights,
                    circuit
                ))
            for future in concurrent.futures.as_completed(futures):
                result.append(future.result())
        
        return result

    def _compute_position_weighted_sum(self, serialized_vals: List[bytes], scaled_weights: List[int], circuit: Any) -> Dict[str, Any]:
        try:
            encrypted_vals = []
            scaling_factors = []
            for val in serialized_vals:
                encrypted_data = pickle.loads(val)
                if encrypted_data['circuit_id'] != 'identity':
                    raise ValueError("Invalid circuit ID")
                scaled_value = encrypted_data['value']
                scaling_factor = encrypted_data['scaling_factor']
                logger.debug(f"Weighted sum: scaled_value={scaled_value}, scaling_factor={scaling_factor}")
                identity_circuit = self._circuit_cache['identity']
                encrypted_input = circuit.encrypt(np.array([scaled_value]))
                encrypted_result = circuit.run(encrypted_input)
                encrypted_vals.append(encrypted_result)
                scaling_factors.append(scaling_factor)
            encrypted_input = circuit.encrypt(np.array([v for v in encrypted_vals]))
            result = circuit.run(encrypted_input, np.array(scaled_weights))
            result_value = circuit.decrypt(result)[0]
            result_data = {
                'value': int(result_value),
                'circuit_id': 'weighted_sum',
                'scaling_factor': scaling_factors[0]
            }
            serialized_result = pickle.dumps(result_data)
            hmac_value = hmac.new(self.hmac_key, serialized_result, hashlib.sha256).hexdigest()
            return {"encrypted": serialized_result, "hmac": hmac_value}
        except Exception as e:
            logger.error(f"Weighted sum computation failed: {e}")
            raise ValueError(f"Weighted sum failed: {str(e)}")

    def save_keys(self, path: str, password: str = None) -> None:
        params = self.save_keys_to_memory()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if password:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'salt_', iterations=100000)
            key = kdf.derive(password.encode())
            fernet = Fernet(Fernet.generate_key())
            encrypted_data = fernet.encrypt(json.dumps(params).encode())
            with open(path, 'wb') as f:
                f.write(encrypted_data)
        else:
            with open(path, 'w') as f:
                json.dump(params, f)
        logger.info(f"FHE parameters saved to {path}")

    def save_keys_to_memory(self) -> Dict[str, Any]:
        return {
            "base_scaling_factor": self.base_scaling_factor,
            "scaling_factor": self.scaling_factor,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "adaptive_scaling": self.adaptive_scaling,
            "max_workers": self.max_workers,
            "hmac_key": self.hmac_key.hex()
        }

    @classmethod
    def generate(cls, min_range: int = -100000000, max_range: int = 100000000, 
                 scaling_factor: int = 10000000, adaptive_scaling: bool = False,
                 max_workers: int = None) -> 'FHEProvider':
        return cls(min_range, max_range, scaling_factor, adaptive_scaling, max_workers)

    @classmethod
    def load_keys(cls, path: str, password: str = None) -> 'FHEProvider':
        if not os.path.exists(path):
            logger.warning(f"FHE parameters file {path} not found. Using default parameters.")
            return cls()
        with open(path, 'rb' if password else 'r') as f:
            if password:
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'salt_', iterations=100000)
                key = kdf.derive(password.encode())
                fernet = Fernet(Fernet.generate_key())
                params = json.loads(fernet.decrypt(f.read()).decode())
            else:
                params = json.load(f)
            return cls.load_keys_from_memory(params)

    @classmethod
    def load_keys_from_memory(cls, params: Dict[str, Any]) -> 'FHEProvider':
        provider = cls(
            min_range=params.get("min_range", -100000000),
            max_range=params.get("max_range", 100000000),
            scaling_factor=params.get("base_scaling_factor", 10000000),
            adaptive_scaling=params.get("adaptive_scaling", False),
            max_workers=params.get("max_workers", None)
        )
        provider.hmac_key = bytes.fromhex(params.get('hmac_key', os.urandom(32).hex()))
        return provider