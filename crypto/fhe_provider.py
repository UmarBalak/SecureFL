try:
    import concrete.numpy as cnp
    from concrete.numpy import EncryptionStatus
    import numpy as np
    import os
    import json
except ImportError:
    raise ImportError("Install concrete-numpy for FHE support.")

import tensorflow as tf
from base import BaseCryptoProvider

class FHEProvider(BaseCryptoProvider):
    def __init__(self):
        """Initialize FHE provider"""
        self.scaling_factor = 1000  # Scale floating point numbers
        
        # Define a simple function that can work with encrypted inputs
        def fhe_process(x):
            # Identity function that works with encrypted integers
            return x
            
        # Create a compiler with the function
        self.compiler = cnp.Compiler(fhe_process, {"x": EncryptionStatus.ENCRYPTED})
        
        # Compile the circuit with a representative inputset - INTEGERS ONLY
        # Use a range that covers scaled floating point values
        self.inputset = [np.array([i]) for i in range(-2000, 2001)]
        
        # Compile a new circuit
        print("Compiling FHE circuit...")
        self.circuit = self.compiler.compile(self.inputset)
        print("FHE circuit compilation complete.")
            
    
    def encrypt(self, value: float):
        """Encrypt a single float value and run the circuit to get an encrypted result."""
        scaled_value = int(value * self.scaling_factor)
        encrypted_input = self.circuit.encrypt(np.array([scaled_value]))
        # Run the circuit on the encrypted input to get encrypted result
        encrypted_result = self.circuit.run(encrypted_input)
        return encrypted_result


    def decrypt(self, encrypted):
        """Decrypt a ciphertext object back to float."""
        # Decrypt and convert back to float
        result = self.circuit.decrypt(encrypted)
        scaled_value = int(result[0])
        return scaled_value / self.scaling_factor

    def encrypt_tensor(self, tensor):
        """Encrypt a tensor to a list of ciphertext objects."""
        numpy_array = tensor.numpy() if isinstance(tensor, tf.Tensor) else np.array(tensor)
        flat_array = numpy_array.flatten()
        # Use circuit.run() for each element to produce compatible encrypted results
        encrypted_array = [self.encrypt(float(x)) for x in flat_array]
        return encrypted_array, numpy_array.shape

    def decrypt_tensor(self, encrypted_weights, shape):
        """Decrypt a list of ciphertext objects back to a tensor."""
        # Decrypt each element in the encrypted array
        decrypted_flat_array = [self.decrypt(x) for x in encrypted_weights]
        decrypted_tensor = np.array(decrypted_flat_array).reshape(shape)
        return tf.convert_to_tensor(decrypted_tensor, dtype=tf.float32)
        
    def save_keys(self, path: str):
        """Save encryption parameters to disk"""
        # Since we can't pickle the entire circuit, we'll save
        # just the parameters needed to recreate it
        params = {
            "scaling_factor": self.scaling_factor,
            # Any other parameters you might need
        }
        with open(path, "w") as f:
            json.dump(params, f)
        print(f"FHE parameters saved to {path}")
    
    def save_keys_to_memory(self):
        """Save encryption parameters to memory."""
        return {
            "scaling_factor": self.scaling_factor,
            # Any other parameters you might need
        }

    @classmethod
    def generate(cls):
        """Generate a new FHE provider with keys"""
        return cls()
        
    @classmethod
    def load_keys(cls, path: str):
        """Load an FHE provider with existing parameters"""
        provider = cls()  # Create a new provider with a fresh circuit
        
        # Load saved parameters if available
        if os.path.exists(path):
            with open(path, "r") as f:
                params = json.load(f)
                provider.scaling_factor = params.get("scaling_factor", 1000)
                # Load any other parameters you saved
                
        return provider

    @classmethod
    def load_keys_from_memory(cls, params):
        """Load an FHE provider with existing parameters from memory."""
        provider = cls()  # Create a new provider with a fresh circuit
        provider.scaling_factor = params.get("scaling_factor", 1000)
        # Load any other parameters you saved
        return provider