import tenseal as ts
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import keras
import os
import struct
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2



def create_mlp_model(input_dim, num_classes, architecture=[128, 64]):
    model = Sequential()
    model.add(Dense(architecture[0], input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(LayerNormalization())

    for units in architecture[1:]:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.0005)))
        model.add(LayerNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    return model


import tenseal as ts
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import keras
import os
import struct
import hashlib

def load_encrypted_weights(context, filename, num_layers):
    """Load encrypted weights from a single file."""
    with open(filename, "rb") as f:
        num_layers_file = struct.unpack("i", f.read(4))[0]
        if num_layers_file != num_layers:
            raise ValueError(f"Expected {num_layers} layers, found {num_layers_file}")
        encrypted_weights = []
        for _ in range(num_layers):
            # Read the size of this serialized vector
            size = struct.unpack("i", f.read(4))[0]
            # Read exactly that many bytes
            serialized_data = f.read(size)
            if len(serialized_data) != size:
                raise ValueError(f"Expected {size} bytes, got {len(serialized_data)}")
            enc_vector = ts.ckks_vector_from(context, serialized_data)
            encrypted_weights.append(enc_vector)
        return encrypted_weights

def load_plaintext_weights(filename):
    """Load plaintext weights from a .npy file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Plaintext weights file not found: {filename}")
    return np.load(filename, allow_pickle=True).tolist()

def client_decrypt(input_dim=47, num_classes=15, architecture=[128, 64], num_layers=10, client_ids=[1, 2]):
    """Decrypt averaged weights, set to model, and verify using original weights."""
    # Load full context
    context_file = "full_context_secure.bin"
    if not os.path.exists(context_file):
        raise FileNotFoundError(f"Full context file not found: {context_file}")
    
    # Compute checksum for debugging
    with open(context_file, "rb") as f:
        context_data = f.read()
        checksum = hashlib.sha256(context_data).hexdigest()
        print(f"Loading {context_file}, checksum (SHA256): {checksum}")
    
    try:
        full_context = ts.context_from(context_data)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize context: {str(e)}")
    
    # Verify secret key
    if not full_context.has_secret_key():
        raise ValueError("Full context does not contain secret key")
    print("Secret key successfully loaded")

    # Test decryption to confirm context
    try:
        test_vector = ts.ckks_vector(full_context, [1.0, 2.0, 3.0])
        decrypted = test_vector.decrypt()
        if not all(abs(decrypted[i] - [1.0, 2.0, 3.0][i]) < 1e-3 for i in range(3)):
            raise ValueError("Test decryption failed; context may be invalid")
        print("Test decryption successful")
    except Exception as e:
        raise RuntimeError(f"Context test failed: {str(e)}")

    secret_key = full_context.secret_key()

    # Create model to get shapes
    model = create_mlp_model(input_dim=input_dim, num_classes=num_classes, architecture=architecture)
    shapes = [w.shape for w in model.get_weights()]

    # Load averaged encrypted weights
    avg_file = "avg_enc_weights.bin"
    if not os.path.exists(avg_file):
        raise FileNotFoundError(f"Averaged encrypted weights file not found: {avg_file}")
    
    encrypted_avg_weights = load_encrypted_weights(full_context, avg_file, num_layers)

    # Decrypt weights
    avg_decrypted_weights = []
    for enc_vec, shape in zip(encrypted_avg_weights, shapes):
        decrypted_flat = enc_vec.decrypt(secret_key)
        decrypted_np = np.array(decrypted_flat).reshape(shape)
        avg_decrypted_weights.append(decrypted_np)

    # Set weights to new model
    avg_model = keras.models.clone_model(model)
    avg_model.set_weights(avg_decrypted_weights)

    # Load original weights for verification
    client_weights = []
    for client_id in client_ids:
        plain_file = f"client{client_id}_plain_weights.npy"
        weights = load_plaintext_weights(plain_file)
        client_weights.append(weights)
    
    # Compute expected average
    expected_avg_weights = client_weights[0]
    for i in range(1, len(client_weights)):
        expected_avg_weights = [w1 + w2 for w1, w2 in zip(expected_avg_weights, client_weights[i])]
    expected_avg_weights = [w / len(client_weights) for w in expected_avg_weights]
    
    # Verify correctness
    results = []
    for i, (actual, expected) in enumerate(zip(avg_decrypted_weights, expected_avg_weights)):
        abs_diff = np.abs(actual - expected)
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean(abs_diff ** 2))
        cos_sim = cosine_similarity([actual.flatten()], [expected.flatten()])[0][0]
        results.append({
            "Layer": i,
            "MAE": mae,
            "RMSE": rmse,
            "CosineSimilarity": cos_sim
        })

    print("Verification results:", results)
    return avg_model

if __name__ == "__main__":
    avg_model = client_decrypt(num_layers=10, client_ids=[1, 2])