from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
import tenseal as ts
import os
import numpy as np
import struct


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
import keras
import os
import struct

def save_encrypted_weights(encrypted_weights, filename):
    """Save all encrypted weights to a single file with size prefixes."""
    with open(filename, "wb") as f:
        f.write(struct.pack("i", len(encrypted_weights)))
        for enc_vector in encrypted_weights:
            serialized_data = enc_vector.serialize()
            # Write the size of this serialized vector first
            f.write(struct.pack("i", len(serialized_data)))
            # Then write the serialized data
            f.write(serialized_data)

def save_plaintext_weights(weights, filename):
    """Save plaintext weights to a single file using NumPy's .npy format."""
    weights_array = np.array(weights, dtype=object)
    np.save(filename, weights_array, allow_pickle=True)


def client_encrypt(client_id, input_dim=47, num_classes=15, architecture=[128, 64]):
    """Encrypt client model weights and save both encrypted and plaintext weights."""
    # Load public context
    context_file = "public_context.bin"
    if not os.path.exists(context_file):
        raise FileNotFoundError(f"Public context file not found: {context_file}")
    
    with open(context_file, "rb") as f:
        serialized_public_context = f.read()
    client_context = ts.context_from(serialized_public_context)

    # Create model and get weights
    model = create_mlp_model(input_dim=input_dim, num_classes=num_classes, architecture=architecture)
    weights = model.get_weights()

    # Encrypt weights
    encrypted_weights = []
    for layer_weights in weights:
        flat_array = layer_weights.flatten()
        enc_vector = ts.ckks_vector(client_context, flat_array)
        encrypted_weights.append(enc_vector)

    # Save encrypted weights
    enc_output_file = f"client{client_id}_enc_weights.bin"
    save_encrypted_weights(encrypted_weights, enc_output_file)
    print(f"Encrypted weights saved to {enc_output_file}")

    # Save plaintext weights
    plain_output_file = f"client{client_id}_plain_weights.npy"
    save_plaintext_weights(weights, plain_output_file)
    print(f"Plaintext weights saved to {plain_output_file}")

if __name__ == "__main__":
    for i in range(1, 3):
        client_encrypt(client_id=i)