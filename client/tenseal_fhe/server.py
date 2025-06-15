import tenseal as ts
import os
import struct

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

def server_average(client_ids, num_layers=10):
    """Load encrypted weights from multiple clients, average them, and save result."""
    # Load public context
    context_file = "public_context.bin"
    if not os.path.exists(context_file):
        raise FileNotFoundError(f"Public context file not found: {context_file}")
    
    with open(context_file, "rb") as f:
        serialized_public_context = f.read()
    server_context = ts.context_from(serialized_public_context)

    # Load encrypted weights from all clients
    total_weights = []
    for client_id in client_ids:
        weight_file = f"client{client_id}_enc_weights.bin"
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Encrypted weights file not found: {weight_file}")
        encrypted_weights = load_encrypted_weights(server_context, weight_file, num_layers)
        total_weights.append(encrypted_weights)

    # Aggregate encrypted weights
    n_models = len(total_weights)
    encrypted_avg_weights = []

    for layer in range(num_layers):
        enc_sum = total_weights[0][layer]
        for i in range(1, n_models):
            enc_sum += total_weights[i][layer]
        enc_avg = enc_sum * (1.0 / n_models)
        encrypted_avg_weights.append(enc_avg)

    # Save averaged encrypted weights using the consistent format
    output_file = "avg_enc_weights.bin"
    save_encrypted_weights(encrypted_avg_weights, output_file)
    print(f"Averaged encrypted weights saved to {output_file}")

if __name__ == "__main__":
    server_average(client_ids=[1, 2], num_layers=10)