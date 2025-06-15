import tenseal as ts
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.regularizers import l2
import numpy as np
import time
import os
import struct
import json
import hashlib
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class FHEBenchmark:
    def __init__(self, input_dim=47, num_classes=15, architecture=[128, 64]):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.architecture = architecture
        self.metrics = {
            'encryption_metrics': {},
            'aggregation_metrics': {},
            'decryption_metrics': {},
            'size_metrics': {},
            'accuracy_metrics': {},
            'overhead_metrics': {},
            'system_info': {},
            'flow_description': {}
        }
        
    def create_mlp_model(self):
        """Create the MLP model used in the federated learning setup."""
        model = Sequential()
        model.add(Dense(self.architecture[0], input_dim=self.input_dim, 
                       activation='relu', kernel_regularizer=l2(0.0005)))
        model.add(LayerNormalization())

        for units in self.architecture[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.0005)))
            model.add(LayerNormalization())

        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy']
        )
        return model
    
    def setup_contexts(self):
        """Setup FHE contexts and measure initialization time."""
        print("Setting up FHE contexts...")
        start_time = time.time()
        
        # Generate full context
        full_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        full_context.generate_galois_keys()
        full_context.global_scale = 2 ** 40
        
        context_setup_time = time.time() - start_time
        
        # Create public context
        public_context = full_context.copy()
        public_context.make_context_public()
        
        # Save contexts
        with open("full_context_secure.bin", "wb") as f:
            f.write(full_context.serialize(save_secret_key=True))
        
        with open("public_context.bin", "wb") as f:
            f.write(public_context.serialize())
        
        self.metrics['system_info']['context_setup_time'] = context_setup_time
        self.metrics['system_info']['poly_modulus_degree'] = 32768
        self.metrics['system_info']['coeff_mod_bit_sizes'] = [60, 40, 40, 60]
        self.metrics['system_info']['scheme'] = 'CKKS'
        
        return full_context, public_context
    
    def get_file_size(self, filename):
        """Get file size in bytes."""
        if os.path.exists(filename):
            return os.path.getsize(filename)
        return 0
    
    def benchmark_client_encryption(self, client_id, num_clients=2):
        """Benchmark the client-side encryption process."""
        print(f"Benchmarking client {client_id} encryption...")
        
        # Load public context
        with open("public_context.bin", "rb") as f:
            public_context = ts.context_from(f.read())
        
        # Create model and get weights
        model = self.create_mlp_model()
        weights = model.get_weights()
        
        # Calculate plaintext weight statistics
        total_params = sum(w.size for w in weights)
        plaintext_size = sum(w.nbytes for w in weights)
        
        # Measure encryption time
        encryption_start = time.time()
        encrypted_weights = []
        
        for layer_idx, layer_weights in enumerate(weights):
            layer_start = time.time()
            flat_array = layer_weights.flatten()
            enc_vector = ts.ckks_vector(public_context, flat_array)
            encrypted_weights.append(enc_vector)
            layer_time = time.time() - layer_start
            
            print(f"  Layer {layer_idx}: {layer_weights.shape} -> {layer_time:.4f}s")
        
        total_encryption_time = time.time() - encryption_start
        
        # Save encrypted weights and measure file size
        enc_filename = f"client{client_id}_enc_weights.bin"
        self.save_encrypted_weights(encrypted_weights, enc_filename)
        encrypted_size = self.get_file_size(enc_filename)
        
        # Save plaintext weights for comparison
        plain_filename = f"client{client_id}_plain_weights.npy"
        np.save(plain_filename, np.array(weights, dtype=object), allow_pickle=True)
        plaintext_file_size = self.get_file_size(plain_filename)
        
        # Store metrics
        client_metrics = {
            'encryption_time': total_encryption_time,
            'plaintext_size_bytes': plaintext_size,
            'plaintext_file_size_bytes': plaintext_file_size,
            'encrypted_file_size_bytes': encrypted_size,
            'compression_ratio': encrypted_size / plaintext_file_size if plaintext_file_size > 0 else 0,
            'total_parameters': total_params,
            'num_layers': len(weights),
            'layer_shapes': [w.shape for w in weights],
            'encryption_rate_params_per_sec': total_params / total_encryption_time if total_encryption_time > 0 else 0
        }
        
        self.metrics['encryption_metrics'][f'client_{client_id}'] = client_metrics
        
        return weights, encrypted_weights
    
    def benchmark_server_aggregation(self, client_ids, num_layers=10):
        """Benchmark the server-side homomorphic aggregation."""
        print("Benchmarking server aggregation...")
        
        # Load public context
        with open("public_context.bin", "rb") as f:
            server_context = ts.context_from(f.read())
        
        aggregation_start = time.time()
        
        # Load encrypted weights from all clients
        total_weights = []
        total_encrypted_size = 0
        
        for client_id in client_ids:
            weight_file = f"client{client_id}_enc_weights.bin"
            encrypted_weights = self.load_encrypted_weights(server_context, weight_file, num_layers)
            total_weights.append(encrypted_weights)
            total_encrypted_size += self.get_file_size(weight_file)
        
        load_time = time.time() - aggregation_start
        
        # Perform homomorphic averaging
        averaging_start = time.time()
        n_models = len(total_weights)
        encrypted_avg_weights = []
        
        for layer in range(num_layers):
            layer_start = time.time()
            enc_sum = total_weights[0][layer]
            
            # Sum all client weights for this layer
            for i in range(1, n_models):
                enc_sum += total_weights[i][layer]
            
            # Divide by number of models
            enc_avg = enc_sum * (1.0 / n_models)
            encrypted_avg_weights.append(enc_avg)
            
            layer_avg_time = time.time() - layer_start
            print(f"  Layer {layer} averaging: {layer_avg_time:.4f}s")
        
        averaging_time = time.time() - averaging_start
        total_aggregation_time = time.time() - aggregation_start
        
        # Save averaged encrypted weights
        avg_filename = "avg_enc_weights.bin"
        self.save_encrypted_weights(encrypted_avg_weights, avg_filename)
        avg_encrypted_size = self.get_file_size(avg_filename)
        
        # Store metrics
        self.metrics['aggregation_metrics'] = {
            'total_aggregation_time': total_aggregation_time,
            'weight_loading_time': load_time,
            'homomorphic_averaging_time': averaging_time,
            'num_clients': len(client_ids),
            'total_input_size_bytes': total_encrypted_size,
            'output_size_bytes': avg_encrypted_size,
            'aggregation_rate_layers_per_sec': num_layers / averaging_time if averaging_time > 0 else 0
        }
        
        return encrypted_avg_weights
    
    def benchmark_client_decryption(self, client_ids, num_layers=10):
        """Benchmark the client-side decryption and verification."""
        print("Benchmarking client decryption...")
        
        # Load full context
        with open("full_context_secure.bin", "rb") as f:
            full_context = ts.context_from(f.read())
        
        # Load averaged encrypted weights
        decryption_start = time.time()
        encrypted_avg_weights = self.load_encrypted_weights(full_context, "avg_enc_weights.bin", num_layers)
        load_time = time.time() - decryption_start
        
        # Create model to get shapes
        model = self.create_mlp_model()
        shapes = [w.shape for w in model.get_weights()]
        
        # Decrypt weights
        decrypt_start = time.time()
        avg_decrypted_weights = []
        
        for i, (enc_vec, shape) in enumerate(zip(encrypted_avg_weights, shapes)):
            layer_start = time.time()
            decrypted_flat = enc_vec.decrypt()
            decrypted_np = np.array(decrypted_flat).reshape(shape)
            avg_decrypted_weights.append(decrypted_np)
            layer_time = time.time() - layer_start
            print(f"  Layer {i} decryption: {shape} -> {layer_time:.4f}s")
        
        decryption_time = time.time() - decrypt_start
        total_decrypt_time = time.time() - decryption_start
        
        # Load original plaintext weights for verification
        client_weights = []
        for client_id in client_ids:
            weights = np.load(f"client{client_id}_plain_weights.npy", allow_pickle=True).tolist()
            client_weights.append(weights)
        
        # Compute expected plaintext average
        plaintext_avg_start = time.time()
        expected_avg_weights = client_weights[0]
        for i in range(1, len(client_weights)):
            expected_avg_weights = [w1 + w2 for w1, w2 in zip(expected_avg_weights, client_weights[i])]
        expected_avg_weights = [w / len(client_weights) for w in expected_avg_weights]
        plaintext_avg_time = time.time() - plaintext_avg_start
        
        # Verify accuracy
        verification_results = []
        total_mae = 0
        total_rmse = 0
        total_cos_sim = 0
        
        for i, (actual, expected) in enumerate(zip(avg_decrypted_weights, expected_avg_weights)):
            abs_diff = np.abs(actual - expected)
            mae = np.mean(abs_diff)
            rmse = np.sqrt(np.mean(abs_diff ** 2))
            cos_sim = cosine_similarity([actual.flatten()], [expected.flatten()])[0][0]
            max_error = np.max(abs_diff)
            
            layer_result = {
                "layer": i,
                "shape": actual.shape,
                "mae": float(mae),
                "rmse": float(rmse),
                "cosine_similarity": float(cos_sim),
                "max_absolute_error": float(max_error)
            }
            verification_results.append(layer_result)
            
            total_mae += mae
            total_rmse += rmse
            total_cos_sim += cos_sim
        
        # Store metrics
        self.metrics['decryption_metrics'] = {
            'total_decryption_time': total_decrypt_time,
            'weight_loading_time': load_time,
            'pure_decryption_time': decryption_time,
            'num_layers': num_layers,
            'decryption_rate_layers_per_sec': num_layers / decryption_time if decryption_time > 0 else 0
        }
        
        self.metrics['accuracy_metrics'] = {
            'layer_wise_results': verification_results,
            'average_mae': float(total_mae / len(verification_results)),
            'average_rmse': float(total_rmse / len(verification_results)),
            'average_cosine_similarity': float(total_cos_sim / len(verification_results)),
            'plaintext_averaging_time': plaintext_avg_time,
            'fhe_vs_plaintext_accuracy_loss': float(total_mae / len(verification_results))
        }
        
        return avg_decrypted_weights, expected_avg_weights
    
    def calculate_overhead_metrics(self):
        """Calculate various overhead metrics."""
        print("Calculating overhead metrics...")
        
        # Time overheads
        enc_metrics = self.metrics['encryption_metrics']
        agg_metrics = self.metrics['aggregation_metrics']
        dec_metrics = self.metrics['decryption_metrics']
        acc_metrics = self.metrics['accuracy_metrics']
        
        total_client_enc_time = sum(client['encryption_time'] for client in enc_metrics.values())
        total_fhe_time = total_client_enc_time + agg_metrics['total_aggregation_time'] + dec_metrics['total_decryption_time']
        plaintext_time = acc_metrics['plaintext_averaging_time']
        
        # Size overheads
        total_plaintext_size = sum(client['plaintext_file_size_bytes'] for client in enc_metrics.values())
        total_encrypted_size = sum(client['encrypted_file_size_bytes'] for client in enc_metrics.values())
        
        # Communication overhead (encrypted vs plaintext)
        communication_overhead = total_encrypted_size / total_plaintext_size if total_plaintext_size > 0 else 0
        
        # Computational overhead
        computational_overhead = total_fhe_time / plaintext_time if plaintext_time > 0 else 0
        
        self.metrics['overhead_metrics'] = {
            'total_fhe_pipeline_time': total_fhe_time,
            'plaintext_averaging_time': plaintext_time,
            'computational_overhead_ratio': computational_overhead,
            'communication_overhead_ratio': communication_overhead,
            'time_breakdown': {
                'encryption_percentage': (total_client_enc_time / total_fhe_time) * 100,
                'aggregation_percentage': (agg_metrics['total_aggregation_time'] / total_fhe_time) * 100,
                'decryption_percentage': (dec_metrics['total_decryption_time'] / total_fhe_time) * 100
            },
            'size_breakdown': {
                'total_plaintext_size_mb': total_plaintext_size / (1024 * 1024),
                'total_encrypted_size_mb': total_encrypted_size / (1024 * 1024),
                'size_inflation_factor': communication_overhead
            }
        }
    
    def save_encrypted_weights(self, encrypted_weights, filename):
        """Save encrypted weights with size prefixes."""
        with open(filename, "wb") as f:
            f.write(struct.pack("i", len(encrypted_weights)))
            for enc_vector in encrypted_weights:
                serialized_data = enc_vector.serialize()
                f.write(struct.pack("i", len(serialized_data)))
                f.write(serialized_data)
    
    def load_encrypted_weights(self, context, filename, num_layers):
        """Load encrypted weights from file."""
        with open(filename, "rb") as f:
            num_layers_file = struct.unpack("i", f.read(4))[0]
            if num_layers_file != num_layers:
                raise ValueError(f"Expected {num_layers} layers, found {num_layers_file}")
            encrypted_weights = []
            for _ in range(num_layers):
                size = struct.unpack("i", f.read(4))[0]
                serialized_data = f.read(size)
                enc_vector = ts.ckks_vector_from(context, serialized_data)
                encrypted_weights.append(enc_vector)
            return encrypted_weights
    
    def run_full_benchmark(self, client_ids=[1, 2], num_layers=10):
        """Run the complete benchmark suite."""
        print("=" * 60)
        print("FHE Federated Learning Benchmark Suite")
        print("=" * 60)
        
        # Setup
        self.setup_contexts()
        
        # Benchmark each phase
        client_weights = {}
        for client_id in client_ids:
            weights, enc_weights = self.benchmark_client_encryption(client_id, len(client_ids))
            client_weights[client_id] = weights
        
        encrypted_avg = self.benchmark_server_aggregation(client_ids, num_layers)
        fhe_avg, plain_avg = self.benchmark_client_decryption(client_ids, num_layers)
        
        # Calculate overheads
        self.calculate_overhead_metrics()
        
        # Describe workflow
        self.describe_workflow()
        
        # Add timestamp and system info
        self.metrics['benchmark_info'] = {
            'timestamp': datetime.now().isoformat(),
            'client_ids': client_ids,
            'num_layers': num_layers,
            'model_architecture': {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'hidden_layers': self.architecture
            }
        }
        
        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        
        return self.metrics
    
    def save_metrics(self, filename="fhe_benchmark_metrics.json"):
        """Save all metrics to JSON file for analysis."""
        # Convert any numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        converted_metrics = convert_types(self.metrics)
        
        with open(filename, 'w') as f:
            json.dump(converted_metrics, f, indent=2)
        
        print(f"Metrics saved to {filename}")
        
        # Also save as pickle for easier Python loading
        pickle_filename = filename.replace('.json', '.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(converted_metrics, f)
        
        print(f"Metrics also saved to {pickle_filename}")
        
        return converted_metrics
    
    def print_summary(self):
        """Print a summary of key metrics."""
        if not self.metrics:
            print("No metrics available. Run benchmark first.")
            return
        
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Timing Summary
        enc_time = sum(client['encryption_time'] for client in self.metrics['encryption_metrics'].values())
        agg_time = self.metrics['aggregation_metrics']['total_aggregation_time']
        dec_time = self.metrics['decryption_metrics']['total_decryption_time']
        total_time = enc_time + agg_time + dec_time
        
        print(f"\n🕐 TIMING ANALYSIS:")
        print(f"   Total FHE Pipeline Time: {total_time:.4f} seconds")
        print(f"   ├─ Encryption Time:      {enc_time:.4f}s ({enc_time/total_time*100:.1f}%)")
        print(f"   ├─ Aggregation Time:     {agg_time:.4f}s ({agg_time/total_time*100:.1f}%)")
        print(f"   └─ Decryption Time:      {dec_time:.4f}s ({dec_time/total_time*100:.1f}%)")
        
        plain_time = self.metrics['accuracy_metrics']['plaintext_averaging_time']
        print(f"   Plaintext Averaging:     {plain_time:.4f}s")
        print(f"   Computational Overhead:  {total_time/plain_time:.1f}x slower")
        
        # Size Analysis  
        overhead = self.metrics['overhead_metrics']
        print(f"\n📊 SIZE ANALYSIS:")
        print(f"   Plaintext Size:  {overhead['size_breakdown']['total_plaintext_size_mb']:.2f} MB")
        print(f"   Encrypted Size:  {overhead['size_breakdown']['total_encrypted_size_mb']:.2f} MB") 
        print(f"   Size Inflation:  {overhead['size_breakdown']['size_inflation_factor']:.1f}x larger")
        
        # Accuracy Analysis
        acc = self.metrics['accuracy_metrics']
        print(f"\n🎯 ACCURACY ANALYSIS:")
        print(f"   Average MAE:           {acc['average_mae']:.2e}")
        print(f"   Average RMSE:          {acc['average_rmse']:.2e}")
        print(f"   Average Cosine Sim:    {acc['average_cosine_similarity']:.6f}")
        print(f"   FHE Accuracy Loss:     {acc['fhe_vs_plaintext_accuracy_loss']:.2e}")
        
        print(f"\n🔐 SECURITY PARAMETERS:")
        print(f"   Scheme: CKKS")
        print(f"   Poly Modulus Degree: {self.metrics['system_info']['poly_modulus_degree']}")
        
        print("=" * 50)


if __name__ == "__main__":
    # Run the comprehensive benchmark
    benchmark = FHEBenchmark(input_dim=47, num_classes=15, architecture=[64, 32])
    
    # Clean up any existing files to ensure fresh run
    files_to_clean = [
        "full_context_secure.bin", "public_context.bin", "avg_enc_weights.bin",
        "client1_enc_weights.bin", "client2_enc_weights.bin",
        "client1_plain_weights.npy", "client2_plain_weights.npy"
    ]
    
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
    
    # Run full benchmark
    metrics = benchmark.run_full_benchmark(client_ids=[1, 2, 3, 4], num_layers=10)
    
    # Save results
    benchmark.save_metrics("fhe_federated_learning_metrics_4.json")
    
    # Print summary
    benchmark.print_summary()
    
    print(f"\n✅ Complete metrics saved to 'fhe_federated_learning_metrics.json'")
    print(f"✅ Use this file with Streamlit, Plotly, or other visualization tools")