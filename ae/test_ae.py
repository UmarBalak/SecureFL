import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- CONFIG ----------------
WEIGHT_DIR = "ae_data/weights"
AE_MODEL_PATH = "ae_data/models/autoencoder.h5"
SAMPLE_COUNT = 10   # Number of test files to evaluate
VISUALIZE = True    # Set to False to skip visualization
SAVE_RESULTS = True # Save results to file

# ---------------- LOAD MODELS & DATA ----------------
try:
    autoencoder = load_model(AE_MODEL_PATH)
    print(f"Loaded autoencoder model from {AE_MODEL_PATH}")
    print(f"Model input shape: {autoencoder.input_shape}")
    print(f"Model output shape: {autoencoder.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    weight_files = sorted([os.path.join(WEIGHT_DIR, f) for f in os.listdir(WEIGHT_DIR) if f.endswith(".npy")])
    
    if len(weight_files) < SAMPLE_COUNT:
        print(f"Warning: Requested {SAMPLE_COUNT} samples but only found {len(weight_files)}.")
        SAMPLE_COUNT = len(weight_files)
    
    # Randomly select samples if there are many files
    if len(weight_files) > SAMPLE_COUNT:
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(weight_files), SAMPLE_COUNT, replace=False)
        test_files = [weight_files[i] for i in sample_indices]
    else:
        test_files = weight_files[:SAMPLE_COUNT]
        
    print(f"Testing Autoencoder on {SAMPLE_COUNT} weight files...\n")
except Exception as e:
    print(f"Error loading weight files: {e}")
    exit(1)

# ---------------- EVALUATION ----------------
results = []
total_mse = 0
total_cosine = 0
all_diffs = []

for i, file_path in enumerate(test_files):
    try:
        original = np.load(file_path)
        
        # Check for NaN or Inf values
        if np.isnan(original).any() or np.isinf(original).any():
            print(f"Warning: File {os.path.basename(file_path)} contains NaN or Inf values, skipping.")
            continue
            
        # Handle input shape issues - ensure it matches model's expected input
        if original.ndim == 1:
            input_data = original[np.newaxis, :]
        else:
            input_data = original.reshape(1, *original.shape)
            
        reconstructed = autoencoder.predict(input_data, verbose=0)[0]
        
        # Reshape back if needed
        if original.shape != reconstructed.shape:
            reconstructed = reconstructed.reshape(original.shape)

        # Calculate metrics
        mse = mean_squared_error(original, reconstructed)
        cos_sim = cosine_similarity([original.flatten()], [reconstructed.flatten()])[0][0]
        
        # Calculate absolute differences
        diff = np.abs(original - reconstructed)
        max_diff = np.max(diff)
        max_idx = np.argmax(diff)
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        all_diffs.append(diff)
        
        # Store results
        results.append({
            'file': os.path.basename(file_path),
            'mse': mse,
            'cosine': cos_sim,
            'max_diff': max_diff,
            'max_idx': max_idx,
            'mean_diff': mean_diff,
            'median_diff': median_diff
        })
        
        total_mse += mse
        total_cosine += cos_sim

        print(f"[Sample {i+1}] {os.path.basename(file_path)}")
        print(f"  Mean Squared Error     : {mse:.6f}")
        print(f"  Cosine Similarity      : {cos_sim:.6f}")
        print(f"  Max weight difference  : {max_diff:.6f} at index {max_idx}")
        print(f"  Mean difference        : {mean_diff:.6f}")
        print(f"  Median difference      : {median_diff:.6f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        continue

# ---------------- SUMMARY ----------------
if results:
    avg_mse = total_mse / len(results)
    avg_cos = total_cosine / len(results)
    
    print(f"\n✅ Autoencoder Evaluation Summary:")
    print(f"Samples Processed       : {len(results)}/{SAMPLE_COUNT}")
    print(f"Average MSE             : {avg_mse:.6f}")
    print(f"Average Cosine Similarity: {avg_cos:.6f}")
    
    # Calculate global diff statistics
    if all_diffs:
        all_diffs_array = np.concatenate([d.flatten() for d in all_diffs])
        print(f"Global diff statistics:")
        print(f"  Min diff              : {np.min(all_diffs_array):.6f}")
        print(f"  Max diff              : {np.max(all_diffs_array):.6f}")
        print(f"  Mean diff             : {np.mean(all_diffs_array):.6f}")
        print(f"  Median diff           : {np.median(all_diffs_array):.6f}")
        print(f"  Std diff              : {np.std(all_diffs_array):.6f}")
        
        # Get percentiles
        p95 = np.percentile(all_diffs_array, 95)
        p99 = np.percentile(all_diffs_array, 99)
        print(f"  95th percentile diff  : {p95:.6f}")
        print(f"  99th percentile diff  : {p99:.6f}")
else:
    print("No valid results to report.")


# ---------------- COMPRESSION RATIO CALCULATION ----------------
print("\n===== COMPRESSION RATIO ANALYSIS =====")

# Get the encoder part of your autoencoder (if it exists)
try:
    encoder = None
    if hasattr(autoencoder, 'get_layer'):
        try:
            encoder = autoencoder.get_layer('encoder')
            print("Found named encoder layer in model")
        except:
            # Try creating a model from input to bottleneck if named layer doesn't exist
            print("No named encoder layer found, identifying bottleneck...")
    
    # If we couldn't find a named encoder, try to find bottleneck layer
    if encoder is None:
        # Find bottleneck layer by looking for smallest representation size
        input_size = np.prod(autoencoder.input_shape[1:])  # Total elements in input
        bottleneck_layer = None
        bottleneck_size = float('inf')
        
        for i, layer in enumerate(autoencoder.layers):
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, tuple) and len(shape) >= 2:
                    size = np.prod(shape[1:])  # Total elements in output
                    if size < bottleneck_size and i > 0:  # Skip input layer
                        bottleneck_size = size
                        bottleneck_layer = layer
        
        if bottleneck_layer:
            # Create encoder model from input to bottleneck
            from tensorflow.keras import Model
            encoder = Model(inputs=autoencoder.input, outputs=bottleneck_layer.output)
            print(f"Created encoder model with bottleneck size: {bottleneck_size} elements")
            
    # Calculate compression ratio using the first few samples
    if encoder:
        compression_ratios = []
        sample_count = min(5, len(test_files))  # Use up to 5 samples
        
        for i in range(sample_count):
            try:
                # Load original weights
                original = np.load(test_files[i])
                
                # Get original size
                original_size = original.size * original.itemsize  # Elements * bytes per element
                
                # Get encoded representation
                if original.ndim == 1:
                    input_data = original[np.newaxis, :]
                else:
                    input_data = original.reshape(1, *original.shape)
                    
                encoded = encoder.predict(input_data, verbose=0)[0]
                encoded_size = encoded.size * encoded.itemsize
                
                # Calculate compression ratio
                ratio = original_size / encoded_size
                compression_ratios.append(ratio)
                
                print(f"Sample {i+1}:")
                print(f"  Original size:      {original_size} bytes ({original.shape})")
                print(f"  Encoded size:       {encoded_size} bytes ({encoded.shape})")
                print(f"  Compression ratio:  {ratio:.2f}x")
                print(f"  Space reduction:    {(1 - encoded_size/original_size) * 100:.2f}%")
            except Exception as e:
                print(f"Error calculating compression for sample {i+1}: {e}")
        
        if compression_ratios:
            avg_ratio = sum(compression_ratios) / len(compression_ratios)
            print(f"\nAverage compression ratio: {avg_ratio:.2f}x")
            print(f"Average space reduction: {(1 - 1/avg_ratio) * 100:.2f}%")
            
            # Add to visualization if it exists
            if VISUALIZE:
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(compression_ratios)), compression_ratios)
                plt.axhline(y=avg_ratio, color='r', linestyle='-', label=f'Average: {avg_ratio:.2f}x')
                plt.xlabel('Sample Index')
                plt.ylabel('Compression Ratio')
                plt.title('Autoencoder Compression Ratio by Sample')
                plt.xticks(range(len(compression_ratios)), [f"{i+1}" for i in range(len(compression_ratios))])
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if SAVE_RESULTS:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plt.savefig(f"compression_ratio_{timestamp}.png")
                    print(f"Saved compression ratio visualization to compression_ratio_{timestamp}.png")
                else:
                    plt.show()
    else:
        # If we couldn't find/create an encoder, try analyzing model architecture
        print("Could not identify encoder. Analyzing model architecture...")
        
        # Get input size
        input_size = np.prod(autoencoder.input_shape[1:])
        
        # Find smallest layer size (bottleneck)
        min_size = input_size
        for layer in autoencoder.layers:
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, tuple) and len(shape) >= 2:
                    size = np.prod(shape[1:])
                    min_size = min(min_size, size)
        
        # Calculate theoretical compression ratio
        if min_size < input_size:
            ratio = input_size / min_size
            print(f"Input size:               {input_size} elements")
            print(f"Estimated bottleneck size: {min_size} elements")
            print(f"Theoretical compression ratio: {ratio:.2f}x")
            print(f"Theoretical space reduction: {(1 - min_size/input_size) * 100:.2f}%")
        else:
            print("Could not identify a bottleneck layer smaller than input.")
            
except Exception as e:
    print(f"Error calculating compression ratio: {e}")

# ---------------- VISUALIZATION ----------------
if VISUALIZE and results:
    plt.figure(figsize=(15, 10))
    
    # Plot MSE for each sample
    plt.subplot(2, 2, 1)
    plt.bar(range(len(results)), [r['mse'] for r in results])
    plt.xlabel('Sample Index')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error per Sample')
    plt.xticks(range(len(results)), [f"{i+1}" for i in range(len(results))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot Cosine Similarity for each sample
    plt.subplot(2, 2, 2)
    plt.bar(range(len(results)), [r['cosine'] for r in results])
    plt.xlabel('Sample Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity per Sample')
    plt.xticks(range(len(results)), [f"{i+1}" for i in range(len(results))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution histogram
    plt.subplot(2, 2, 3)
    all_diffs_array = np.concatenate([d.flatten() for d in all_diffs])
    plt.hist(all_diffs_array, bins=50, alpha=0.75)
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True, alpha=0.3)
    
    # Plot CDF of errors
    plt.subplot(2, 2, 4)
    values, base = np.histogram(all_diffs_array, bins=100)
    cumulative = np.cumsum(values) / len(all_diffs_array)
    plt.plot(base[:-1], cumulative, 'b-')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Absolute Difference')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Reconstruction Errors')
    
    plt.tight_layout()
    
    # Save or show the plot
    if SAVE_RESULTS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"autoencoder_eval_{timestamp}.png")
        print(f"\nSaved visualization to autoencoder_eval_{timestamp}.png")
    else:
        plt.show()

# Save detailed results to CSV
if SAVE_RESULTS and results:
    import csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"autoencoder_eval_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['file', 'mse', 'cosine', 'max_diff', 'max_idx', 'mean_diff', 'median_diff']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        
    print(f"Saved detailed results to {csv_filename}")
