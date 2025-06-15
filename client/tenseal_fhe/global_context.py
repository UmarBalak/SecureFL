import tenseal as ts
import hashlib
import os

def compute_checksum(file_path):
    """Compute SHA256 checksum of a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Prevent overwrite if files exist
full_context_file = "full_context_secure.bin"
public_context_file = "public_context.bin"

if os.path.exists(full_context_file):
    print(f"Warning: {full_context_file} already exists. Skipping generation.")
else:
    # Generate full context
    full_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    full_context.generate_galois_keys()
    full_context.global_scale = 2 ** 40

    # Verify secret key
    if not full_context.has_secret_key():
        raise ValueError("Full context does not contain secret key before saving")

    # Test encryption/decryption to confirm context is functional
    test_vector = ts.ckks_vector(full_context, [1.0, 2.0, 3.0])
    decrypted = test_vector.decrypt()
    if not all(abs(decrypted[i] - [1.0, 2.0, 3.0][i]) < 1e-3 for i in range(3)):
        raise ValueError("Test decryption failed; context may be invalid")

    # Save full context
    with open(full_context_file, "wb") as f:
        serialized_full_context = full_context.serialize(save_secret_key=True)
        f.write(serialized_full_context)
    print(f"Full context saved to {full_context_file}")
    print(f"Full context checksum (SHA256): {compute_checksum(full_context_file)}")

if os.path.exists(public_context_file):
    print(f"Warning: {public_context_file} already exists. Skipping generation.")
else:
    # Load full context to create public context
    with open(full_context_file, "rb") as f:
        full_context = ts.context_from(f.read())
    
    # Create and save public context
    public_context = full_context.copy()
    public_context.make_context_public()
    with open(public_context_file, "wb") as f:
        serialized_public_context = public_context.serialize()
        f.write(serialized_public_context)
    print(f"Public context saved to {public_context_file}")
    print(f"Public context checksum (SHA256): {compute_checksum(public_context_file)}")