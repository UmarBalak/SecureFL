import dp_accounting

def calibrate_noise_multiplier_fast(target_epsilon, delta, num_samples, batch_size, epochs):
    """
    Fast noise_multiplier calibration using RDP accounting
    Optimized by composing events in bulk
    """
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    sampling_prob = batch_size / num_samples
    
    # Pre-compute RDP orders once
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    # Binary search
    low, high = 0.1, 100.0
    best_noise_mult = None
    best_epsilon = None
    
    for i in range(50):
        noise_mult = (low + high) / 2.0
        
        # Create accountant
        accountant = dp_accounting.rdp.RdpAccountant(orders)
        
        # Create event once
        event = dp_accounting.PoissonSampledDpEvent(
            sampling_prob,
            dp_accounting.GaussianDpEvent(noise_mult)
        )
        
        # Compose ALL steps at once (key optimization!)
        accountant.compose(event, total_steps)
        
        achieved_epsilon = accountant.get_epsilon(target_delta=delta)
        
        # Check convergence
        if abs(achieved_epsilon - target_epsilon) < 0.01:
            return noise_mult, achieved_epsilon
        
        # Adjust search
        if achieved_epsilon > target_epsilon:
            low = noise_mult  # Need MORE noise (higher multiplier = more noise = lower epsilon)
        else:
            high = noise_mult  # Need LESS noise
        
        best_noise_mult = noise_mult
        best_epsilon = achieved_epsilon
    
    return best_noise_mult, best_epsilon

# Run calibration
epsilon_values = [1, 2, 3, 5, 10, 15, 20]
for eps in epsilon_values:
    noise_mult, achieved_epsilon = calibrate_noise_multiplier_fast(
      target_epsilon=eps, 
      delta=1e-5, 
      num_samples=100986, 
      batch_size=1024, 
      epochs=20
    )

    print(f"\nFinal result: noise_multiplier={noise_mult:.4f}, epsilon={achieved_epsilon:.4f}")


## Results:
# Final result: noise_multiplier=2.0024, epsilon=0.9913

# Final result: noise_multiplier=1.2219, epsilon=1.9935

# Final result: noise_multiplier=0.9811, epsilon=2.9988

# Final result: noise_multiplier=0.7936, epsilon=4.9929

# Final result: noise_multiplier=0.6255, epsilon=9.9999

# Final result: noise_multiplier=0.5506, epsilon=15.0047

# Final result: noise_multiplier=0.5049, epsilon=20.0098