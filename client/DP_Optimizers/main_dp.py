import os
import sys
import gc
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# FIXED: Local imports with correct paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing_global import IoTDataPreprocessor
from evaluate import evaluate_model
from functions_dp import wait_for_csv

# Corrected trainer implementing both TRUE DP approaches
from training_dp_advanced import IoTModelTrainer

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
script_directory = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = "DATA/global_train.csv"
TEST_DATASET_PATH = "DATA/global_test.csv"
GLOBAL_MODEL_PATH = os.path.join(script_directory, "GLOBAL_MODELS")

def main(epochs=20, target_epsilon=2.0, target_delta=1e-5, l2_norm_clip=3.0, 
         noise_type='gaussian', batch_size=128, learning_rate=1e-3):
    """
    CORRECTED: Main function with proper DP comparison
    """

    config = {
        'train_data_path_pattern': DATASET_PATH,
        'test_data_path_pattern': TEST_DATASET_PATH,
        'epochs': epochs,
        'random_state': 42,
        'model_architecture': [256, 256],  # Simpler architecture for DP
        'learning_rate': 0.001,  # Standard learning rate
        'num_classes': 15,
        'class_names': [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ],
        'artifacts_dir': "artifacts",
    }
    
    ## Configuration A: Comparable Privacy (ε ≈ 3.0) (only for 20 epochs, if number of epochs or samples changed, gaussian ε will be different.)
    gaussian_config = {
        'l2_norm_clip': 3.0,
        'noise_multiplier': 0.9811, # Will give ε = 2.9988 (approx) after 20 epochs and 100986 samples
        'delta': 1e-5,
        'batch_size': 1024,

        # not to use
        'l1_norm_clip': 3.0,
        'epsilon_total': 500.0,
    }
    t_laplace_config = {
        'l1_norm_clip': 3.0, # L1 is 80× LARGER than L2!
        'epsilon_total': 3.0,
        'batch_size': 1024,

        # not to use
        'noise_multiplier': 1.0,
        'delta': 1e-5,
        'l2_norm_clip': 3.0,
    }
    a_laplace_config = {
        'l2_norm_clip': 3.0, # 3.0
        'epsilon_total': 3.0, # Uses total ε as traditional laplace, but applies similar to gaussian, means ε increases with epochs.
        'batch_size': 1024,

        # not to use
        'noise_multiplier': 1.0,
        'delta': 1e-5,
        'l1_norm_clip': 3.0,
    }

    ## Configuration B: Comparable Utility (similar accuracy)
    # gaussian_config = {
    #     'l2_norm_clip': 3.0,
    #     'noise_multiplier': 1.1, # Will give ε = 3.1520 (approx) after 20 epochs.
    #     'delta': 1e-5,
    #     'batch_size': 1024,

    #     # not to use
    #     'l1_norm_clip': 3.0,
    #     'epsilon_total': 500.0,
    # }
    # t_laplace_config = {
    #     'l1_norm_clip': 3.0,
    #     'epsilon_total': 300.0,
    #     'batch_size': 1024,

    #     # not to use
    #     'noise_multiplier': 1.0,
    #     'delta': 1e-5,
    #     'l2_norm_clip': 3.0,
    # }
    # a_laplace_config = {
    #     'l2_norm_clip': 3.0,
    #     'epsilon_total': 10.0,
    #     'batch_size': 1024,

    #     # not to use
    #     'noise_multiplier': 1.0,
    #     'delta': 1e-5,
    #     'l1_norm_clip': 3.0,
    # }
    

    # Resolve CSVs
    config['train_data_path'] = wait_for_csv(config['train_data_path_pattern'])
    config['test_data_path'] = wait_for_csv(config['test_data_path_pattern'])

    # Seeds for reproducibility
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])

    # Create directories
    for d in ['models', 'logs', 'plots', 'data', 'federated_models', 'results', config['artifacts_dir']]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 80)
    print("CORRECTED DIFFERENTIAL PRIVACY COMPARISON - RESEARCH READY")
    print("1) TensorFlow Privacy Gaussian (Built-in DP-SGD with RDP accounting)")
    print("2) PyDP Laplace (Custom DP-SGD with Laplace mechanism)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Data preprocessing
    try:
        preprocessor = IoTDataPreprocessor(
            global_class_names=config['class_names'],
            artifacts_dir=config['artifacts_dir']
        )

        X_all, y_all, train_num_classes, artifacts = preprocessor.fit_preprocessor(
            config['train_data_path'], target_num_classes=config['num_classes']
        )

        # Enhanced stratification handling
        unique_classes, class_counts = np.unique(y_all, return_counts=True)
        min_class_count = int(np.min(class_counts))

        print(f"Dataset stats: {len(unique_classes)} classes, min class size: {min_class_count}")

        if min_class_count >= 2:
            stratify_label = y_all
            print("Using stratified split")
        else:
            stratify_label = None
            print("Some classes have <2 samples, using random split")

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=config['random_state'],
            stratify=stratify_label
        )

        X_test, y_test, test_num_classes = preprocessor.transform(
            config['test_data_path'], target_num_classes=config['num_classes']
        )
        
        # Handle missing test data
        if X_test is None or len(X_test) == 0:
            print("Test data not loaded, using validation data for final evaluation.")
            X_test, y_test = X_val, y_val

        # Convert to categorical
        y_train_cat = to_categorical(y_train, num_classes=config['num_classes'])
        y_val_cat = to_categorical(y_val, num_classes=config['num_classes'])
        y_test_cat = to_categorical(y_test, num_classes=config['num_classes'])

        print(f"Dataset prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        print(f"Features: {X_train.shape[1]}, Classes: {train_num_classes}")

    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    noise_types_to_test = ['t_laplace', 'a_laplace', 'gaussian'] # ['t_laplace', 'a_laplace', 'gaussian']

    all_results = {}
    
    for noise_type in noise_types_to_test:
        if noise_type == 't_laplace':
            current_config = t_laplace_config
        elif noise_type == 'a_laplace':
            current_config = a_laplace_config
        else:
            current_config = gaussian_config

        print("\n" + "=" * 60)
        print(f"TESTING APPROACH: {noise_type.upper()}")
        print("=" * 60)

        start_time = time.time()

        try:
            trainer = IoTModelTrainer(random_state=config['random_state'])

            model = trainer.create_model(
                input_dim=X_train.shape[1],
                num_classes=config['num_classes'],
                architecture=config['model_architecture']
            )

            # def train_model(self, X_train, y_train_cat, X_val, y_val_cat,
            #     model, epochs=20, batch_size=128, verbose=2,
            #     use_dp=True, noise_type='gaussian', 
                
            #     # Universal parameters
            #     l2_norm_clip=1.0,  # Used by Gaussian and Advanced Laplace
            #     l1_norm_clip=1.0,  # Used by Traditional Laplace only
                
            #     # Gaussian parameters
            #     noise_multiplier=1.0, delta=1e-5,
                
            #     # Laplace parameters  
            #     epsilon_total=10.0,  # Used by both Laplace methods
                
            #     learning_rate=1e-3):

            # 1. Gaussian mechanism
                # history_gaussian, _, eps_gaussian = trainer.train_model(
                #     X_train, y_train_cat, X_val, y_val_cat, model,
                #     noise_type='gaussian',
                #     noise_multiplier=1.0, l2_norm_clip=3.0, delta=1e-5
            # )

            # 2. Advanced Laplace (research paper) - Fair comparison
                # history_adv_laplace, _, eps_adv_laplace = trainer.train_model(  
                #     X_train, y_train_cat, X_val, y_val_cat, model,
                #     noise_type='a_laplace',
                #     epsilon_total=10.0, l2_norm_clip=3.0, delta=1e-5
            # )

            # 3. Traditional Laplace - For completeness
                # history_trad_laplace, _, eps_trad_laplace = trainer.train_model(
                #     X_train, y_train_cat, X_val, y_val_cat, model,
                #     noise_type='t_laplace', 
                #     epsilon_total=500.0, l1_norm_clip=3.0
            # )
            
            # Train
            history, final_eps = trainer.train_model(
                X_train, y_train_cat, X_val, y_val_cat,
                model=model,
                epochs=config['epochs'],
                batch_size=current_config['batch_size'],
                verbose=2,
                use_dp=True,
                noise_type=noise_type,
                l2_norm_clip=current_config['l2_norm_clip'],
                l1_norm_clip=current_config['l1_norm_clip'],
                noise_multiplier=eps,
                epsilon_total=current_config['epsilon_total'],
                delta=current_config['delta'],
                learning_rate=config['learning_rate']
            )

            training_success = True
            print(f"{noise_type.upper()} training completed successfully!")

        except Exception as e:
            print(f"ERROR training with {noise_type}: {e}")
            import traceback
            traceback.print_exc()
            training_success = False
            continue

        total_time = time.time() - start_time

        # Enhanced evaluation with error handling
        try:
            eval_results = evaluate_model(model, X_test, y_test_cat, class_names=config['class_names'])
            test_metrics = eval_results['test']
            print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"Test Macro F1-Score: {test_metrics.get('macro_f1', 0):.4f}")
        except Exception as e:
            print(f"Evaluation failed: {e}, using default metrics")
            test_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Store results
        acc = float(test_metrics.get('accuracy', 0.0) or 0.0)
        final_eps_safe = max(final_eps or 1e-6, 1e-6)

        results = {
            'noise_type': noise_type,
            'approach_info': {
                'algorithm': 'Gaussian' if noise_type == 'gaussian' else 'Laplace',
                'epsilon_calculation': 'RDP accounting' if noise_type == 'gaussian' else 'Pure DP accounting',
                'privacy_guarantees': 'Differential Privacy (ε,δ)-DP'
            },
            'training_samples': len(X_train),
            'batch_size': current_config['batch_size'],
            'epochs': config['epochs'],
            'learning_rate': config['learning_rate'],
            'final_test_metrics': test_metrics,
            'final_epsilon': final_eps_safe,
            'privacy_info_per_epoch': history.get('privacy_info', []),
            'training_time_sec': float(total_time),
            'training_success': history.get('training_success', False)
        }

        all_results[noise_type] = results

        # Save individual results
        path = f"Results_Fixed_Privacy_Budget/{noise_type}_dp_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved: {path}")

        print(f"{noise_type.upper():12} | ε = {final_eps_safe:.4f} | Test Acc = {acc:.4f} | Time = {total_time:.1f}s")

        # Cleanup
        try:
            del model, history, trainer
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass

        # Final comparison
        print("\n" + "=" * 100)
        print("FINAL CORRECTED DP COMPARISON")
        print("=" * 100)
        print(f"{'Approach':<12} {'Algorithm':<12} {'Epsilon':<12} {'TestAcc':<12} {'Time(s)':<12}")
        print("-" * 100)

        for k, v in all_results.items():
            if v['training_success']:
                alg = v['approach_info']['algorithm']
                eps = float(v['final_epsilon'] or 0.0)
                acc = float(v['final_test_metrics'].get('accuracy', 0.0) or 0.0)
                tt = float(v['training_time_sec'] or 0.0)

                print(f"{k:<12} {alg:<12} {eps:<12.4f} {acc:<12.4f} {tt:<12.1f}")
            else:
                print(f"{k:<12} {'-':<12} {'-':<12} {'-':<12} {'-':<12} {'-':<12}")

        # Save combined results
        combined_path = f"Results_Fixed_Privacy_Budget/combined_dp_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nCombined results saved: {combined_path}")

    print(f"\nCORRECTED DP comparison completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run corrected differential privacy comparison')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    
    main(epochs=args.epochs)
