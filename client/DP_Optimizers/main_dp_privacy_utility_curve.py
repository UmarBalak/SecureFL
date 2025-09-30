import os
import sys
import gc
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing_global import IoTDataPreprocessor
from evaluate import evaluate_model
from functions_dp import wait_for_csv
from training_dp_advanced import IoTModelTrainer

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
script_directory = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = "DATA/global_train.csv"
TEST_DATASET_PATH = "DATA/global_test.csv"
GLOBAL_MODEL_PATH = os.path.join(script_directory, "GLOBAL_MODELS")

def main(epochs=20):
    """
    Privacy-Utility Curves: Advanced Laplace vs Gaussian
    Tests multiple epsilon values for comprehensive comparison
    """
    config = {
        'train_data_path_pattern': DATASET_PATH,
        'test_data_path_pattern': TEST_DATASET_PATH,
        'epochs': epochs,
        'random_state': 42,
        'model_architecture': [256, 256],
        'learning_rate': 0.001,
        'num_classes': 15,
        'class_names': [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ],
        'artifacts_dir': "artifacts",
    }

    # Epsilon values and corresponding noise multipliers (calibrated)
    epsilon_values = [5] # [1, 3, 5, 10]
    noise_mult_values = [0.7936] # [2.0024, 0.9811, 0.7936, 0.6255]

    # Fixed parameters
    delta = 1e-5
    l2_clip = 3.0
    l1_clip = 3.0
    batch_size = 1024

    # Create output directory with timestamp
    output_dir = f"Results_Privacy_Utility_Curves"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    print("\n" + "=" * 80)
    print("PRIVACY-UTILITY CURVES: ADVANCED LAPLACE vs GAUSSIAN")
    print(f"Output directory: {output_dir}")
    print(f"Testing epsilon values: {epsilon_values}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Resolve CSVs and setup
    config['train_data_path'] = wait_for_csv(config['train_data_path_pattern'])
    config['test_data_path'] = wait_for_csv(config['test_data_path_pattern'])

    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])

    # Data preprocessing
    try:
        preprocessor = IoTDataPreprocessor(
            global_class_names=config['class_names'],
            artifacts_dir=config['artifacts_dir']
        )

        X_all, y_all, train_num_classes, artifacts = preprocessor.fit_preprocessor(
            config['train_data_path'], target_num_classes=config['num_classes']
        )

        # Stratified split
        unique_classes, class_counts = np.unique(y_all, return_counts=True)
        min_class_count = int(np.min(class_counts))
        print(f"Dataset: {len(unique_classes)} classes, min class size: {min_class_count}")

        stratify_label = y_all if min_class_count >= 2 else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=config['random_state'],
            stratify=stratify_label
        )

        X_test, y_test, test_num_classes = preprocessor.transform(
            config['test_data_path'], target_num_classes=config['num_classes']
        )

        if X_test is None or len(X_test) == 0:
            print("Using validation data for testing")
            X_test, y_test = X_val, y_val

        # Convert to categorical
        y_train_cat = to_categorical(y_train, num_classes=config['num_classes'])
        y_val_cat = to_categorical(y_val, num_classes=config['num_classes'])
        y_test_cat = to_categorical(y_test, num_classes=config['num_classes'])

        print(f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Storage for all results
    all_results = {
        'advanced_laplace': [],
        'gaussian': []
    }

    detailed_results = []

    # Main experiment loop
    for idx, target_eps in enumerate(epsilon_values):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {idx+1}/{len(epsilon_values)}: TARGET EPSILON = {target_eps}")
        print("=" * 80)

        # ===================================================================
        # ADVANCED LAPLACE
        # ===================================================================
        print(f"\n--- Advanced Laplace (ε={target_eps}) ---")
        start_time = time.time()

        try:
            trainer_adv = IoTModelTrainer(random_state=config['random_state'])
            model_adv = trainer_adv.create_model(
                input_dim=X_train.shape[1],
                num_classes=config['num_classes'],
                architecture=config['model_architecture']
            )

            history_adv, final_eps_adv = trainer_adv.train_model(
                X_train, y_train_cat, X_val, y_val_cat,
                model=model_adv,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                use_dp=True,
                noise_type='a_laplace',
                l2_norm_clip=l2_clip,
                epsilon_total=target_eps,
                delta=delta,
                learning_rate=config['learning_rate']
            )

            adv_time = time.time() - start_time

            # Evaluate
            eval_adv = evaluate_model(model_adv, X_test, y_test_cat, 
                                     class_names=config['class_names'])
            test_acc_adv = eval_adv['test']['accuracy']

            print(f"✓ Advanced Laplace: Acc={test_acc_adv:.4f}, ε={final_eps_adv:.4f}, Time={adv_time:.1f}s")

            # Store results
            all_results['advanced_laplace'].append({
                'epsilon': target_eps,
                'accuracy': test_acc_adv,
                'final_epsilon': final_eps_adv,
                'time': adv_time,
                'metrics': eval_adv['test']
            })

            adv_success = True

        except Exception as e:
            print(f"✗ Advanced Laplace FAILED: {e}")
            import traceback
            traceback.print_exc()
            adv_success = False
            test_acc_adv = 0.0
            final_eps_adv = 0.0
            adv_time = 0.0
            eval_adv = {'test': {}}

        finally:
            try:
                del model_adv, trainer_adv
                tf.keras.backend.clear_session()
                gc.collect()
            except:
                pass

        # ===================================================================
        # GAUSSIAN
        # ===================================================================
        print(f"\n--- Gaussian (ε={target_eps}, noise_mult={noise_mult_values[idx]:.4f}) ---")
        start_time = time.time()

        try:
            trainer_gauss = IoTModelTrainer(random_state=config['random_state'])
            model_gauss = trainer_gauss.create_model(
                input_dim=X_train.shape[1],
                num_classes=config['num_classes'],
                architecture=config['model_architecture']
            )

            history_gauss, final_eps_gauss = trainer_gauss.train_model(
                X_train, y_train_cat, X_val, y_val_cat,
                model=model_gauss,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                use_dp=True,
                noise_type='gaussian',
                l2_norm_clip=l2_clip,
                noise_multiplier=noise_mult_values[idx],
                delta=delta,
                learning_rate=config['learning_rate']
            )

            gauss_time = time.time() - start_time

            # Evaluate
            eval_gauss = evaluate_model(model_gauss, X_test, y_test_cat,
                                       class_names=config['class_names'])
            test_acc_gauss = eval_gauss['test']['accuracy']

            print(f"✓ Gaussian: Acc={test_acc_gauss:.4f}, ε={final_eps_gauss:.4f}, Time={gauss_time:.1f}s")

            # Store results
            all_results['gaussian'].append({
                'epsilon': target_eps,
                'accuracy': test_acc_gauss,
                'final_epsilon': final_eps_gauss,
                'time': gauss_time,
                'metrics': eval_gauss['test']
            })

            gauss_success = True

        except Exception as e:
            print(f"✗ Gaussian FAILED: {e}")
            import traceback
            traceback.print_exc()
            gauss_success = False
            test_acc_gauss = 0.0
            final_eps_gauss = 0.0
            gauss_time = 0.0
            eval_gauss = {'test': {}}

        finally:
            try:
                del model_gauss, trainer_gauss
                tf.keras.backend.clear_session()
                gc.collect()
            except:
                pass

        # Store detailed comparison
        detailed_results.append({
            'target_epsilon': target_eps,
            'noise_multiplier': noise_mult_values[idx],
            'advanced_laplace': {
                'accuracy': float(test_acc_adv),
                'final_epsilon': float(final_eps_adv),
                'time_seconds': float(adv_time),
                'test_metrics': eval_adv['test'] if adv_success else {}
            },
            'gaussian': {
                'accuracy': float(test_acc_gauss),
                'final_epsilon': float(final_eps_gauss),
                'time_seconds': float(gauss_time),
                'test_metrics': eval_gauss['test'] if gauss_success else {}
            },
            'accuracy_gap': float(test_acc_gauss - test_acc_adv)
        })

        print(f"\n📊 Gap: {(test_acc_gauss - test_acc_adv)*100:.2f}% (Gaussian advantage)")

    # ===================================================================
    # SAVE ALL RESULTS
    # ===================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # 1. Save detailed JSON
    with open(f"{output_dir}/detailed_results_5.json", "w") as f:
        json.dump(detailed_results, f, indent=4)
    print(f"✓ Saved: {output_dir}/detailed_results_5.json")

    # 2. Save summary CSV
    summary_df = pd.DataFrame({
        'Epsilon': epsilon_values,
        'Adv_Laplace_Acc': [r['advanced_laplace']['accuracy'] for r in detailed_results],
        'Adv_Laplace_Final_Eps': [r['advanced_laplace']['final_epsilon'] for r in detailed_results],
        'Gaussian_Acc': [r['gaussian']['accuracy'] for r in detailed_results],
        'Gaussian_Final_Eps': [r['gaussian']['final_epsilon'] for r in detailed_results],
        'Noise_Multiplier': noise_mult_values,
        'Accuracy_Gap': [r['accuracy_gap'] for r in detailed_results]
    })
    summary_df.to_csv(f"{output_dir}/summary_results_5.csv", index=False)
    print(f"✓ Saved: {output_dir}/summary_results_5.csv")

    # 3. Create privacy-utility curve plot
    fig, ax = plt.subplots(figsize=(10, 6))

    adv_eps = [r['epsilon'] for r in all_results['advanced_laplace']]
    adv_acc = [r['accuracy'] * 100 for r in all_results['advanced_laplace']]
    gauss_eps = [r['epsilon'] for r in all_results['gaussian']]
    gauss_acc = [r['accuracy'] * 100 for r in all_results['gaussian']]

    ax.plot(adv_eps, adv_acc, 'o-', label='Advanced Laplace', linewidth=2.5, markersize=10, color='#e74c3c')
    ax.plot(gauss_eps, gauss_acc, 's-', label='Gaussian', linewidth=2.5, markersize=10, color='#3498db')

    ax.set_xlabel('Privacy Budget (ε)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Privacy-Utility Tradeoff: Advanced Laplace vs Gaussian\n(δ=1e-5, L2 clip=3.0)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(epsilon_values) + 1)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/privacy_utility_curve.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/plots/privacy_utility_curve.pdf", bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/plots/privacy_utility_curve.png")
    print(f"✓ Saved: {output_dir}/plots/privacy_utility_curve.pdf")
    plt.close()

    # 4. Create comparison table
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Epsilon':<10} {'Adv Laplace Acc':<18} {'Gaussian Acc':<18} {'Gap':<12} {'Adv Time(s)':<15} {'Gauss Time(s)'}")
    print("-" * 100)

    for r in detailed_results:
        eps = r['target_epsilon']
        adv_acc = r['advanced_laplace']['accuracy'] * 100
        gauss_acc = r['gaussian']['accuracy'] * 100
        gap = r['accuracy_gap'] * 100
        adv_time = r['advanced_laplace']['time_seconds']
        gauss_time = r['gaussian']['time_seconds']
        print(f"{eps:<10} {adv_acc:<18.2f} {gauss_acc:<18.2f} {gap:<12.2f} {adv_time:<15.1f} {gauss_time:.1f}")

    print("\n" + "=" * 100)
    print(f"ALL RESULTS SAVED TO: {output_dir}/")
    print("=" * 100)
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    return all_results, output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Privacy-Utility Curves Experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    results, output_dir = main(epochs=args.epochs)
    with open(f"{output_dir}/all_results_5.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Success! Check {output_dir}/ for all results")
