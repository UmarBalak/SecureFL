# main_global_complete.py - Complete global model training script

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv

# Local imports
from preprocessing_global import IoTDataPreprocessor
from training_global_with_wandb import GlobalModelTrainer
from functions import upload_file, save_weights, wait_for_csv
from evaluate_global_with_wandb import evaluate_model_with_wandb
from unified_fl_tracker import fl_tracker
import wandb

load_dotenv(dotenv_path=".env.server")

# Configuration
TRAIN_DATASET_PATH = "./DATA/DATA_VARIABLE/global_train.csv"
TEST_DATASET_PATH = "./DATA/DATA_VARIABLE/global_test.csv"
SEED = 42

# Set reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main_global_training_complete(epochs=100):
    """
    COMPLETE: Global model training with comprehensive WandB tracking
    Goal: Train initial global model that clients will use for FL
    """
    
    print("="*80)
    print("🎯 FEDERATED LEARNING - GLOBAL MODEL TRAINING")
    print("="*80)
    print("Goal: Train initial baseline model for FL system")
    print("Output: Global model weights + preprocessing artifacts")
    print("="*80)
    
    # Enhanced configuration
    config = {
        'train_data_path': TRAIN_DATASET_PATH,
        'test_data_path': TEST_DATASET_PATH,
        'epochs': epochs,
        'batch_size': 128,
        'random_state': SEED,
        'model_architecture': [256, 256],
        'learning_rate': 5e-5,
        'num_classes': 15,
        'class_names': [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ],
        'artifacts_dir': "artifacts",
        'dataset': "ML-Edge-IIoT",
        'features': 25,
        'component': "global_training",
        'purpose': "initial_fl_baseline"
    }
    
    # Initialize WandB for global training
    global_run = fl_tracker.initialize_global_training_run(config=config)
    
    try:
        # Wait for data files
        config['train_data_path'] = wait_for_csv(config['train_data_path'])
        config['test_data_path'] = wait_for_csv(config['test_data_path'])
        
        # Create directories
        directories = [
            'models', 'models/global_checkpoints', 'logs', 'plots', 
            'data', 'federated_models', config['artifacts_dir']
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)
        
        print(f"\n🎯 Global Training Configuration:")
        print(f"   WandB Run: {wandb.run.name}")
        print(f"   WandB URL: {wandb.run.url}")
        print(f"   Architecture: {config['model_architecture']}")
        print(f"   Learning Rate: {config['learning_rate']}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {config['batch_size']}")
        
        # 1. Preprocessing - Fit on global data
        print("\n" + "="*60)
        print("🔧 STEP 1: Global Preprocessing Setup")
        print("="*60)
        
        preprocessor = IoTDataPreprocessor(
            global_class_names=config['class_names'],
            artifacts_dir=config['artifacts_dir']
        )
        
        X_all, y_all, actual_classes, artifacts = preprocessor.fit_preprocessor(
            config['train_data_path'], target_num_classes=config['num_classes']
        )
        
        print(f"✅ Preprocessing complete:")
        print(f"   Original features: 25")
        print(f"   Enhanced features: {X_all.shape[1]}")
        print(f"   Total samples: {len(X_all)}")
        print(f"   Classes present: {actual_classes}/{config['num_classes']}")
        
        # Log preprocessing to WandB
        wandb.log({
            "preprocessing/original_features": 25,
            "preprocessing/enhanced_features": X_all.shape[1],
            "preprocessing/expansion_ratio": X_all.shape[1] / 25,
            "preprocessing/total_samples": len(X_all),
            "preprocessing/classes_present": actual_classes,
            "preprocessing/target_classes": config['num_classes']
        })
        
        # Save preprocessing artifacts
        preprocessor_path = os.path.join(config['artifacts_dir'], "preprocessor.pkl")
        global_le_path = os.path.join(config['artifacts_dir'], "global_label_encoder.pkl")
        feature_info_path = os.path.join(config['artifacts_dir'], "feature_info.pkl")
        
        joblib.dump(artifacts['preprocessor'], preprocessor_path)
        joblib.dump(artifacts['global_label_encoder'], global_le_path)
        joblib.dump(artifacts['feature_info'], feature_info_path)
        
        # Upload artifacts
        try:
            upload_file(preprocessor_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
            upload_file(global_le_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
            upload_file(feature_info_path, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
            
            # Save to WandB
            wandb.save(preprocessor_path)
            wandb.save(global_le_path)
            wandb.save(feature_info_path)
            
            print("✅ Preprocessing artifacts uploaded and saved to WandB")
            
        except Exception as e:
            print(f"⚠️ Artifact upload warning: {e}")
        
        # 2. Data splitting
        print("\n" + "="*60)
        print("🔧 STEP 2: Train/Validation Split")
        print("="*60)
        
        min_class_count = int(np.min(np.bincount(y_all)))
        stratify_label = y_all if min_class_count >= 2 else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=config['random_state'],
            stratify=stratify_label
        )
        
        print(f"✅ Data split complete:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Validation samples: {X_val.shape[0]}")
        print(f"   Feature dimensions: {X_train.shape[1]}")
        
        # Log class distribution
        class_distribution = np.bincount(y_all, minlength=config['num_classes'])
        class_dist_dict = {}
        for i, class_name in enumerate(config['class_names']):
            class_dist_dict[f"data_distribution/{class_name}"] = int(class_distribution[i])
        
        wandb.log(class_dist_dict)
        
        # 3. Test data preparation
        print("\n" + "="*60)
        print("🔧 STEP 3: Test Data Preparation")
        print("="*60)
        
        X_test, y_test, test_num_classes = preprocessor.transform(
            config['test_data_path'], target_num_classes=config['num_classes']
        )
        
        print(f"✅ Test data prepared:")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Test classes found: {test_num_classes}")
        
        # 4. Model creation
        print("\n" + "="*60)
        print("🔧 STEP 4: Global Model Architecture")
        print("="*60)
        
        trainer = GlobalModelTrainer(random_state=config['random_state'])
        model = trainer.create_model(
            input_dim=X_train.shape[1],
            num_classes=config['num_classes'],
            architecture=config['model_architecture']
        )
        
        print(f"✅ Model architecture created:")
        print(f"   Input dimension: {X_train.shape[1]}")
        print(f"   Output dimension: {config['num_classes']}")
        print(f"   Hidden layers: {config['model_architecture']}")
        
        # Log model details
        try:
            total_params = model.count_params()
            wandb.log({
                "model/total_parameters": total_params,
                "model/input_dim": X_train.shape[1],
                "model/output_dim": config['num_classes'],
                "model/hidden_layers": len(config['model_architecture'])
            })
            
            # Log model architecture
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary_str = '\n'.join(model_summary)
            fl_tracker.log_model_architecture(model_summary_str)
            
        except Exception as e:
            print(f"⚠️ Model logging warning: {e}")
        
        # 5. Prepare training data
        print("\n" + "="*60)
        print("🔧 STEP 5: Data Encoding for Training")
        print("="*60)
        
        # CRITICAL: Always use 15 classes for FL compatibility
        y_train_cat = to_categorical(y_train, num_classes=config['num_classes'])
        y_val_cat = to_categorical(y_val, num_classes=config['num_classes'])
        y_test_cat = to_categorical(y_test, num_classes=config['num_classes'])
        
        print(f"✅ Categorical encoding complete:")
        print(f"   Train shape: {y_train_cat.shape}")
        print(f"   Validation shape: {y_val_cat.shape}")
        print(f"   Test shape: {y_test_cat.shape}")
        print(f"   Fixed classes: {config['num_classes']} (FL compatible)")
        
        # 6. Global model training
        print("\n" + "="*60)
        print("🚀 STEP 6: GLOBAL MODEL TRAINING")
        print("="*60)
        
        history, training_time, num_samples = trainer.train_global_model(
            X_train, y_train_cat, X_val, y_val_cat,
            model=model,
            architecture=config['model_architecture'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_classes=config['num_classes'],
            verbose=2,
            wandb_tracker=fl_tracker
        )
        
        model = trainer.get_model()
        
        print(f"✅ Global model training completed:")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Samples processed: {num_samples}")
        
        # 7. Save global model
        print("\n" + "="*60)
        print("🔧 STEP 7: Save Global Model")
        print("="*60)
        
        weights_path = save_weights(model, "models", "g0.h5")
        arch_file = os.path.join("models", "global_model_architecture.json")
        with open(arch_file, "w") as f:
            f.write(model.to_json())
        
        print(f"✅ Global model saved:")
        print(f"   Weights: {weights_path}")
        print(f"   Architecture: {arch_file}")
        
        # 8. Comprehensive evaluation
        print("\n" + "="*60)
        print("📊 STEP 8: GLOBAL MODEL EVALUATION")
        print("="*60)
        
        eval_results = evaluate_model_with_wandb(
            model, X_test, y_test_cat,
            class_names=config['class_names'],
            fl_round=0,  # This is the baseline global model
            log_to_wandb=True
        )
        test_metrics = eval_results['test']
        
        print(f"✅ Global model evaluation complete:")
        print(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"   Test Loss: {test_metrics['loss']:.4f}")
        print(f"   Macro F1-Score: {test_metrics['macro_f1']:.4f}")
        print(f"   Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
        
        # Log evaluation results to WandB
        fl_tracker.log_global_evaluation_results(test_metrics, config['class_names'])
        
        # 9. Save complete metadata
        print("\n" + "="*60)
        print("🔧 STEP 9: Save Metadata & Artifacts")
        print("="*60)
        
        complete_metadata = {
            "component": "global_training",
            "purpose": "initial_fl_baseline",
            "test_metrics": test_metrics,
            "model_architecture": config['model_architecture'],
            "learning_rate": config['learning_rate'],
            "enhanced_features": True,
            "original_features": 25,
            "feature_dimensions": X_train.shape[1],
            "preprocessing_pipeline": "StandardScaler + OneHotEncoder",
            "epochs": epochs,
            "batch_size": config['batch_size'],
            "num_training_samples": num_samples,
            "global_classes": config['class_names'],
            "num_classes_fixed": config['num_classes'],
            "classes_present_in_data": int(actual_classes),
            "training_duration_seconds": training_time,
            "wandb_run_id": wandb.run.id,
            "wandb_run_name": wandb.run.name,
            "wandb_project": fl_tracker.project_name,
            "wandb_url": wandb.run.url,
            "timestamp": datetime.now().isoformat(),
            "ready_for_fl": True
        }
        
        metadata_file = os.path.join("models", "global_model_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(complete_metadata, f, indent=2)
        
        # Upload final artifacts
        try:
            upload_file(weights_path, os.getenv("SERVER_CONTAINER_NAME"), test_metrics)
            upload_file(metadata_file, os.getenv("SERVER_CONTAINER_NAME"), metadata={})
            
            # Save to WandB
            wandb.save(weights_path)
            wandb.save(metadata_file)
            wandb.save(arch_file)
            
            print("✅ All artifacts uploaded and saved to WandB")
            
        except Exception as e:
            print(f"⚠️ Final upload warning: {e}")
        
        # 10. Final summary
        final_summary = {
            "global_training_complete": True,
            "ready_for_fl": True,
            "preprocessing_artifacts_saved": True,
            "final_test_accuracy": test_metrics['accuracy'],
            "final_test_loss": test_metrics['loss'],
            "final_macro_f1": test_metrics['macro_f1'],
            "wandb_tracking_complete": True,
            "baseline_model_ready": True
        }
        
        print("\n" + "="*80)
        print("🎉 GLOBAL MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"   WandB Run: {wandb.run.name}")
        print(f"   WandB URL: {wandb.run.url}")
        print(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"   Test Loss: {test_metrics['loss']:.4f}")
        print(f"   Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"   Model ready for FL clients!")
        print("="*80)
        
        return {
            "test_metrics": test_metrics,
            "wandb_run_id": wandb.run.id,
            "wandb_url": wandb.run.url,
            "global_model_ready": True,
            "summary": final_summary
        }
        
    except Exception as e:
        print(f"❌ Global training failed: {e}")
        wandb.log({"global_training_error": str(e)})
        raise
        
    finally:
        # Finalize WandB run
        try:
            fl_tracker.finalize_run(summary_metrics=final_summary if 'final_summary' in locals() else {})
            print("✅ WandB global training run finalized")
        except Exception as e:
            print(f"⚠️ WandB finalization warning: {e}")

if __name__ == "__main__":
    try:
        print("🚀 Starting Global Model Training for FL System...")
        result = main_global_training_complete(epochs=10)  # Test with 10 epochs
        print(f"🎯 Global Training Success!")
        print(f"   WandB Dashboard: {result['wandb_url']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 Global training script completed")
