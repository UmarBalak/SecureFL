# evaluate_with_wandb.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from unified_fl_tracker import fl_tracker
import wandb

def evaluate_model_with_wandb(model, X_test, y_test_cat, class_names=None, save_plots=True, 
                              fl_round=None, log_to_wandb=True):
    """
    Production-grade evaluation with WandB integration for FL
    """
    print("\n" + "="*70)
    print("🔧 Production Model Evaluation with WandB Integration")
    print("="*70)
    
    def compute_stable_metrics(X, y_cat, dataset_name="Test Set"):
        """
        Stable metric computation with WandB logging
        """
        print(f"\n📊 Evaluating {dataset_name}...")
        
        # Clean input validation
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️ Cleaning NaN/Inf values in input data...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y_cat)) or np.any(np.isinf(y_cat)):
            print("⚠️ Cleaning NaN/Inf values in target data...")
            y_cat = np.nan_to_num(y_cat, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            # Manual prediction-based evaluation (bypasses TF evaluate() bug)
            print("🎯 Computing predictions with numerical stability...")
            y_pred_probs = model.predict(X, verbose=0, batch_size=128)
            
            # Ensure valid probabilities (critical for loss calculation)
            y_pred_probs = np.clip(y_pred_probs, 1e-15, 1.0 - 1e-15)
            
            # Manual categorical crossentropy (the correct loss)
            manual_loss = -np.mean(np.sum(y_cat * np.log(y_pred_probs), axis=1))
            
            # Accuracy calculation
            y_true_classes = np.argmax(y_cat, axis=1)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            accuracy = np.mean(y_true_classes == y_pred_classes)
            
            # Try TensorFlow's evaluate for comparison
            try:
                tf_results = model.evaluate(X, y_cat, verbose=0, batch_size=128)
                tf_loss = tf_results[0] if isinstance(tf_results, list) else tf_results
                
                if tf_loss > 100:  # TensorFlow bug detected
                    print(f"🐛 TensorFlow evaluate() bug detected: {tf_loss:.2f}")
                    print(f"✅ Using manual calculation: {manual_loss:.4f}")
                    final_loss = manual_loss
                else:
                    print(f"📊 TensorFlow evaluate() working correctly: {tf_loss:.4f}")
                    final_loss = tf_loss
                    
            except Exception as e:
                print(f"⚠️ TensorFlow evaluate() failed: {e}")
                final_loss = manual_loss

            print(f"✅ {dataset_name} - Loss: {final_loss:.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error in prediction computation: {e}")
            return {"error": str(e)}
        
        # Detailed sklearn metrics
        try:
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average=None, zero_division=0
            )
            
            # Aggregate metrics
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average='macro', zero_division=0
            )
            
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            
        except Exception as e:
            print(f"❌ Error in sklearn metrics: {e}")
            return {"error": str(e)}
        
        # Print comprehensive results
        print(f"\n📋 {dataset_name} Per-Class Performance:")
        print("-" * 60)
        
        class_labels = class_names if class_names else [f"Class_{i}" for i in range(len(precision))]
        
        for i, label in enumerate(class_labels):
            if i < len(precision):
                print(f"{label:20}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, "
                      f"F1={f1[i]:.4f}, Support={support[i]:4d}")
        
        print(f"\n📈 {dataset_name} Aggregate Performance:")
        print(f"Macro Avg    : Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}")
        print(f"Weighted Avg : Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1={weighted_f1:.4f}")
        
        # Enhanced confusion matrix plot
        if save_plots:
            try:
                plot_confusion_matrix_enhanced_wandb(
                    cm, class_labels, dataset_name, 
                    accuracy, macro_f1, weighted_f1, final_loss, fl_round
                )
            except Exception as e:
                print(f"⚠️ Plot generation warning: {e}")
        
        return {
            'loss': float(final_loss),
            'accuracy': float(accuracy),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'confusion_matrix': cm.tolist(),
            'num_classes': len(class_labels),
            'total_samples': len(y_true_classes),
            'y_true': y_true_classes.tolist(),  # For additional analysis
            'y_pred': y_pred_classes.tolist()   # For additional analysis
        }
    
    # Main evaluation
    results = compute_stable_metrics(X_test, y_test_cat, "Test Set")
    
    # Print production summary
    print(f"\n" + "="*70)
    print("🎯 PRODUCTION EVALUATION SUMMARY WITH WANDB")
    print("="*70)
    print(f"Test Accuracy      : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Test Loss          : {results['loss']:.4f}")
    print(f"Macro F1-Score     : {results['macro_f1']:.4f}")
    print(f"Weighted F1-Score  : {results['weighted_f1']:.4f}")
    print(f"Total Test Samples : {results['total_samples']:,}")
    print(f"Number of Classes  : {results['num_classes']}")
    
    # Performance interpretation
    if results['loss'] < 2.0:
        print("✅ Loss value is in excellent range for multiclass classification!")
    elif results['loss'] < 5.0:
        print("✅ Loss value is in good range for cybersecurity data!")
    else:
        print("⚠️ Loss is high - consider model architecture adjustments")
    
    if results['accuracy'] > 0.60:
        print("✅ Accuracy is good for imbalanced cybersecurity data!")
    elif results['accuracy'] > 0.45:
        print("✅ Accuracy is acceptable for 15-class cybersecurity classification!")
    else:
        print("⚠️ Accuracy is low - consider data balancing techniques")
    
    if results['macro_f1'] < 0.40:
        print("📊 Low macro F1 indicates class imbalance - this is normal for cybersecurity data")
        print("💡 Consider: class weights, SMOTE, or focal loss for improvement")
    
    print("="*70)
    
    return {'test': results}

def plot_confusion_matrix_enhanced_wandb(cm, class_labels, dataset_name, accuracy, 
                                        macro_f1, weighted_f1, loss, fl_round=None):
    """
    Enhanced confusion matrix with WandB integration
    """
    os.makedirs('evaluation_plots', exist_ok=True)
    
    plt.figure(figsize=(16, 12))
    
    # Create heatmap
    mask = cm == 0
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                square=True,
                mask=mask,
                cbar_kws={'label': 'Number of Samples'})
    
    title_text = f'{dataset_name} - Cybersecurity Attack Classification'
    if fl_round is not None:
        title_text += f' (FL Round {fl_round})'
    
    plt.title(f'{title_text}\n'
             f'Accuracy: {accuracy:.3f} | Loss: {loss:.3f} | Macro F1: {macro_f1:.3f} | Weighted F1: {weighted_f1:.3f}',
             fontsize=14, pad=20)
    
    plt.xlabel('Predicted Attack Type', fontsize=12)
    plt.ylabel('True Attack Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add FL-specific performance interpretation
    plt.figtext(0.02, 0.02,
                f"📊 FL Performance: Round {fl_round if fl_round else 'N/A'}\n"
                f"🔒 Dataset: ML-Edge-IIoT Non-IID (15 classes)\n"
                f"⚡ Evaluation: Production-grade with WandB tracking",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'evaluation_plots/{dataset_name.lower().replace(" ", "_")}_fl_round_{fl_round if fl_round else "unknown"}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Log to WandB
    try:
        wandb.log({f"confusion_matrix_plot_round_{fl_round}": wandb.Image(filename)})
        print(f"📊 Confusion matrix logged to WandB: {filename}")
    except:
        print(f"⚠️ Could not log confusion matrix to WandB")
    
    plt.close()

# Alias for backward compatibility
evaluate_model = evaluate_model_with_wandb
