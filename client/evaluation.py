import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)

class IoTModelEvaluator:
    def __init__(self, attack_type_map):
        """
        Initialize the model evaluator

        Parameters:
        -----------
        attack_type_map : dict
            Mapping of attack type names to class indices
        """
        self.attack_type_map = attack_type_map
        self.inv_attack_map = {v: k for k, v in attack_type_map.items()}

    def evaluate_model(self, model, X_test, y_test, y_test_cat):
        """
        Evaluate model performance with comprehensive metrics

        Parameters:
        -----------
        model : tf.keras.models.Sequential
            Trained model
        X_test : numpy.ndarray or pandas.DataFrame
            Test features
        y_test : numpy.ndarray or pandas.Series
            Test labels (original format)
        y_test_cat : numpy.ndarray
            One-hot encoded test labels

        Returns:
        --------
        eval_results : dict
            Dictionary containing evaluation metrics
        """
        print("\nEvaluating model performance...")

        # Generate predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = y_test.values if hasattr(y_test, 'values') else y_test

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        try:
            # ROC-AUC for multiclass
            roc_auc = roc_auc_score(y_test_cat, y_pred_proba, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"ROC-AUC calculation error: {e}")
            roc_auc = None

        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")

        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Get per-class metrics
        per_class_metrics = {}
        for class_id, class_name in self.inv_attack_map.items():
            per_class_metrics[class_name] = {
                'precision': class_report.get(str(class_id), {}).get('precision', 0),
                'recall': class_report.get(str(class_id), {}).get('recall', 0),
                'f1_score': class_report.get(str(class_id), {}).get('f1-score', 0),
                'support': class_report.get(str(class_id), {}).get('support', 0)
            }

        # Compile evaluation results
        eval_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist()
        }

        # Save evaluation results
        joblib.dump(eval_results, 'logs/evaluation_results.joblib')
        print("Evaluation results saved to 'logs/evaluation_results.joblib'")

        return eval_results

    def calculate_advanced_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate advanced metrics for model evaluation

        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        y_pred_proba : numpy.ndarray
            Predicted probabilities

        Returns:
        --------
        advanced_metrics : dict
            Dictionary containing advanced metrics
        """
        # Macro averaged metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')

        # Per-class metrics
        classes = np.unique(y_true)
        per_class_metrics = {}

        for cls in classes:
            cls_indices = y_true == cls

            # True positives, false positives, etc.
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            tn = np.sum((y_true != cls) & (y_pred != cls))

            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            per_class_metrics[self.inv_attack_map[cls]] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }

        return {
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'per_class_metrics': per_class_metrics
        }