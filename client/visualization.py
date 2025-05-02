import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

class IoTVisualizer:
    def __init__(self, attack_type_map, random_state=42):
        """
        Initialize the visualization class

        Parameters:
        -----------
        attack_type_map : dict
            Mapping of attack type names to class indices
        random_state : int
            Random seed for reproducibility
        """
        self.attack_type_map = attack_type_map
        self.inv_attack_map = {v: k for k, v in attack_type_map.items()}
        self.random_state = random_state

        # Ensure plots directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Create confusion matrix visualization

        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        """
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(self.attack_type_map.keys()),
                    yticklabels=list(self.attack_type_map.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved to 'plots/confusion_matrix.png'")

    def plot_training_history(self, history):
        """
        Plot training and validation metrics over epochs

        Parameters:
        -----------
        history : tf.keras.callbacks.History
            Training history
        """
        if not history:
            print("No training history available")
            return

        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig('plots/training_history.png')
        plt.close()
        print("Training history plot saved to 'plots/training_history.png'")

    def plot_roc_curves(self, X_test, y_test_cat, model):
        """
        Plot ROC curves for each class

        Parameters:
        -----------
        X_test : numpy.ndarray or pandas.DataFrame
            Test features
        y_test_cat : numpy.ndarray
            One-hot encoded test labels
        model : tf.keras.models.Sequential
            Trained model

        Returns:
        --------
        roc_auc : dict
            Dictionary containing ROC-AUC values for each class
        """
        y_pred_proba = model.predict(X_test)
        n_classes = y_test_cat.shape[1]

        plt.figure(figsize=(12, 10))

        # Plot ROC for each class
        roc_auc = {}
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label=f'{self.inv_attack_map[i]} (AUC = {roc_auc[i]:.2f})')

        # Plot random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('plots/roc_curves.png')
        plt.close()
        print("ROC curves saved to 'plots/roc_curves.png'")

        return roc_auc

    def plot_precision_recall_curves(self, X_test, y_test_cat, model):
        """
        Plot precision-recall curves for each class

        Parameters:
        -----------
        X_test : numpy.ndarray or pandas.DataFrame
            Test features
        y_test_cat : numpy.ndarray
            One-hot encoded test labels
        model : tf.keras.models.Sequential
            Trained model

        Returns:
        --------
        pr_auc : dict
            Dictionary containing PR-AUC values for each class
        """
        y_pred_proba = model.predict(X_test)
        n_classes = y_test_cat.shape[1]

        plt.figure(figsize=(12, 10))

        # Plot PR curve for each class
        pr_auc = {}
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_cat[:, i], y_pred_proba[:, i])
            pr_auc[i] = average_precision_score(y_test_cat[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, lw=2,
                     label=f'{self.inv_attack_map[i]} (AP = {pr_auc[i]:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig('plots/precision_recall_curves.png')
        plt.close()
        print("Precision-recall curves saved to 'plots/precision_recall_curves.png'")

        return pr_auc

    def plot_feature_importance(self, X_train, y_train, feature_names, n_features=20):
        """
        Plot feature importance using a Random Forest classifier

        Parameters:
        -----------
        X_train : numpy.ndarray or pandas.DataFrame
            Training features
        y_train : numpy.ndarray or pandas.Series
            Training labels
        feature_names : list
            List of feature names
        n_features : int
            Number of top features to plot
        """
        # Train a Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_train, y_train)

        # Get feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot top n_features
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(n_features), importances[indices[:n_features]], align='center')
        plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90)
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
        print("Feature importance plot saved to 'plots/feature_importance.png'")

        # Save feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        feature_importance.to_csv('logs/feature_importance.csv', index=False)
        print("Feature importance data saved to 'logs/feature_importance.csv'")

        return feature_importance

    def plot_class_distribution(self, y, title_prefix=""):
        """
        Plot class distribution

        Parameters:
        -----------
        y : numpy.ndarray or pandas.Series
            Class labels
        title_prefix : str
            Prefix for plot title
        """
        plt.figure(figsize=(12, 6))

        # Convert to Series if numpy array
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Map numerical labels to attack types
        y_mapped = y.map(self.inv_attack_map)

        # Plot distribution
        ax = sns.countplot(x=y_mapped, order=y_mapped.value_counts().index)
        plt.title(f'{title_prefix} Class Distribution')
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

        # Add count labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha = 'center', va = 'bottom',
                      xytext = (0, 5), textcoords = 'offset points')

        plt.tight_layout()
        file_suffix = title_prefix.lower().replace(' ', '_')
        plt.savefig(f'plots/class_distribution_{file_suffix}.png')
        plt.close()
        print(f"Class distribution plot saved to 'plots/class_distribution_{file_suffix}.png'")

    def visualize_data_embeddings(self, X, y, method='tsne'):
        """
        Visualize data embeddings using dimensionality reduction

        Parameters:
        -----------
        X : numpy.ndarray or pandas.DataFrame
            Features
        y : numpy.ndarray or pandas.Series
            Labels
        method : str
            Dimensionality reduction method ('tsne' or 'pca')
        """
        # Sample data if too large
        max_samples = 5000
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y

        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.random_state)
            title = 't-SNE Visualization of Attack Types'
            filename = 'plots/tsne_visualization.png'
        else:  # PCA
            reducer = PCA(n_components=2, random_state=self.random_state)
            title = 'PCA Visualization of Attack Types'
            filename = 'plots/pca_visualization.png'

        # Transform data
        X_transformed = reducer.fit_transform(X_sample)

        # Convert to DataFrame for plotting
        embedding_df = pd.DataFrame({
            'Component 1': X_transformed[:, 0],
            'Component 2': X_transformed[:, 1],
            'Attack Type': [self.inv_attack_map[label] for label in y_sample]
        })

        # Plot
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=embedding_df,
            x='Component 1',
            y='Component 2',
            hue='Attack Type',
            palette='tab20',
            alpha=0.7
        )
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Data embedding visualization saved to '{filename}'")

    def plot_per_class_metrics(self, class_metrics):
        """
        Plot performance metrics for each class

        Parameters:
        -----------
        class_metrics : dict
            Dictionary containing metrics for each class
        """
        metrics = ['precision', 'recall', 'f1_score']

        # Create dataframe from class metrics
        data = []
        for attack_type, metrics_dict in class_metrics.items():
            row = {'Attack Type': attack_type}
            row.update({k: v for k, v in metrics_dict.items() if k in metrics})
            data.append(row)

        df = pd.DataFrame(data)

        # Melt dataframe for plotting
        df_melted = df.melt(id_vars=['Attack Type'],
                           value_vars=metrics,
                           var_name='Metric',
                           value_name='Value')

        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Attack Type', y='Value', hue='Metric', data=df_melted)
        plt.title('Performance Metrics by Attack Type')
        plt.xlabel('Attack Type')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig('plots/per_class_metrics.png')
        plt.close()
        print("Per-class metrics plot saved to 'plots/per_class_metrics.png'")