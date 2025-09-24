import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class IoTDataPreprocessor:
    def __init__(self):
        # Expanded protected sparse features to retain more IoT-relevant columns
        self.protected_sparse_features = [
            'tcp.flags', 'tcp.flags.ack', 'tcp.dstport', 'tcp.srcport',
            'dns.qry.name.len', 'mqtt.topic_len', 'mqtt.hdrflags',
            'mqtt.len', 'mqtt.msgtype', 'mqtt.ver',
            'http.content_length', 'dns.qry.type', 'dns.qry.qu',
            'tcp.ack', 'tcp.len', 'tcp.seq', 'tcp.connection.fin',
            'tcp.connection.rst', 'tcp.connection.syn', 'tcp.connection.synack',
            'udp.port', 'udp.stream', 'udp.time_delta',
            'icmp.checksum', 'icmp.seq_le', 'arp.opcode', 'arp.hw.size',
            'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags',
            'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id'
        ]
        # Categorical columns to encode
        self.categorical_columns = [
            'http.request.method', 'dns.qry.qu', 'dns.qry.type',
            'mqtt.msg_decoded_as', 'mqtt.protoname'
        ]

    def save_selected_features(self, df, final_features):
        # Path to save the new CSV file with selected features
        new_csv_path = "selected_features_dataset.csv"

        # Filter the dataframe by selected features if they exist in the original data
        existing_features = [f for f in final_features if f in df.columns]
        missing_features = [f for f in final_features if f not in df.columns]

        if missing_features:
            print(f"Warning: The following selected features were not found in the original dataset and will be skipped:\n{missing_features}")

        # Create new dataframe with only selected features (existing ones)
        df_selected = df[existing_features].copy()

        # Save to new CSV
        df_selected.to_csv(new_csv_path, index=False)

        print(f"New CSV file with selected features saved as: {new_csv_path}")

    def analyze_correlations(self, df, target_col, threshold=0.1, plot=True):
        """
        Analyze correlations between features and target column
        """
        print(f"\n{'='*50}")
        print("CORRELATION ANALYSIS")
        print(f"{'='*50}")
        
        # Calculate correlations with target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations.drop(target_col)
        
        print(f"\nTop 15 features correlated with {target_col}:")
        print("-" * 45)
        for feature, corr in correlations.head(15).items():
            print(f"{feature:<30}: {corr:.4f}")
        
        # Features above threshold
        high_corr_features = correlations[correlations >= threshold].index.tolist()
        print(f"\nFeatures with |correlation| >= {threshold}: {len(high_corr_features)}")
        
        # Plot correlation heatmap for top features
        if plot and len(high_corr_features) > 0:
            plt.figure(figsize=(12, 8))
            top_features = high_corr_features[:20] + [target_col]  # Top 20 + target
            corr_matrix = df[top_features].corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, fmt='.3f')
            plt.title(f'Correlation Heatmap: Top Features vs {target_col}')
            plt.tight_layout()
            plt.show()
        
        return high_corr_features, correlations

    def remove_multicollinearity(self, df, features, threshold=0.95):
        """
        Remove highly correlated features among themselves to reduce multicollinearity
        """
        print(f"\n{'='*50}")
        print("MULTICOLLINEARITY ANALYSIS")
        print(f"{'='*50}")
        
        if len(features) <= 1:
            return features
            
        # Calculate correlation matrix for selected features
        corr_matrix = df[features].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (>= {threshold}):")
        
        # Remove redundant features
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"{feat1} <-> {feat2}: {corr_val:.4f}")
            # Keep the feature that appears first in our original list
            if feat2 not in features_to_remove:
                features_to_remove.add(feat2)
        
        final_features = [f for f in features if f not in features_to_remove]
        print(f"\nRemoved {len(features_to_remove)} redundant features")
        print(f"Final feature count: {len(final_features)}")
        
        return final_features

    def mutual_information_selection(self, X, y, feature_names, k=25):
        """
        Use mutual information for feature selection
        """
        print(f"\n{'='*50}")
        print("MUTUAL INFORMATION ANALYSIS")
        print(f"{'='*50}")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
        
        print(f"\nTop 15 features by Mutual Information:")
        print("-" * 45)
        for feature, score in mi_series.head(15).items():
            print(f"{feature:<30}: {score:.4f}")
        
        # Select top k features
        top_features = mi_series.head(k).index.tolist()
        print(f"\nSelected top {k} features by Mutual Information")
        
        return top_features, mi_series

    def preprocess_data(self, path, correlation_threshold=0.05, 
                       multicollinearity_threshold=0.95, 
                       final_feature_count=25):
        print(f"Found dataset: {path}\n{'='*70}")
        df = pd.read_csv(path)

        target_column = 'Attack_type'

        print("DataFrame Info:\n" + "-"*30)
        print(f"Shape: {df.shape}")
        print(f"Missing Values: {df.isna().sum().sum()} ({df.isna().sum().sum() / df.size * 100:.2f}%)")
        print(f"Duplicate Rows: {df.duplicated().sum()}")

        # Handle missing values
        print("\nHandling missing values...")
        for col in df.columns:
            if col in self.categorical_columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            else:
                df[col].fillna(0, inplace=True)

        # Drop constant columns
        print("\nRemoving constant columns...")
        constant_dropped = []
        for col in df.columns:
            if df[col].nunique() == 1:
                df.drop(col, axis=1, inplace=True)
                constant_dropped.append(col)
        if constant_dropped:
            print(f"Dropped constant columns: {constant_dropped}")

        # Drop irrelevant high-cardinality or unstructured text columns
        drop_columns = [
            'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
            'http.file_data', 'http.request.full_uri', 'http.request.version',
            'tcp.options', 'tcp.payload', 'mqtt.msg', 'mqtt.topic',
            'http.referer', 'dns.qry.name'
        ]
        dropped = [col for col in drop_columns if col in df.columns]
        df.drop(columns=dropped, inplace=True)
        if dropped:
            print(f"\nDropped irrelevant columns: {dropped}")

        # Encode categorical columns
        print("\nEncoding categorical columns...")
        le_dict = {}
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
                print(f"Encoded {col} with {len(le.classes_)} unique values")

        # Drop sparse features with higher threshold
        zero_dropped = []
        for col in df.columns:
            if col in self.protected_sparse_features or col in ['Attack_label', 'Attack_type']:
                continue
            try:
                if df[col].dtype in [np.float64, np.int64]:
                    zero_ratio = (df[col] == 0).sum() / len(df)
                    if zero_ratio > 0.85:
                        df.drop(col, axis=1, inplace=True)
                        zero_dropped.append((col, zero_ratio))
            except:
                continue
        if zero_dropped:
            print("\nZero-dominant columns dropped:")
            for col, ratio in zero_dropped:
                print(f"{col}: {ratio:.2%} zeros")
        print(f"Shape after zero-drop: {df.shape}")

        # Encode Attack_type
        le = LabelEncoder()
        df['Attack_type'] = le.fit_transform(df['Attack_type'])

        print("\n--- Class Distributions Before Augmentation ---")
        print("Attack_label:\n", df['Attack_label'].value_counts())
        print("Attack_type:\n", df['Attack_type'].value_counts())

        # Ensure all columns are numeric before correlation analysis
        print("\nEnsuring all features are numeric for correlation analysis...")
        feature_columns = [col for col in df.columns if col not in ['Attack_label', 'Attack_type']]
        
        # Convert any remaining non-numeric columns
        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                print(f"Converting {col} to numeric...")
                # Try to convert to numeric, coerce errors to NaN, then fill with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Double-check: remove any columns that still can't be converted
        non_numeric_final = df[feature_columns].select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_final) > 0:
            print(f"Dropping final non-numeric columns: {non_numeric_final.tolist()}")
            df.drop(columns=non_numeric_final, inplace=True)
        
        print(f"Shape before correlation analysis: {df.shape}")
        
        # CORRELATION-BASED FEATURE SELECTION
        # Step 1: Find features correlated with target
        target_features, all_correlations = self.analyze_correlations(
            df, 'Attack_type', threshold=correlation_threshold, plot=False
        )
        
        # Step 2: Remove multicollinearity among selected features
        if len(target_features) > 1:
            final_features = self.remove_multicollinearity(
                df, target_features, threshold=multicollinearity_threshold
            )
        else:
            final_features = target_features

        # Ensure we have enough features
        if len(final_features) < final_feature_count:
            print(f"\nWarning: Only {len(final_features)} features found with correlation >= {correlation_threshold}")
            print("Adding more features based on correlation ranking...")
            additional_needed = final_feature_count - len(final_features)
            remaining_features = [f for f in all_correlations.index 
                                if f not in final_features and f not in ['Attack_label', 'Attack_type']]
            final_features.extend(remaining_features[:additional_needed])

        print(f"\nFinal selected features ({len(final_features)}):")
        for i, feat in enumerate(final_features, 1):
            corr_val = all_correlations.get(feat, 0)
            print(f"{i:2d}. {feat:<30}: {corr_val:.4f}")

        # Prepare data for processing
        feature_df = df[final_features].copy()
        y_multiclass = df['Attack_type'].copy()

        # Ensure only numeric columns remain
        non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"\nWarning: Dropping non-numeric columns: {non_numeric.tolist()}")
            feature_df.drop(columns=non_numeric, inplace=True)
            final_features = [f for f in final_features if f not in non_numeric]

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)

        # --- No SMOTE here ---
        X_resampled, y_resampled = X_scaled, y_multiclass.to_numpy()
        print("\nSMOTE skipped (central schema only)")

        # Final validation with Mutual Information
        mi_features, mi_scores = self.mutual_information_selection(
            X_resampled, y_resampled, final_features, k=len(final_features)
        )

        num_classes = len(np.unique(y_resampled))
        print(f"\n✅ Preprocessing complete!")
        print(f"Final shape: {X_resampled.shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Selected features: {len(final_features)}")
        
        for i, feat in enumerate(final_features, 1):
            print(f"{i:2d}. {feat}")

        self.save_selected_features(df, final_features+[target_column])

        return X_resampled, y_resampled, num_classes, final_features


# Usage example
if __name__ == "__main__":
    preprocessor = IoTDataPreprocessor()
    
    # Example with your small.csv
    X, y, num_classes, features = preprocessor.preprocess_data(
    "ML-EdgeIIoT-dataset.csv", 
    correlation_threshold=0.1,  # Higher threshold for large dataset
    multicollinearity_threshold=0.95,
    final_feature_count=25  # Optimal for 150K rows
)
    
    print(f"\nFinal Results:")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Selected features: {features}")