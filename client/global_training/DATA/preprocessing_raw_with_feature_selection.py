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

    def preprocess_data(self, path, apply_smote):
        print(f"Found dataset: {path}\n{'='*70}")
        df = pd.read_csv(path)

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

        # Encode categorical columns
        print("\nEncoding categorical columns...")
        le_dict = {}
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
                print(f"Encoded {col} with {len(le.classes_)} unique values")

        # Encode Attack_type
        le = LabelEncoder()
        df['Attack_type'] = le.fit_transform(df['Attack_type'])

        # Ensure all columns are numeric before correlation analysis
        print("\nEnsuring all features are numeric for correlation analysis...")
        feature_columns = [col for col in df.columns if col not in ['Attack_type']]
        
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
        
        # Prepare data for processing
        feature_df = df[feature_columns].copy()
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

        #####################################################################################
        
        if apply_smote:
            # Optional: Apply SMOTE for class balancing
            print("\nApplying SMOTE for data augmentation...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y_multiclass)
            print("\n--- Class Distributions After SMOTE Augmentation ---")
        else:
            X_resampled, y_resampled = X_scaled, y_multiclass

        #####################################################################################
        
        unique, counts = np.unique(y_resampled, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples")

        num_classes = len(np.unique(y_resampled))
        print(f"\n✅ Preprocessing complete!")
        print(f"Final shape: {X_resampled.shape}")
        print(f"Number of classes: {num_classes}")

        return X_resampled, y_resampled, num_classes


# # Usage example
# if __name__ == "__main__":
#     preprocessor = IoTDataPreprocessor()
    
#     # Example with your small.csv
#     X, y, num_classes, features = preprocessor.preprocess_data(
#     "DATA/ML-EdgeIIoT-dataset.csv", 
#     correlation_threshold=0.1,  # Higher threshold for large dataset
#     multicollinearity_threshold=0.95,
#     final_feature_count=25  # Optimal for 150K rows
# )
    
#     print(f"\nFinal Results:")
#     print(f"Features shape: {X.shape}")
#     print(f"Labels shape: {y.shape}")
#     print(f"Selected features: {features}")