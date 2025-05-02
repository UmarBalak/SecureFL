import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

class IoTDataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.attack_type_map = {
            'Normal': 0, 'MITM': 1, 'Fingerprinting': 2, 'Ransomware': 3,
            'Uploading': 4, 'SQL_injection': 5, 'DDoS_HTTP': 6, 'DDoS_TCP': 7,
            'Password': 8, 'Port_Scanning': 9, 'Vulnerability_scanner': 10,
            'Backdoor': 11, 'XSS': 12, 'DDoS_UDP': 13, 'DDoS_ICMP': 14
        }
        self.inv_attack_map = {v: k for k, v in self.attack_type_map.items()}
        self.scaler = None
        self.imputer = None

        # Ensure directories exist
        dirs = ['models', 'data', 'logs', 'plots', 'federated_models']
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_data(self, file_path):
        """Load the dataset and prepare features and labels"""
        print("Loading dataset...")
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

            # Map attack types to numerical labels
            df['label'] = df['Attack_type'].map(self.attack_type_map)

            # Extract features to keep
            features_to_keep = [
                'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'arp.opcode',
                'icmp.checksum', 'icmp.seq_le', 'icmp.transmit_timestamp',
                'http.file_data', 'http.content_length', 'http.request.uri.query', 'http.request.method',
                'http.referer', 'http.request.full_uri', 'http.request.version', 'http.response',
                'tcp.options', 'tcp.payload', 'tcp.srcport', 'tcp.flags', 'tcp.flags.ack',
                'tcp.connection.syn', 'tcp.connection.rst', 'tcp.connection.fin',
                'udp.time_delta',
                'dns.qry.name', 'dns.qry.name.len', 'dns.qry.qu', 'dns.qry.type', 'dns.retransmission',
                'mqtt.protoname', 'mqtt.topic', 'mqtt.conack.flags', 'mqtt.msg', 'mqtt.len',
                'mqtt.msgtype', 'mqtt.hdrflags',
                'frame.time'
            ]

            # Filter features
            X = df[features_to_keep]
            y = df['label']

            # Convert to numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            print(f"Features prepared: {X.shape[1]} features selected")
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self, X, y, max_samples=None, test_size=0.2):
        """Preprocess the data with advanced feature engineering and handling missing values"""
        print("\nPreprocessing data...")

        # Limit samples if specified
        if max_samples and max_samples < len(X):
            X, _, y, _ = train_test_split(X, y, train_size=max_samples,
                                        random_state=self.random_state, stratify=y)
            print(f"Using limited dataset: {max_samples} samples")
        else:
            print(f"Using full dataset: {len(X)} samples")

        # Handle missing values
        print("Handling missing values...")
        self.imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        # # Remove any remaining NaN rows
        # mask = ~X.isna().any(axis=1)
        # X = X[mask]
        # y = y[mask]
        # Remove any remaining NaN rows and align the indices of X and y
        mask = ~X.isna().any(axis=1)
        X = X[mask].reset_index(drop=True)  # Reset index for X
        y = y[mask.values].reset_index(drop=True)  # Filter y using mask values and reset index

        # Add advanced features
        X = self.add_advanced_features(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y)

        # Reset indices
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to DataFrame for future use
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Fill any remaining NaN values
        X_train_scaled = X_train_scaled.fillna(0)
        X_test_scaled = X_test_scaled.fillna(0)

        # Balance the training data using SMOTE
        print("\nBalancing dataset with SMOTE...")
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        # Convert labels to one-hot encoding for neural network
        num_classes = len(self.attack_type_map)
        y_train_categorical = to_categorical(y_train_balanced, num_classes=num_classes)
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)

        print(f"Preprocessing complete. Training set: {X_train_balanced.shape}, Test set: {X_test_scaled.shape}")

        # Save preprocessing components
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.imputer, 'models/imputer.joblib')

        preprocessing_stats = {
            'training_samples': X_train_balanced.shape[0],
            'test_samples': X_test_scaled.shape[0],
            'features': X_train_balanced.shape[1],
            'classes': num_classes,
            'class_distribution_before_smote': y_train.value_counts().to_dict(),
            'class_distribution_after_smote': pd.Series(y_train_balanced).value_counts().to_dict()
        }

        return (X_train_balanced, X_test_scaled, y_train_balanced, y_test,
                y_train_categorical, y_test_categorical, preprocessing_stats)

    def add_advanced_features(self, X):
        """Engineer additional features for improved model performance"""
        print("Engineering advanced features...")

        # Time-based features
        if 'udp.time_delta' in X.columns:
            X['udp_time_stats'] = X['udp.time_delta'].rolling(window=5, min_periods=1).std()
            X['udp_time_mean'] = X['udp.time_delta'].rolling(window=5, min_periods=1).mean()
            X['udp_time_max'] = X['udp.time_delta'].rolling(window=5, min_periods=1).max()

            # Additional advanced features
            X['udp_time_entropy'] = X['udp.time_delta'].rolling(window=10, min_periods=1).apply(
                lambda x: -np.sum(np.square(x/x.sum()) * np.log(x/x.sum() + 1e-10))
            )

        # DNS features
        if 'dns.qry.name.len' in X.columns:
            X['dns_name_length_ratio'] = X['dns.qry.name.len'] / (X['dns.qry.name.len'].mean() + 1)
            X['dns_length_normalized'] = X['dns.qry.name.len'] / X['dns.qry.name.len'].max()

            # Detect anomalous DNS query lengths (potential data exfiltration)
            X['dns_anomaly_score'] = np.abs(X['dns.qry.name.len'] - X['dns.qry.name.len'].mean()) / X['dns.qry.name.len'].std()

        # MQTT features
        if 'mqtt.msg' in X.columns:
            X['mqtt.msg'] = X['mqtt.msg'].fillna('').astype(str)
            X['mqtt_msg_density'] = X['mqtt.msg'].apply(len) / (X['mqtt.msg'].str.len().mean() + 1)

            # Extract numeric-only features from mqtt fields
            if 'mqtt.len' in X.columns:
                X['mqtt_len_normalized'] = X['mqtt.len'] / (X['mqtt.len'].max() + 1)

        # TCP connection features
        tcp_columns = [col for col in X.columns if col.startswith('tcp')]
        if len(tcp_columns) > 0:
            if 'tcp.flags' in X.columns and 'tcp.srcport' in X.columns:
                X['tcp_port_flag_ratio'] = X['tcp.flags'] / (X['tcp.srcport'] + 1)

            # Count TCP features that are non-zero (as proxy for connection complexity)
            X['tcp_feature_count'] = X[tcp_columns].fillna(0).astype(bool).sum(axis=1)

        # Fill remaining missing values
        X = X.fillna(0)

        print(f"Added {X.shape[1]} features after engineering")
        return X