import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

    def preprocess_data(self, path):
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
                df[col].fillna(0, inplace=True)  # Numeric IoT features often use 0 for missing

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
            'http.referer', 'dns.qry.name'  # High-cardinality, dropped for simplicity
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

        # Feature engineering: Example for http.referer (if not dropped)
        # if 'http.referer' in df.columns:
        #     df['http.referer_present'] = df['http.referer'].notna().astype(int)
        #     df.drop('http.referer', axis=1, inplace=True)

        # Drop sparse features with higher threshold
        zero_dropped = []
        for col in df.columns:
            if col in self.protected_sparse_features or col in ['Attack_label', 'Attack_type']:
                continue
            try:
                if df[col].dtype in [np.float64, np.int64]:
                    zero_ratio = (df[col] == 0).sum() / len(df)
                    if zero_ratio > 0.85:  # Relaxed threshold
                        df.drop(col, axis=1, inplace=True)
                        zero_dropped.append((col, zero_ratio))
            except:
                continue
        if zero_dropped:
            print("\nZero-dominant columns dropped:")
            for col, ratio in zero_dropped:
                print(f"{col}: {ratio:.2%} zeros")
        print(f"Shape after zero-drop: {df.shape}")

        # Debug: Check column types before scaling
        print("\nColumn types before scaling:\n", df.dtypes)

        # Encode Attack_type
        le = LabelEncoder()
        df['Attack_type'] = le.fit_transform(df['Attack_type'])

        print("\n--- Class Distributions ---")
        print("Attack_label:\n", df['Attack_label'].value_counts())
        print("Attack_type:\n", df['Attack_type'].value_counts())

        # Targets
        y_binary = df.pop('Attack_label')
        y_multiclass = df.pop('Attack_type')

        # Ensure only numeric columns remain
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"\nWarning: Dropping non-numeric columns: {non_numeric.tolist()}")
            df.drop(columns=non_numeric, inplace=True)

        # Feature scaling
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df_scaled, y_multiclass,
            test_size=0.2,
            stratify=y_multiclass,
            random_state=42
        )

        # Feature selection using Random Forest
        print("\nPerforming feature selection...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        feature_importance = pd.Series(rf.feature_importances_, index=df.columns).sort_values(ascending=False)
        print("\nTop 10 features by importance:\n", feature_importance.head(10))
        top_features = feature_importance.head(40).index  # Keep top 40
        feature_indices = [df.columns.get_loc(col) for col in top_features]
        X_train = X_train[:, feature_indices]
        X_test = X_test[:, feature_indices]

        num_classes = len(np.unique(y_multiclass))
        print(f"\n✅ Preprocessing complete. Final shape: {X_train.shape}, Classes: {num_classes}")

        return X_train, X_test, y_train, y_test, y_multiclass, num_classes, top_features.tolist()