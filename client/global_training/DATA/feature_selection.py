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

class IoTFeatureSelector:
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
        
        # Categorical columns to encode (temporarily for correlation analysis)
        self.categorical_columns = [
            'http.request.method', 'dns.qry.qu', 'dns.qry.type',
            'mqtt.msg_decoded_as', 'mqtt.protoname'
        ]

    def save_raw_selected_features(self, original_df, selected_features, target_column='Attack_type'):
        """
        Save RAW columns (no preprocessing) with only selected features + target
        """
        # Ensure target column is included
        if target_column not in selected_features:
            final_columns = selected_features + [target_column]
        else:
            final_columns = selected_features
        
        # Get only existing columns from original dataframe
        existing_columns = [col for col in final_columns if col in original_df.columns]
        missing_columns = [col for col in final_columns if col not in original_df.columns]
        
        if missing_columns:
            print(f"⚠️  Missing columns (skipped): {missing_columns}")
        
        # Create new dataframe with RAW data (no preprocessing)
        raw_selected_df = original_df[existing_columns].copy()
        
        # Save to CSV
        output_path = "selected_features_dataset.csv"
        raw_selected_df.to_csv(output_path, index=False)
        
        print(f"\n✅ RAW feature-selected dataset saved as: {output_path}")
        print(f"   Columns saved: {len(existing_columns)}")
        print(f"   Rows: {len(raw_selected_df):,}")
        print(f"   Selected features: {[col for col in existing_columns if col != target_column]}")
        
        return output_path

    def analyze_correlations(self, df, target_col, threshold=0.1):
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

    def select_features_only(self, path, correlation_threshold=0.05,
                            multicollinearity_threshold=0.95,
                            final_feature_count=25):
        """
        Select features using temporary preprocessing but save RAW data
        """
        print(f"Found dataset: {path}\n{'='*70}")
        
        # Load original raw data
        original_df = pd.read_csv(path)
        target_column = 'Attack_type'
        
        print("Original DataFrame Info:\n" + "-"*30)
        print(f"Shape: {original_df.shape}")
        print(f"Missing Values: {original_df.isna().sum().sum()} ({original_df.isna().sum().sum() / original_df.size * 100:.2f}%)")
        print(f"Duplicate Rows: {original_df.duplicated().sum()}")
        
        # Create a COPY for temporary processing (don't modify original)
        df_temp = original_df.copy()
        
        # TEMPORARY PREPROCESSING FOR FEATURE SELECTION ONLY
        print(f"\n🔄 Temporary preprocessing for feature selection analysis...")
        
        # Handle missing values temporarily
        for col in df_temp.columns:
            if col in self.categorical_columns:
                df_temp[col].fillna(df_temp[col].mode()[0] if not df_temp[col].mode().empty else 'Unknown', inplace=True)
            else:
                df_temp[col].fillna(0, inplace=True)
        
        # Drop constant columns
        constant_dropped = []
        for col in df_temp.columns:
            if df_temp[col].nunique() == 1:
                df_temp.drop(col, axis=1, inplace=True)
                constant_dropped.append(col)
        
        if constant_dropped:
            print(f"Dropped constant columns: {constant_dropped}")
        
        # Drop irrelevant high-cardinality columns
        drop_columns = [
            'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
            'http.file_data', 'http.request.full_uri', 'http.request.version',
            'tcp.options', 'tcp.payload', 'mqtt.msg', 'mqtt.topic',
            'http.referer', 'dns.qry.name'
        ]
        
        dropped = [col for col in drop_columns if col in df_temp.columns]
        df_temp.drop(columns=dropped, inplace=True)
        if dropped:
            print(f"Dropped irrelevant columns: {dropped}")
        
        # Encode categorical columns TEMPORARILY
        le_dict = {}
        for col in self.categorical_columns:
            if col in df_temp.columns:
                le = LabelEncoder()
                df_temp[col] = le.fit_transform(df_temp[col].astype(str))
                le_dict[col] = le
        
        # Drop sparse features
        zero_dropped = []
        for col in df_temp.columns:
            if col in self.protected_sparse_features or col in ['Attack_label', 'Attack_type']:
                continue
            try:
                if df_temp[col].dtype in [np.float64, np.int64]:
                    zero_ratio = (df_temp[col] == 0).sum() / len(df_temp)
                    if zero_ratio > 0.85:
                        df_temp.drop(col, axis=1, inplace=True)
                        zero_dropped.append((col, zero_ratio))
            except:
                continue
        
        # Encode Attack_type temporarily
        le_target = LabelEncoder()
        df_temp['Attack_type'] = le_target.fit_transform(df_temp['Attack_type'])
        
        # Ensure all columns are numeric for correlation analysis
        feature_columns = [col for col in df_temp.columns if col not in ['Attack_label', 'Attack_type']]
        
        for col in feature_columns:
            if df_temp[col].dtype == 'object' or df_temp[col].dtype == 'string':
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
        
        # Remove any remaining non-numeric columns
        non_numeric_final = df_temp[feature_columns].select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_final) > 0:
            df_temp.drop(columns=non_numeric_final, inplace=True)
        
        print(f"Shape after temporary preprocessing: {df_temp.shape}")
        
        # FEATURE SELECTION ANALYSIS
        print(f"\n🎯 Starting feature selection analysis...")
        
        # Step 1: Find features correlated with target
        target_features, all_correlations = self.analyze_correlations(
            df_temp, 'Attack_type', threshold=correlation_threshold
        )
        
        # Step 2: Remove multicollinearity among selected features
        if len(target_features) > 1:
            final_features = self.remove_multicollinearity(
                df_temp, target_features, threshold=multicollinearity_threshold
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
        
        # Limit to exactly final_feature_count
        final_features = final_features[:final_feature_count]
        
        print(f"\n📋 Final selected features ({len(final_features)}):")
        for i, feat in enumerate(final_features, 1):
            corr_val = all_correlations.get(feat, 0)
            print(f"{i:2d}. {feat:<30}: {corr_val:.4f}")
        
        # SAVE RAW DATA with selected features
        print(f"\n💾 Saving RAW data with selected features...")
        output_path = self.save_raw_selected_features(original_df, final_features, target_column)
        
        print(f"\n✅ Feature selection complete!")
        print(f"Selected {len(final_features)} features from {original_df.shape[1]} total columns")
        print(f"Output file: {output_path}")
        
        return final_features, output_path

# Usage example
if __name__ == "__main__":
    selector = IoTFeatureSelector()
    
    # Select top 25 features and save RAW data
    selected_features, output_file = selector.select_features_only(
        "ML-EdgeIIoT-dataset.csv",
        correlation_threshold=0.1,  # Higher threshold for large dataset
        multicollinearity_threshold=0.95,
        final_feature_count=25  # Exactly 25 features
    )
    
    print(f"\nSelected features: {selected_features}")
    print(f"RAW data saved to: {output_file}")
