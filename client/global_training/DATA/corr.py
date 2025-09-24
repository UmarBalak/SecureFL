import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureCorrelationAnalyzer:
    def __init__(self):
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
        self.categorical_columns = [
            'http.request.method', 'dns.qry.qu', 'dns.qry.type',
            'mqtt.msg_decoded_as', 'mqtt.protoname'
        ]

    def preprocess_for_correlation(self, path):
        """Preprocess data specifically for correlation analysis"""
        print(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        
        print(f"Original shape: {df.shape}")
        
        # Handle missing values
        for col in df.columns:
            if col in self.categorical_columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            else:
                df[col].fillna(0, inplace=True)
        
        # Drop constant columns
        constant_dropped = []
        for col in df.columns:
            if df[col].nunique() == 1:
                df.drop(col, axis=1, inplace=True)
                constant_dropped.append(col)
        
        # Drop irrelevant columns
        drop_columns = [
            'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
            'http.file_data', 'http.request.full_uri', 'http.request.version',
            'tcp.options', 'tcp.payload', 'mqtt.msg', 'mqtt.topic',
            'http.referer', 'dns.qry.name'
        ]
        dropped = [col for col in drop_columns if col in df.columns]
        df.drop(columns=dropped, inplace=True)
        
        # Encode categorical columns
        le_dict = {}
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
        
        # Encode Attack_type
        if 'Attack_type' in df.columns:
            le = LabelEncoder()
            df['Attack_type'] = le.fit_transform(df['Attack_type'])
        
        # Ensure only numeric columns remain
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Dropping non-numeric columns: {non_numeric.tolist()}")
            df.drop(columns=non_numeric, inplace=True)
        
        print(f"Shape after preprocessing: {df.shape}")
        return df

    def analyze_correlations(self, df, correlation_threshold=0.8):
        """Analyze feature correlations and identify highly correlated pairs"""
        
        # Calculate correlation matrix
        print("Calculating correlation matrix...")
        correlation_matrix = df.corr()
        
        # Find highly correlated feature pairs
        highly_correlated_pairs = []
        upper_triangle = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > correlation_threshold:
                    highly_correlated_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Sort by correlation strength
        highly_correlated_pairs = sorted(highly_correlated_pairs, 
                                       key=lambda x: x['correlation'], 
                                       reverse=True)
        
        return correlation_matrix, highly_correlated_pairs

    def plot_correlation_heatmap(self, correlation_matrix, figsize=(15, 12)):
        """Create correlation heatmap"""
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(correlation_matrix))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=False,  # Set to True if you want to see correlation values
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.1,
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix Heatmap', fontsize=16, pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_high_correlation_pairs(self, highly_correlated_pairs, top_n=20):
        """Plot top highly correlated feature pairs"""
        if not highly_correlated_pairs:
            print("No highly correlated pairs found!")
            return
        
        top_pairs = highly_correlated_pairs[:top_n]
        
        # Create feature pair labels
        pair_labels = [f"{pair['feature1']}\nvs\n{pair['feature2']}" 
                      for pair in top_pairs]
        correlations = [pair['correlation'] for pair in top_pairs]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(correlations)), correlations, 
                      color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(correlations))))
        
        plt.title(f'Top {len(top_pairs)} Highly Correlated Feature Pairs', fontsize=16, pad=20)
        plt.xlabel('Feature Pairs', fontsize=12)
        plt.ylabel('Absolute Correlation', fontsize=12)
        plt.xticks(range(len(correlations)), 
                  [f"{pair['feature1']} vs {pair['feature2']}" for pair in top_pairs],
                  rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def suggest_features_to_remove(self, highly_correlated_pairs, df, target_columns=['Attack_label', 'Attack_type']):
        """Suggest which features to remove based on correlation analysis"""
        features_to_remove = set()
        feature_importance_with_target = {}
        
        # Calculate correlation with target variables
        for target in target_columns:
            if target in df.columns:
                target_correlations = df.corr()[target].abs().sort_values(ascending=False)
                for feature, corr in target_correlations.items():
                    if feature != target:
                        if feature not in feature_importance_with_target:
                            feature_importance_with_target[feature] = 0
                        feature_importance_with_target[feature] += corr
        
        print("Features suggested for removal due to high correlation:")
        print("="*60)
        
        for pair in highly_correlated_pairs:
            feature1, feature2 = pair['feature1'], pair['feature2']
            corr = pair['correlation']
            
            # Skip if one is a target variable
            if feature1 in target_columns or feature2 in target_columns:
                continue
            
            # Decide which feature to remove based on target correlation
            f1_target_importance = feature_importance_with_target.get(feature1, 0)
            f2_target_importance = feature_importance_with_target.get(feature2, 0)
            
            if f1_target_importance > f2_target_importance:
                to_remove = feature2
                to_keep = feature1
            else:
                to_remove = feature1
                to_keep = feature2
            
            features_to_remove.add(to_remove)
            print(f"Remove '{to_remove}' (keep '{to_keep}') - Correlation: {corr:.3f}")
        
        print(f"\nTotal features suggested for removal: {len(features_to_remove)}")
        print(f"Remaining features after removal: {df.shape[1] - len(features_to_remove)}")
        
        return list(features_to_remove)

    def distribution_analysis(self, df):
        """Analyze feature distributions"""
        # Calculate zero ratios
        zero_ratios = {}
        for col in df.columns:
            if col not in ['Attack_label', 'Attack_type']:
                try:
                    if df[col].dtype in [np.float64, np.int64]:
                        zero_ratio = (df[col] == 0).sum() / len(df)
                        zero_ratios[col] = zero_ratio
                except:
                    continue
        
        # Plot zero ratios
        zero_ratios_sorted = dict(sorted(zero_ratios.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(15, 8))
        features = list(zero_ratios_sorted.keys())[:30]  # Top 30 most sparse
        ratios = [zero_ratios_sorted[f] for f in features]
        
        bars = plt.bar(range(len(features)), ratios, 
                      color=plt.cm.Reds(np.linspace(0.3, 0.9, len(features))))
        
        plt.title('Top 30 Features by Zero Ratio (Sparsity)', fontsize=16, pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Zero Ratio', fontsize=12)
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add threshold line
        plt.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='85% threshold')
        plt.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return zero_ratios_sorted

# Usage example:
def run_correlation_analysis(csv_path, correlation_threshold=0.8):
    """Run complete correlation analysis"""
    analyzer = FeatureCorrelationAnalyzer()
    
    # Preprocess data
    df = analyzer.preprocess_for_correlation(csv_path)
    
    # Analyze correlations
    corr_matrix, high_corr_pairs = analyzer.analyze_correlations(df, correlation_threshold)
    
    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (threshold: {correlation_threshold})")
    
    # Create visualizations
    print("\n1. Creating correlation heatmap...")
    analyzer.plot_correlation_heatmap(corr_matrix)
    
    print("\n2. Plotting highly correlated pairs...")
    analyzer.plot_high_correlation_pairs(high_corr_pairs)
    
    print("\n3. Analyzing feature sparsity...")
    zero_ratios = analyzer.distribution_analysis(df)
    
    # Suggest features to remove
    print("\n4. Suggesting features for removal...")
    features_to_remove = analyzer.suggest_features_to_remove(high_corr_pairs, df)
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'features_to_remove': features_to_remove,
        'zero_ratios': zero_ratios,
        'processed_df': df
    }

# Run the analysis
if __name__ == "__main__":
    # Replace with your CSV path
    csv_path = "DATA/ML-EdgeIIoT-dataset.csv"  # Update this path
    
    results = run_correlation_analysis(csv_path, correlation_threshold=0.8)
    
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total features analyzed: {results['processed_df'].shape[1]}")
    print(f"Highly correlated pairs found: {len(results['high_correlation_pairs'])}")
    print(f"Features suggested for removal: {len(results['features_to_remove'])}")
    print(f"Final recommended feature count: {results['processed_df'].shape[1] - len(results['features_to_remove'])}")