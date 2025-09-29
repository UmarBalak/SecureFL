import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings("ignore")

class IoTDataPreprocessor:
    """
    Fixed preprocessing with numerical stability for test evaluation
    """
    
    def __init__(self, global_class_names=None, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        self.global_class_names = global_class_names or [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ]
        
        self.global_le = LabelEncoder()
        self.global_le.fit(self.global_class_names)
        
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self._preprocessor_file = os.path.join(self.artifacts_dir, "preprocessor.pkl")
        self._global_le_file = os.path.join(self.artifacts_dir, "global_label_encoder.pkl")
    
    def _detect_feature_types(self, df, target_col='Attack_type'):
        """
        Stable feature detection without hashing to prevent numerical issues
        """
        print("Detecting features with numerical stability...")
        
        feature_columns = [col for col in df.columns if col != target_col]
        numeric_features = []
        categorical_features = []
        
        for col in feature_columns:
            col_data = df[col]
            is_numeric_dtype = pd.api.types.is_numeric_dtype(col_data)
            cardinality = col_data.nunique()
            
            if is_numeric_dtype:
                numeric_features.append(col)
                print(f"Column {col}: NUMERIC (dtype: {col_data.dtype})")
            else:
                if cardinality <= 50:  # Increased threshold for safety
                    categorical_features.append(col)
                    print(f"Column {col}: CATEGORICAL (cardinality: {cardinality})")
                else:
                    # Drop high cardinality categorical features instead of hashing
                    print(f"Column {col}: DROPPED (high cardinality: {cardinality})")
                    df = df.drop(columns=[col])
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        print(f"\nFinal stable features:")
        print(f" Numeric: {len(numeric_features)} features")
        print(f" Categorical: {len(categorical_features)} features")
        
        return df
    
    def fit_preprocessor(self, train_path, target_num_classes=15):
        """
        Stable preprocessing pipeline that prevents numerical issues in test evaluation
        """
        print(f"Loading dataset with stable preprocessing: {train_path}")
        df = pd.read_csv(train_path, low_memory=False)
        
        if 'Attack_type' not in df.columns:
            raise ValueError("Attack_type column not found")
        
        print(f"Original dataset shape: {df.shape}")
        
        # Stable feature detection
        df = self._detect_feature_types(df, target_col='Attack_type')
        
        # Handle missing values with stability
        print("\nHandling missing values with numerical stability...")
        
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                missing_count = df[col].isna().sum()
                
                if missing_count > 0:
                    # Use median for stability
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f" {col}: {missing_count} missing -> filled with median {median_val:.4f}")
                
                # Clip extreme outliers for numerical stability
                Q1 = df[col].quantile(0.01)  # More aggressive clipping
                Q99 = df[col].quantile(0.99)
                
                if Q99 > Q1:
                    outliers_before = ((df[col] < Q1) | (df[col] > Q99)).sum()
                    df[col] = df[col].clip(lower=Q1, upper=Q99)
                    if outliers_before > 0:
                        print(f" {col}: {outliers_before} outliers clipped to [{Q1:.4f}, {Q99:.4f}]")
        
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                missing_count = df[col].isin(['nan', 'None', '']).sum()
                if missing_count > 0:
                    df[col] = df[col].replace(['nan', 'None', ''], 'Unknown')
                    print(f" {col}: {missing_count} missing -> filled with 'Unknown'")
        
        # Build stable preprocessing pipeline
        feature_columns = self.numeric_features + self.categorical_features
        X = df[feature_columns].copy()
        
        print(f"\nBuilding numerically stable preprocessing pipeline...")
        print(f"Features to process: {len(feature_columns)}")
        
        transformers = []
        
        # Use StandardScaler only for numerical stability
        if self.numeric_features:
            print(f" Numeric pipeline: StandardScaler for {len(self.numeric_features)} features")
            num_pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Single stable scaler
            ])
            transformers.append(('num', num_pipeline, self.numeric_features))
        
        # Categorical with strict limits
        if self.categorical_features:
            print(f" Categorical pipeline: OneHotEncoder for {len(self.categorical_features)} features")
            from sklearn.preprocessing import OneHotEncoder
            transformers.append(('cat', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                max_categories=20,  # Strict limit
                drop='if_binary'    # Prevent redundant features
            ), self.categorical_features))
        
        self.preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        # Fit and transform with error handling
        try:
            X_processed = self.preprocessor.fit_transform(X)
            print(f"Processed shape: {X_processed.shape}")
            
            # Check for any NaN or inf values
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                print("WARNING: NaN or Inf values detected in processed features!")
                # Replace NaN/Inf with 0
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
                print("Replaced NaN/Inf values with 0.0")
            
            print(f"Feature range: [{X_processed.min():.4f}, {X_processed.max():.4f}]")
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise
        
        # Process target with validation
        print(f"\nProcessing target variable...")
        df['Attack_type'] = df['Attack_type'].apply(
            lambda x: x if x in self.global_class_names else 'Normal'
        )
        
        y = self.global_le.transform(df['Attack_type'].values)
        unique_classes = np.unique(y)
        print(f"Target classes found: {len(unique_classes)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Save artifacts
        print(f"\nSaving stable preprocessing artifacts...")
        joblib.dump(self.preprocessor, self._preprocessor_file)
        joblib.dump(self.global_le, self._global_le_file)
        
        feature_info = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        joblib.dump(feature_info, os.path.join(self.artifacts_dir, "feature_info.pkl"))
        
        artifacts_dict = {
            "preprocessor": self.preprocessor,
            "global_label_encoder": self.global_le,
            "feature_info": feature_info
        }
        
        print(f"Stable preprocessing complete!")
        print(f"Final shape: {X_processed.shape}")
        
        return X_processed, y, len(unique_classes), artifacts_dict
    
    def transform(self, path, target_num_classes=15):
        """
        Stable transform for test data
        """
        print(f"Transforming dataset with stable preprocessing: {path}")
        df = pd.read_csv(path, low_memory=False)
        
        if 'Attack_type' not in df.columns:
            raise ValueError("Attack_type column not found")
        
        # Load artifacts
        if self.preprocessor is None:
            self.preprocessor = joblib.load(self._preprocessor_file)
            
        if not hasattr(self, 'global_le') or self.global_le is None:
            self.global_le = joblib.load(self._global_le_file)
            
        # Load feature info
        feature_info = joblib.load(os.path.join(self.artifacts_dir, "feature_info.pkl"))
        self.numeric_features = feature_info['numeric_features']
        self.categorical_features = feature_info['categorical_features']
        
        # Apply same preprocessing logic as training
        print("Applying consistent preprocessing to test data...")
        
        # Handle missing values consistently
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'None', ''], 'Unknown')
        
        # Extract features
        feature_columns = self.numeric_features + self.categorical_features
        X = df[feature_columns].copy()
        
        # Apply preprocessing with error handling
        try:
            X_processed = self.preprocessor.transform(X)
            
            # Ensure no NaN or inf values
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                print("WARNING: NaN or Inf detected in test preprocessing!")
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
                print("Replaced NaN/Inf values with 0.0")
                
        except Exception as e:
            print(f"Error in test preprocessing: {e}")
            raise
        
        # Process target
        df['Attack_type'] = df['Attack_type'].apply(
            lambda x: x if x in self.global_class_names else 'Normal'
        )
        
        y = self.global_le.transform(df['Attack_type'].values)
        actual_num_classes = len(np.unique(y))
        
        print(f"Test transform complete. Shape: {X_processed.shape}, Classes: {actual_num_classes}")
        print(f"Feature range: [{X_processed.min():.4f}, {X_processed.max():.4f}]")
        
        return X_processed, y, actual_num_classes
