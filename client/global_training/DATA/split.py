import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random

def save_fl_splits(split_result, output_dir="./DATA/"):
    """Save all splits with comprehensive metadata"""
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save global sets
    global_train_path = os.path.join(output_dir, "global_train.csv")
    global_test_path = os.path.join(output_dir, "global_test.csv")
    
    split_result["global_train"].to_csv(global_train_path, index=False)
    split_result["global_test"].to_csv(global_test_path, index=False)
    
    print(f"Saved global training set: {global_train_path}")
    print(f"Saved global test set: {global_test_path}")
    
    # Save client data
    client_info = {}
    for cid, client_df in split_result["clients"].items():
        client_path = os.path.join(output_dir, f"client_{cid+1}.csv")
        client_df.to_csv(client_path, index=False)
        
        if len(client_df) > 0:
            client_classes = client_df.iloc[:, -1].value_counts().to_dict()  # Assuming last column is label
            client_info[f"client_{cid+1}"] = {
                "path": client_path,
                "samples": len(client_df),
                "classes": len(client_classes),
                "class_distribution": client_classes
            }
        else:
            client_info[f"client_{cid+1}"] = {
                "path": client_path,
                "samples": 0,
                "classes": 0,
                "class_distribution": {}
            }
    
    # Save metadata
    metadata = {
        "split_info": split_result["split_info"],
        "client_info": client_info,
        "global_sets": {
            "train_path": global_train_path,
            "train_samples": len(split_result["global_train"]),
            "test_path": global_test_path,
            "test_samples": len(split_result["global_test"])
        }
    }
    
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Saved split metadata: {metadata_path}")
    
    return metadata

def visualize_split_distribution(split_result, label_col="Attack_type", save_path="split_distribution.png"):
    """Create visualization of data distribution across splits"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Federated Learning Data Split Distribution", fontsize=16)
    
    # Global train distribution
    train_counts = split_result["global_train"][label_col].value_counts()
    axes[0, 0].pie(train_counts.values, labels=train_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title(f"Global Train Set\n({len(split_result['global_train'])} samples)")
    
    # Global test distribution  
    test_counts = split_result["global_test"][label_col].value_counts()
    axes[0, 1].pie(test_counts.values, labels=test_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title(f"Global Test Set\n({len(split_result['global_test'])} samples)")
    
    # Client distribution overview
    client_samples = [len(df) for df in split_result["clients"].values()]
    client_labels = [f"Client {i+1}" for i in range(len(client_samples))]
    
    axes[1, 0].bar(client_labels, client_samples)
    axes[1, 0].set_title("Samples per Client")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Client class diversity
    client_diversity = [df[label_col].nunique() if len(df) > 0 else 0 for df in split_result["clients"].values()]
    axes[1, 1].bar(client_labels, client_diversity)
    axes[1, 1].set_title("Unique Classes per Client")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution visualization: {save_path}")
    
    return fig

def analyze_class_distribution(df, label_col):
    """Analyze and report class distribution"""
    class_counts = df[label_col].value_counts()
    class_props = df[label_col].value_counts(normalize=True)
    
    print("\nClass Distribution Analysis:")
    print("-" * 50)
    for class_name in class_counts.index:
        count = class_counts[class_name]
        prop = class_props[class_name] * 100
        print(f"{class_name:<20}: {count:>8} ({prop:>6.2f}%)")
    
    return class_counts, class_props

def enhanced_fl_split_with_class_fluctuation(
    input_csv: str,
    label_col: str,
    global_train_pct: float = 0.4,
    global_test_pct: float = 0.15,
    n_clients: int = 10,
    alpha: float = 1.0,
    min_samples_per_class: int = 50,
    seed: int = 42,
    ensure_all_classes_in_global: bool = True,
    # NEW: Class distribution control parameters
    class_distribution_type: str = "variable",  # "fixed", "variable", "extreme"
    min_classes_per_client: int = 2,           # Minimum classes each client must have
    max_classes_per_client: int = 8,           # Maximum classes each client can have
    dominant_class_prob: float = 0.7,          # Probability that dominant class takes majority of samples
    rare_class_clients: int = 2,               # Number of clients that get rare classes exclusively
):
    """
    Enhanced FL split with fine-grained control over class distribution per client
    
    Parameters:
    -----------
    class_distribution_type: str
        - "fixed": All clients get same number of classes
        - "variable": Clients get different numbers of classes (your request!)
        - "extreme": Some clients very specialized, others diverse
    
    min_classes_per_client: int
        Minimum number of classes each client must have
    
    max_classes_per_client: int  
        Maximum number of classes each client can have
        
    dominant_class_prob: float
        Probability that one class dominates a client's data
        
    rare_class_clients: int
        Number of clients that specialize in rare attack types
    """
    
    print(f"🎯 Enhanced FL Split with Class Fluctuation Control")
    print(f"   Distribution Type: {class_distribution_type}")
    print(f"   Classes per client: {min_classes_per_client}-{max_classes_per_client}")
    print(f"   Dominant class probability: {dominant_class_prob}")
    print(f"   Rare class specialists: {rare_class_clients}")
    
    # Your existing excellent global split logic
    df = pd.read_csv(input_csv, low_memory=False)
    original_counts, original_props = analyze_class_distribution(df, label_col)
    np.random.seed(seed)
    random.seed(seed)
    
    # [Keep your existing global split logic exactly as is - it's perfect!]
    if ensure_all_classes_in_global:
        print(f"\nEnsuring minimum {min_samples_per_class} samples per class in global sets...")
        
        minority_classes = []
        majority_classes = []
        
        for class_name, count in original_counts.items():
            if count < min_samples_per_class * 10:
                minority_classes.append(class_name)
            else:
                majority_classes.append(class_name)
        
        print(f"Minority classes: {minority_classes}")
        print(f"Majority classes: {majority_classes}")
        
        # Your existing logic for handling minority/majority classes
        minority_dfs = []
        majority_df_parts = []
        
        for class_name in minority_classes:
            class_df = df[df[label_col] == class_name].copy()
            
            if len(class_df) < min_samples_per_class * 3:
                global_train_size = max(int(len(class_df) * 0.6), min_samples_per_class // 2)
                global_test_size = max(int(len(class_df) * 0.2), min_samples_per_class // 4)
            else:
                global_train_size = max(int(len(class_df) * 0.4), min_samples_per_class)
                global_test_size = max(int(len(class_df) * 0.2), min_samples_per_class // 2)
            
            class_train, class_temp = train_test_split(
                class_df, train_size=global_train_size, random_state=seed, shuffle=True
            )
            
            if len(class_temp) > global_test_size:
                class_test, class_client = train_test_split(
                    class_temp, train_size=global_test_size, random_state=seed, shuffle=True
                )
            else:
                class_test = class_temp
                class_client = pd.DataFrame(columns=class_df.columns)
                
            minority_dfs.append({
                'class': class_name,
                'train': class_train,
                'test': class_test,
                'client': class_client
            })
        
        for class_name in majority_classes:
            class_df = df[df[label_col] == class_name].copy()
            
            class_train, class_temp = train_test_split(
                class_df, train_size=global_train_pct, random_state=seed, shuffle=True
            )
            
            test_size_adj = global_test_pct / (1 - global_train_pct)
            class_test, class_client = train_test_split(
                class_temp, train_size=test_size_adj, random_state=seed, shuffle=True
            )
            
            majority_df_parts.append({
                'class': class_name,
                'train': class_train, 
                'test': class_test,
                'client': class_client
            })
        
        all_class_parts = minority_dfs + majority_df_parts
        global_train_parts = [part['train'] for part in all_class_parts]
        global_test_parts = [part['test'] for part in all_class_parts]  
        client_pool_parts = [part['client'] for part in all_class_parts if not part['client'].empty]
        
        df_global_train = pd.concat(global_train_parts, ignore_index=True).sample(frac=1, random_state=seed)
        df_global_test = pd.concat(global_test_parts, ignore_index=True).sample(frac=1, random_state=seed)
        
        if client_pool_parts:
            df_client_pool = pd.concat(client_pool_parts, ignore_index=True).sample(frac=1, random_state=seed)
        else:
            df_client_pool = pd.DataFrame(columns=df.columns)
    
    else:
        # Fallback logic (your existing code)
        df_global_train, df_temp = train_test_split(
            df, train_size=global_train_pct, stratify=df[label_col], random_state=seed
        )
        
        test_size_adj = global_test_pct / (1 - global_train_pct)
        df_global_test, df_client_pool = train_test_split(
            df_temp, train_size=test_size_adj, stratify=df_temp[label_col], random_state=seed
        )
    
    # Analyze global distributions
    print(f"\nGlobal Training Set ({len(df_global_train)} samples):")
    analyze_class_distribution(df_global_train, label_col)
    
    print(f"\nGlobal Test Set ({len(df_global_test)} samples):")
    analyze_class_distribution(df_global_test, label_col)
    
    # =====================================
    # NEW: Enhanced Non-IID Client Distribution with Class Fluctuation
    # =====================================
    print(f"\n🎯 Distributing {len(df_client_pool)} samples with class fluctuation control...")
    
    if len(df_client_pool) == 0:
        print("Warning: No samples left for client distribution!")
        client_data = {i: pd.DataFrame(columns=df.columns) for i in range(n_clients)}
    else:
        available_classes = list(df_client_pool[label_col].unique())
        n_available_classes = len(available_classes)
        
        print(f"Available classes for clients: {available_classes}")
        
        # Step 1: Determine class allocation strategy per client
        client_class_allocations = []
        
        if class_distribution_type == "fixed":
            # All clients get same number of classes
            classes_per_client = min(max_classes_per_client, n_available_classes)
            for cid in range(n_clients):
                allocated_classes = random.sample(available_classes, classes_per_client)
                client_class_allocations.append({
                    'client_id': cid,
                    'classes': allocated_classes,
                    'dominant_class': random.choice(allocated_classes),
                    'is_specialist': False
                })
                
        elif class_distribution_type == "variable":
            # Each client gets different number of classes (YOUR REQUEST!)
            for cid in range(n_clients):
                # Randomly choose number of classes for this client
                n_classes_for_client = random.randint(min_classes_per_client, 
                                                    min(max_classes_per_client, n_available_classes))
                
                # Randomly select which classes this client gets
                allocated_classes = random.sample(available_classes, n_classes_for_client)
                
                # Choose dominant class (may take majority of samples)
                dominant_class = random.choice(allocated_classes)
                
                client_class_allocations.append({
                    'client_id': cid,
                    'classes': allocated_classes,
                    'dominant_class': dominant_class,
                    'is_specialist': len(allocated_classes) <= 3
                })
                
        elif class_distribution_type == "extreme":
            # Some clients are specialists, others are diverse
            specialist_clients = random.sample(range(n_clients), rare_class_clients)
            
            for cid in range(n_clients):
                if cid in specialist_clients:
                    # Specialist client - only 1-2 classes, focus on rare ones
                    rare_classes = [cls for cls in minority_classes if cls in available_classes]
                    if rare_classes:
                        n_classes_for_client = random.randint(1, min(2, len(rare_classes)))
                        allocated_classes = random.sample(rare_classes, n_classes_for_client)
                    else:
                        allocated_classes = random.sample(available_classes, min(2, len(available_classes)))
                    
                    client_class_allocations.append({
                        'client_id': cid,
                        'classes': allocated_classes, 
                        'dominant_class': allocated_classes[0],
                        'is_specialist': True
                    })
                else:
                    # Diverse client - many classes
                    n_classes_for_client = random.randint(
                        max(min_classes_per_client, n_available_classes//2), 
                        min(max_classes_per_client, n_available_classes)
                    )
                    allocated_classes = random.sample(available_classes, n_classes_for_client)
                    
                    client_class_allocations.append({
                        'client_id': cid,
                        'classes': allocated_classes,
                        'dominant_class': random.choice(allocated_classes),
                        'is_specialist': False
                    })
        
        # Step 2: Distribute samples according to class allocations
        client_data = {i: [] for i in range(n_clients)}
        
        # Track which clients get each class
        class_to_clients = defaultdict(list)
        for allocation in client_class_allocations:
            for cls in allocation['classes']:
                class_to_clients[cls].append(allocation)
        
        # Distribute each class among its assigned clients
        for class_name in available_classes:
            class_data = df_client_pool[df_client_pool[label_col] == class_name]
            n_samples = len(class_data)
            
            if n_samples == 0:
                continue
                
            assigned_clients = class_to_clients[class_name]
            if not assigned_clients:
                continue
            
            # Create distribution weights based on dominant class preferences
            weights = []
            for allocation in assigned_clients:
                if allocation['dominant_class'] == class_name:
                    # This client prefers this class
                    weights.append(dominant_class_prob)
                else:
                    # This client gets some of this class but it's not dominant
                    weights.append((1 - dominant_class_prob) / (len(allocation['classes']) - 1))
            
            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(assigned_clients)) / len(assigned_clients)
            
            # Convert to sample counts
            sample_counts = (weights * n_samples).astype(int)
            
            # Handle rounding
            while sample_counts.sum() < n_samples:
                sample_counts[np.random.randint(0, len(sample_counts))] += 1
            while sample_counts.sum() > n_samples:
                idx = np.random.choice(np.where(sample_counts > 0)[0])
                sample_counts[idx] -= 1
            
            # Distribute samples
            start_idx = 0
            for i, allocation in enumerate(assigned_clients):
                if sample_counts[i] > 0:
                    end_idx = start_idx + sample_counts[i]
                    client_samples = class_data.iloc[start_idx:end_idx]
                    client_data[allocation['client_id']].append(client_samples)
                    start_idx = end_idx
        
        # Combine and shuffle client data
        for cid in client_data.keys():
            if client_data[cid]:
                client_data[cid] = pd.concat(client_data[cid]).sample(frac=1, random_state=seed).reset_index(drop=True)
            else:
                client_data[cid] = pd.DataFrame(columns=df.columns)
    
    # Step 3: Enhanced client distribution analysis
    print(f"\n🎯 Client Data Distribution with Class Fluctuation:")
    print("-" * 80)
    
    total_client_samples = 0
    client_stats = []
    
    for cid, client_df in client_data.items():
        if len(client_df) > 0:
            unique_classes = client_df[label_col].nunique()
            total_samples = len(client_df)
            
            # Get class distribution for this client
            class_counts = client_df[label_col].value_counts()
            most_common_class = class_counts.index[0]
            most_common_pct = (class_counts.iloc[0] / total_samples) * 100
            
            # Get allocation info
            allocation_info = client_class_allocations[cid] if cid < len(client_class_allocations) else None
            is_specialist = allocation_info['is_specialist'] if allocation_info else False
            dominant_class = allocation_info['dominant_class'] if allocation_info else most_common_class
            
            client_stats.append({
                'client_id': cid + 1,
                'samples': total_samples,
                'unique_classes': unique_classes,
                'dominant_class': most_common_class,
                'dominant_pct': most_common_pct,
                'is_specialist': is_specialist,
                'class_list': list(class_counts.index)
            })
            
            specialist_tag = "🎯 SPECIALIST" if is_specialist else ""
            print(f"Client {cid+1}: {total_samples:>6} samples, {unique_classes:>2} classes, "
                  f"dominant: {most_common_class} ({most_common_pct:.1f}%) {specialist_tag}")
            
            total_client_samples += total_samples
        else:
            print(f"Client {cid+1}: {0:>6} samples, 0 classes, dominant: None")
            client_stats.append({
                'client_id': cid + 1,
                'samples': 0,
                'unique_classes': 0,
                'dominant_class': 'None',
                'dominant_pct': 0,
                'is_specialist': False,
                'class_list': []
            })
    
    print(f"\nTotal client samples: {total_client_samples}")
    
    # Enhanced statistics
    if client_stats:
        non_empty_clients = [c for c in client_stats if c['samples'] > 0]
        if non_empty_clients:
            avg_classes = np.mean([c['unique_classes'] for c in non_empty_clients])
            std_classes = np.std([c['unique_classes'] for c in non_empty_clients])
            specialists = len([c for c in non_empty_clients if c['is_specialist']])
            
            print(f"\n📊 Class Fluctuation Statistics:")
            print(f"   Average classes per client: {avg_classes:.1f} ± {std_classes:.1f}")
            print(f"   Class range: {min(c['unique_classes'] for c in non_empty_clients)} - {max(c['unique_classes'] for c in non_empty_clients)}")
            print(f"   Specialist clients: {specialists}/{len(non_empty_clients)}")
    
    # Verification
    total_original = len(df)
    total_split = len(df_global_train) + len(df_global_test) + total_client_samples
    print(f"\nSplit Verification:")
    print(f"Original dataset: {total_original:>8} samples")
    print(f"After splitting: {total_split:>8} samples")
    print(f"Difference: {total_original - total_split:>8} samples")
    
    return {
        "global_train": df_global_train.reset_index(drop=True),
        "global_test": df_global_test.reset_index(drop=True),
        "clients": client_data,
        "client_stats": client_stats,  # NEW: Detailed client statistics
        "split_info": {
            "global_train_pct": len(df_global_train) / len(df) * 100,
            "global_test_pct": len(df_global_test) / len(df) * 100,
            "client_pool_pct": total_client_samples / len(df) * 100,
            "alpha": alpha,
            "n_clients": n_clients,
            "original_size": len(df),
            "minority_classes": [cls for cls, count in original_counts.items() if count < min_samples_per_class * 10],
            "class_distribution_type": class_distribution_type,
            "min_classes_per_client": min_classes_per_client,
            "max_classes_per_client": max_classes_per_client,
            "avg_classes_per_client": avg_classes if 'avg_classes' in locals() else 0
        }
    }

# Keep your existing save_fl_splits and visualize_split_distribution functions exactly as they are

# Updated example usage
if __name__ == "__main__":
    print("🎯 Creating enhanced FL splits with class fluctuation control...")
    
    # Scenario 1: Moderate fluctuation (good for testing)
    # result = enhanced_fl_split_with_class_fluctuation(
    #     input_csv="selected_features_dataset.csv",
    #     label_col="Attack_type",
    #     class_distribution_type="variable",
    #     min_classes_per_client=3,
    #     max_classes_per_client=7,
    #     dominant_class_prob=0.6
    # )

    # Scenario 2: Extreme non-IID (challenging for FL)
    # result = enhanced_fl_split_with_class_fluctuation(
    #     input_csv="selected_features_dataset.csv", 
    #     label_col="Attack_type",
    #     class_distribution_type="extreme",
    #     min_classes_per_client=1,
    #     max_classes_per_client=10,
    #     dominant_class_prob=0.8,
    #     rare_class_clients=3
    # )

    # Scenario 3: Balanced but diverse (production-like)
    result = enhanced_fl_split_with_class_fluctuation(
        input_csv="selected_features_dataset.csv",
        label_col="Attack_type", 
        class_distribution_type="variable",
        min_classes_per_client=4,
        max_classes_per_client=9,
        dominant_class_prob=0.5
    )

    # Save the variable distribution
    save_fl_splits(result, output_dir="DATA_VARIABLE/")
    visualize_split_distribution(result, save_path="split_distribution_variable.png")