# wandb_fl_tracker.py - Comprehensive FL tracking with WandB

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import os

class FederatedLearningTracker:
    """
    Comprehensive WandB tracker for Federated Learning with cybersecurity focus
    """
    
    def __init__(self, project_name="SecureFL-Cybersecurity", entity=None, config=None):
        """
        Initialize WandB FL tracker
        
        Args:
            project_name: WandB project name
            entity: WandB entity (team/user)
            config: FL configuration dictionary
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.fl_round = 0
        self.all_metrics = []
        
        # Default config for cybersecurity FL
        self.default_config = {
            "architecture": [256, 256],
            "learning_rate": 5e-5,
            "num_classes": 15,
            "batch_size": 128,
            "dataset": "ML-Edge-IIoT",
            "features": 25,
            "fl_algorithm": "FedAvg",
            "aggregation_strategy": "weighted_average",
            "data_distribution": "non_iid",
            "total_clients": 10,
            "min_clients_per_round": 2,
            "differential_privacy": True,
            "epsilon": 1.0,
            "delta": 1e-5,
        }
        
        if config:
            self.default_config.update(config)
    
    def initialize_run(self, run_name=None, tags=None, notes=None, config=None):
        """
        FIXED: Initialize WandB run with proper config handling
        """
        if run_name is None:
            run_name = f"FL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Default tags for cybersecurity FL
        default_tags = ["federated_learning", "cybersecurity", "non_iid", "ml_edge_iiot"]
        if tags:
            default_tags.extend(tags)
        
        # FIXED: Merge provided config with default config
        final_config = self.default_config.copy()
        if config:
            final_config.update(config)
        
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=final_config,  # FIXED: Use merged config
            tags=default_tags,
            notes=notes or "Federated Learning experiment on cybersecurity dataset with non-IID data distribution"
        )
        
        # Define custom metrics
        wandb.define_metric("fl_round")
        wandb.define_metric("global_model/*", step_metric="fl_round")
        wandb.define_metric("per_class/*", step_metric="fl_round")
        wandb.define_metric("client_contribution/*", step_metric="fl_round")
        
        print(f"🎯 WandB run initialized: {run_name}")
        print(f"🔗 WandB URL: {self.run.url}")
        return self.run

    def log_global_model_performance(self, fl_round, test_metrics, num_contributing_clients, 
                                   client_ids=None, model_version=None):
        """
        Log comprehensive global model performance metrics
        
        Args:
            fl_round: Current FL round number
            test_metrics: Dictionary containing test evaluation metrics
            num_contributing_clients: Number of clients that contributed
            client_ids: List of contributing client IDs
            model_version: Global model version number
        """
        self.fl_round = fl_round
        
        # Core performance metrics
        core_metrics = {
            "fl_round": fl_round,
            "global_model/accuracy": test_metrics.get("accuracy", 0),
            "global_model/loss": test_metrics.get("loss", float('inf')),
            "global_model/macro_f1": test_metrics.get("macro_f1", 0),
            "global_model/weighted_f1": test_metrics.get("weighted_f1", 0),
            "global_model/macro_precision": test_metrics.get("macro_precision", 0),
            "global_model/macro_recall": test_metrics.get("macro_recall", 0),
            "global_model/weighted_precision": test_metrics.get("weighted_precision", 0),
            "global_model/weighted_recall": test_metrics.get("weighted_recall", 0),
            "client_contribution/num_clients": num_contributing_clients,
            "client_contribution/participation_rate": num_contributing_clients / self.default_config.get("total_clients", 10),
        }
        
        if model_version:
            core_metrics["global_model/version"] = model_version
        
        # Per-class metrics (crucial for cybersecurity non-IID analysis)
        if "precision" in test_metrics and "recall" in test_metrics and "f1_score" in test_metrics:
            class_names = [
                'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
            ]
            
            for i, class_name in enumerate(class_names):
                if i < len(test_metrics["precision"]):
                    core_metrics.update({
                        f"per_class/{class_name}/precision": test_metrics["precision"][i],
                        f"per_class/{class_name}/recall": test_metrics["recall"][i], 
                        f"per_class/{class_name}/f1": test_metrics["f1_score"][i],
                        f"per_class/{class_name}/support": test_metrics.get("support", [0] * 15)[i],
                    })
        
        # Client participation tracking
        if client_ids:
            core_metrics["client_contribution/active_clients"] = len(set(client_ids))
            # Log client ID distribution for analysis
            client_freq = pd.Series(client_ids).value_counts().to_dict()
            for client_id, freq in client_freq.items():
                core_metrics[f"client_contribution/client_{client_id}_contributions"] = freq
        
        # Log to WandB
        wandb.log(core_metrics)
        
        # Store for trend analysis
        metric_record = {
            "fl_round": fl_round,
            "timestamp": datetime.now(),
            **test_metrics,
            "num_contributing_clients": num_contributing_clients
        }
        self.all_metrics.append(metric_record)
        
        print(f"📊 FL Round {fl_round}: Logged metrics to WandB")
        print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"   Macro F1: {test_metrics.get('macro_f1', 0):.4f}")
        print(f"   Contributing clients: {num_contributing_clients}")
    
    def create_performance_trends_plot(self):
        """
        Create comprehensive performance trends visualization
        """
        if len(self.all_metrics) < 2:
            print("⚠️ Need at least 2 FL rounds to create trends plot")
            return None
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.all_metrics)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Global Model Accuracy", "Global Model Loss", 
                          "F1-Score Progression", "Client Participation"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy trend
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['accuracy'], 
                      mode='lines+markers', name='Test Accuracy',
                      line=dict(color='#1f77b4', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # Loss trend
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['loss'],
                      mode='lines+markers', name='Test Loss',
                      line=dict(color='#ff7f0e', width=3),
                      marker=dict(size=8)),
            row=1, col=2
        )
        
        # F1-Score trends
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['macro_f1'],
                      mode='lines+markers', name='Macro F1',
                      line=dict(color='#2ca02c', width=3),
                      marker=dict(size=8)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['weighted_f1'],
                      mode='lines+markers', name='Weighted F1',
                      line=dict(color='#d62728', width=3),
                      marker=dict(size=8)),
            row=2, col=1
        )
        
        # Client participation
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['num_contributing_clients'],
                      mode='lines+markers', name='Contributing Clients',
                      line=dict(color='#9467bd', width=3),
                      marker=dict(size=8)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Federated Learning Performance Trends - Cybersecurity Classification',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="FL Round", row=2, col=1)
        fig.update_xaxes(title_text="FL Round", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="F1-Score", row=2, col=1)
        fig.update_yaxes(title_text="Number of Clients", row=2, col=2)
        
        # Log to WandB
        wandb.log({"performance_trends": wandb.Html(fig.to_html())})
        
        return fig
    
    def create_per_class_performance_heatmap(self):
        """
        Create per-class performance heatmap across FL rounds
        """
        if len(self.all_metrics) < 2:
            print("⚠️ Need at least 2 FL rounds to create per-class heatmap")
            return None
        
        # Extract per-class F1 scores across rounds
        class_names = [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ]
        
        # Build per-class performance matrix
        performance_matrix = []
        fl_rounds = []
        
        for metrics in self.all_metrics:
            if 'f1_score' in metrics and metrics['f1_score']:
                performance_matrix.append(metrics['f1_score'][:len(class_names)])
                fl_rounds.append(f"Round {metrics['fl_round']}")
        
        if not performance_matrix:
            print("⚠️ No per-class F1 scores available for heatmap")
            return None
        
        # Create DataFrame
        df_heatmap = pd.DataFrame(performance_matrix, 
                                 index=fl_rounds, 
                                 columns=class_names)
        
        # Create plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df_heatmap.values,
            x=df_heatmap.columns,
            y=df_heatmap.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="F1-Score"),
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         '<b>%{x}</b><br>' +
                         'F1-Score: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Per-Class F1-Score Evolution Across FL Rounds',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Attack Type",
            yaxis_title="FL Round",
            height=600,
            template="plotly_white"
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        # Log to WandB
        wandb.log({"per_class_performance_heatmap": wandb.Html(fig.to_html())})
        
        return fig
    
    def create_confusion_matrix_comparison(self, current_cm, class_names, fl_round):
        """
        Create and log confusion matrix with comparison to previous round
        """
        # Create confusion matrix visualization
        fig = go.Figure(data=go.Heatmap(
            z=current_cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            colorbar=dict(title="Count"),
            hovertemplate='<b>True: %{y}</b><br>' +
                         '<b>Pred: %{x}</b><br>' +
                         'Count: %{z}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'Confusion Matrix - FL Round {fl_round}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Predicted Attack Type",
            yaxis_title="True Attack Type",
            height=600,
            template="plotly_white"
        )
        
        # Rotate labels
        fig.update_xaxes(tickangle=45)
        
        # Log to WandB
        wandb.log({
            f"confusion_matrix_round_{fl_round}": wandb.Html(fig.to_html()),
            f"confusion_matrix_data_round_{fl_round}": wandb.Table(
                data=current_cm.tolist(),
                columns=class_names
            )
        })
        
        return fig
    
    def log_model_architecture_summary(self, model_summary_str):
        """
        Log model architecture details
        """
        wandb.log({"model_architecture": wandb.Html(f"<pre>{model_summary_str}</pre>")})
    
    def log_data_distribution_info(self, client_stats):
        """
        Log client data distribution information for non-IID analysis
        """
        if not client_stats:
            return
        
        # Create client distribution summary
        distribution_data = []
        for stat in client_stats:
            distribution_data.append({
                "client_id": stat["client_id"],
                "samples": stat["samples"], 
                "unique_classes": stat["unique_classes"],
                "dominant_class": stat["dominant_class"],
                "dominant_pct": stat["dominant_pct"],
                "is_specialist": stat["is_specialist"]
            })
        
        # Log as WandB table
        wandb.log({
            "client_data_distribution": wandb.Table(
                data=[[d["client_id"], d["samples"], d["unique_classes"], 
                      d["dominant_class"], d["dominant_pct"], d["is_specialist"]] 
                     for d in distribution_data],
                columns=["Client ID", "Samples", "Unique Classes", 
                        "Dominant Class", "Dominant %", "Is Specialist"]
            )
        })
        
        # Create visualization of data heterogeneity
        df_dist = pd.DataFrame(distribution_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Samples per Client", "Classes per Client")
        )
        
        # Samples distribution
        fig.add_trace(
            go.Bar(x=df_dist["client_id"], y=df_dist["samples"], 
                  name="Samples", marker_color="lightblue"),
            row=1, col=1
        )
        
        # Classes distribution  
        fig.add_trace(
            go.Bar(x=df_dist["client_id"], y=df_dist["unique_classes"],
                  name="Classes", marker_color="lightcoral"),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Non-IID Data Distribution Across Clients",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Client ID")
        fig.update_yaxes(title_text="Number of Samples", row=1, col=1)
        fig.update_yaxes(title_text="Number of Classes", row=1, col=2)
        
        wandb.log({"client_data_heterogeneity": wandb.Html(fig.to_html())})
    
    def finalize_run(self, summary_metrics=None):
        """
        Finalize WandB run with summary metrics and artifacts
        """
        if not self.run:
            print("⚠️ No active WandB run to finalize")
            return
        
        # Create final performance summary
        if len(self.all_metrics) > 0:
            final_metrics = self.all_metrics[-1]
            
            # Log final summary
            final_summary = {
                "final_accuracy": final_metrics.get("accuracy", 0),
                "final_macro_f1": final_metrics.get("macro_f1", 0),
                "final_weighted_f1": final_metrics.get("weighted_f1", 0),
                "final_loss": final_metrics.get("loss", float('inf')),
                "total_fl_rounds": self.fl_round,
                "best_accuracy": max([m.get("accuracy", 0) for m in self.all_metrics]),
                "best_macro_f1": max([m.get("macro_f1", 0) for m in self.all_metrics]),
            }
            
            if summary_metrics:
                final_summary.update(summary_metrics)
            
            # Log summary
            for key, value in final_summary.items():
                wandb.run.summary[key] = value
            
            # Create final visualizations
            self.create_performance_trends_plot()
            self.create_per_class_performance_heatmap()
        
        print(f"🎯 WandB run finalized: {wandb.run.name}")
        wandb.finish()

# Global tracker instance
fl_tracker = FederatedLearningTracker()
