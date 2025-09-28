import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
from typing import Dict, List, Any

# Function to extract classes and metrics from the dict
def extract_metrics(model_dict: Dict[str, Any]):
    classes = [k for k in model_dict['classification_report'] if k not in ['accuracy', 'macro avg', 'weighted avg']]
    precision = model_dict['precision']
    recall = model_dict['recall']
    f1_score = model_dict['f1_score']
    support = [model_dict['classification_report'][cls]['support'] for cls in classes]
    cm = np.array(model_dict['confusion_matrix'])  # Ensure it's numpy for easier manipulation
    loss = model_dict['loss']
    accuracy = model_dict['accuracy']
    macro_precision = model_dict['macro_precision']
    macro_recall = model_dict['macro_recall']
    macro_f1 = model_dict['macro_f1']
    weighted_precision = model_dict['weighted_precision']
    weighted_recall = model_dict['weighted_recall']
    weighted_f1 = model_dict['weighted_f1']
    
    return {
        'classes': classes,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'support': support,
        'cm': cm,
        'loss': loss,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

# Normalize confusion matrix (row-wise)
def normalize_cm(cm):
    return cm / cm.sum(axis=1, keepdims=True)

# Function to compute deltas between consecutive models
def compute_deltas(df: pd.DataFrame, metric_cols: List[str]):
    delta_df = df.copy()
    for col in metric_cols:
        delta_df[f'{col}_delta'] = delta_df[col].diff()  # Difference from previous
    return delta_df

# Streamlit app
st.title("Federated Learning Model Metrics Visualization")

st.markdown("""
This app provides comprehensive and detailed visualizations of evaluation metrics for federated learning global models from uploaded JSON files.
Emphasis is placed on comparative plots to highlight performance changes across cycles, such as improvements in per-class metrics.
Each JSON contains metrics for one model (initial or after a cycle). Upload multiple files for cross-model comparisons.
Visualizations highlight imbalances, per-class performance, and evolution over cycles, including deltas/improvements.
Note: Data is non-IID, imbalanced, with some clients having few classes and limited testing data.
Model order is based on upload sequence for chronological comparisons.
""")

# File uploader for multiple JSON files
uploaded_files = st.file_uploader("Upload JSON files (one per model, in order of cycles)", type="json", accept_multiple_files=True)

if uploaded_files:
    model_data = {}
    overall_metrics = []  # For overall comparison
    per_class_precision = {}  # Dict of lists for per-class comparisons
    per_class_recall = {}
    per_class_f1 = {}
    per_class_support = {}
    
    model_names = []  # To keep track of order
    
    for idx, file in enumerate(uploaded_files):
        try:
            model_dict = json.load(file)
            extracted = extract_metrics(model_dict)
            model_name = file.name.replace('.json', '') or f"Model_{idx+1}"  # Fallback if no name
            model_names.append(model_name)
            model_data[model_name] = extracted
            
            # Collect overall metrics
            overall_metrics.append({
                'model': model_name,
                'loss': extracted['loss'],
                'accuracy': extracted['accuracy'],
                'macro_precision': extracted['macro_precision'],
                'macro_recall': extracted['macro_recall'],
                'macro_f1': extracted['macro_f1'],
                'weighted_precision': extracted['weighted_precision'],
                'weighted_recall': extracted['weighted_recall'],
                'weighted_f1': extracted['weighted_f1']
            })
            
            # Collect per-class metrics
            for metric, values in zip(['precision', 'recall', 'f1_score', 'support'], [extracted['precision'], extracted['recall'], extracted['f1_score'], extracted['support']]):
                for cls, val in zip(extracted['classes'], values):
                    if metric == 'precision':
                        if cls not in per_class_precision:
                            per_class_precision[cls] = []
                        per_class_precision[cls].append({'model': model_name, 'value': val})
                    elif metric == 'recall':
                        if cls not in per_class_recall:
                            per_class_recall[cls] = []
                        per_class_recall[cls].append({'model': model_name, 'value': val})
                    elif metric == 'f1_score':
                        if cls not in per_class_f1:
                            per_class_f1[cls] = []
                        per_class_f1[cls].append({'model': model_name, 'value': val})
                    elif metric == 'support':
                        if cls not in per_class_support:
                            per_class_support[cls] = []
                        per_class_support[cls].append({'model': model_name, 'value': val})
            
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    
    if model_data:
        # Tabs for different views
        tab1, tab2 = st.tabs(["Single Model Details", "Cross-Model Comparisons"])
        
        with tab1:
            # Selectbox to choose model for detailed view
            selected_model = st.selectbox("Select Model for Detailed Visualization", model_names)
            
            if selected_model:
                data = model_data[selected_model]
                classes = data['classes']
                precision = data['precision']
                recall = data['recall']
                f1_score = data['f1_score']
                support = data['support']
                cm = data['cm']
                
                st.header(f"Detailed Metrics for {selected_model}")
                
                # Display scalar metrics in columns
                st.subheader("Overall Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Loss", f"{data['loss']:.4f}")
                col2.metric("Accuracy", f"{data['accuracy']:.4f}")
                col3.metric("Macro Precision", f"{data['macro_precision']:.4f}")
                col4.metric("Macro Recall", f"{data['macro_recall']:.4f}")
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Macro F1", f"{data['macro_f1']:.4f}")
                col6.metric("Weighted Precision", f"{data['weighted_precision']:.4f}")
                col7.metric("Weighted Recall", f"{data['weighted_recall']:.4f}")
                col8.metric("Weighted F1", f"{data['weighted_f1']:.4f}")
                
                # Per-class metrics dataframe
                per_class_df = pd.DataFrame({
                    'Class': classes,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1_score,
                    'Support': support
                })
                
                st.subheader("Per-Class Metrics Table")
                st.dataframe(per_class_df.style.format("{:.4f}", subset=['Precision', 'Recall', 'F1-Score']).format("{:.0f}", subset=['Support']))
                
                # Bar chart for per-class metrics
                st.subheader("Per-Class Precision, Recall, F1-Score (Bar Chart)")
                melted_df = per_class_df.melt(id_vars=['Class'], value_vars=['Precision', 'Recall', 'F1-Score'])
                fig_bar = px.bar(
                    melted_df,
                    x='Class',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Per-Class Metrics',
                    labels={'value': 'Score', 'variable': 'Metric'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45, height=600)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Radar chart for per-class metrics
                st.subheader("Per-Class Metrics (Radar Chart)")
                radar_df = per_class_df.melt(id_vars=['Class'], value_vars=['Precision', 'Recall', 'F1-Score'])
                fig_radar = px.line_polar(
                    radar_df,
                    r='value',
                    theta='Class',
                    color='variable',
                    line_close=True,
                    title='Radar Chart of Per-Class Metrics'
                )
                fig_radar.update_layout(height=600)
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Scatter plot: Precision vs Recall, sized by Support
                st.subheader("Precision vs Recall Scatter (Bubble Size: Support)")
                fig_scatter = px.scatter(
                    per_class_df,
                    x='Precision',
                    y='Recall',
                    size='Support',
                    color='Class',
                    hover_name='Class',
                    title='Precision vs Recall per Class',
                    labels={'Precision': 'Precision', 'Recall': 'Recall'}
                )
                fig_scatter.update_layout(height=600)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Class distribution bar chart
                st.subheader("Class Distribution (Support)")
                fig_support = px.bar(
                    per_class_df,
                    x='Class',
                    y='Support',
                    title='Class Support Distribution (Highlighting Imbalance)',
                    labels={'Support': 'Number of Samples'}
                )
                fig_support.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig_support, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                normalize = st.checkbox("Normalize Confusion Matrix (Row-wise)", value=False, key=f"norm_{selected_model}")
                display_cm = normalize_cm(cm) if normalize else cm
                fig_cm = px.imshow(
                    display_cm,
                    labels=dict(x="Predicted", y="True", color="Value" if normalize else "Count"),
                    x=classes,
                    y=classes,
                    text_auto='.2f' if normalize else True,
                    aspect="auto",
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(title='Confusion Matrix Heatmap (Normalized)' if normalize else 'Confusion Matrix Heatmap', height=800, width=800)
                st.plotly_chart(fig_cm, use_container_width=True)
        
        with tab2:
            if len(overall_metrics) > 1:
                st.header("Cross-Model Comparisons (Focus on Evolutions and Improvements)")
                
                # Overall metrics table
                st.subheader("Overall Metrics Summary Table")
                overall_df = pd.DataFrame(overall_metrics)
                overall_df = overall_df.set_index('model')  # For delta calc
                st.dataframe(overall_df.style.format("{:.4f}"))
                
                # Compute deltas for overall metrics
                metric_cols = ['loss', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1']
                delta_overall_df = compute_deltas(overall_df.reset_index(), metric_cols)
                delta_overall_df = delta_overall_df.set_index('model')
                
                st.subheader("Overall Metrics Deltas (Change from Previous Model)")
                st.dataframe(delta_overall_df[[col for col in delta_overall_df.columns if '_delta' in col]].style.format("{:.4f}"))
                
                # Line charts for overall metrics evolution
                st.subheader("Evolution of Overall Metrics (Line Chart)")
                overall_melted = overall_df.reset_index().melt(id_vars=['model'], value_vars=metric_cols)
                fig_overall_line = px.line(
                    overall_melted,
                    x='model',
                    y='value',
                    color='variable',
                    markers=True,
                    title='Overall Metric Evolution Over Models/Cycles',
                    labels={'value': 'Score', 'variable': 'Metric', 'model': 'Model/Cycle'}
                )
                fig_overall_line.update_layout(xaxis_title='Model/Cycle', yaxis_title='Score', height=600)
                st.plotly_chart(fig_overall_line, use_container_width=True)
                
                # Bar chart for overall deltas
                st.subheader("Improvements in Overall Metrics (Delta Bar Chart)")
                delta_melted = delta_overall_df.reset_index().melt(id_vars=['model'], value_vars=[f'{col}_delta' for col in metric_cols if col != 'loss'])  # Exclude loss for positive improvements
                delta_melted['variable'] = delta_melted['variable'].str.replace('_delta', '')
                fig_delta_bar = px.bar(
                    delta_melted,
                    x='model',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Delta (Improvement) in Overall Metrics from Previous',
                    labels={'value': 'Delta Score', 'variable': 'Metric', 'model': 'Model/Cycle'}
                )
                fig_delta_bar.update_layout(height=600)
                st.plotly_chart(fig_delta_bar, use_container_width=True)
                
                # Per-class comparisons with focus on deltas
                for metric, data_dict in zip(['Precision', 'Recall', 'F1-Score', 'Support'], [per_class_precision, per_class_recall, per_class_f1, per_class_support]):
                    st.subheader(f"Evolution and Improvements in Per-Class {metric}")
                    all_class_data = []
                    for cls, vals in data_dict.items():
                        for val in vals:
                            val['class'] = cls
                            all_class_data.append(val)
                    per_class_df = pd.DataFrame(all_class_data)
                    
                    # Pivot for easier delta calc
                    pivot_df = per_class_df.pivot(index='model', columns='class', values='value')
                    
                    # Evolution line chart
                    fig_per_class_line = px.line(
                        per_class_df,
                        x='model',
                        y='value',
                        color='class',
                        markers=True,
                        title=f'Per-Class {metric} Evolution Over Models/Cycles',
                        labels={'value': metric, 'class': 'Class', 'model': 'Model/Cycle'}
                    )
                    fig_per_class_line.update_layout(xaxis_title='Model/Cycle', yaxis_title=metric, height=600)
                    st.plotly_chart(fig_per_class_line, use_container_width=True)
                    
                    # Heatmap for per-class metric
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='Viridis',
                        text=pivot_df.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size":10}
                    ))
                    fig_heatmap.update_layout(
                        title=f'Heatmap of Per-Class {metric} Across Models',
                        height=800,
                        xaxis_title='Class',
                        yaxis_title='Model/Cycle'
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Compute per-class deltas
                    delta_pivot_df = pivot_df.diff()  # Delta from previous
                    st.subheader(f"Deltas in Per-Class {metric} (Change from Previous Model)")
                    st.dataframe(delta_pivot_df.style.format("{:.4f}"))
                    
                    # Delta heatmap
                    fig_delta_heatmap = go.Figure(data=go.Heatmap(
                        z=delta_pivot_df.values,
                        x=delta_pivot_df.columns,
                        y=delta_pivot_df.index,
                        colorscale='RdBu',
                        text=delta_pivot_df.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size":10}
                    ))
                    fig_delta_heatmap.update_layout(
                        title=f'Heatmap of Deltas in Per-Class {metric} (Red: Decrease, Blue: Increase)',
                        height=800,
                        xaxis_title='Class',
                        yaxis_title='Model/Cycle'
                    )
                    st.plotly_chart(fig_delta_heatmap, use_container_width=True)
                    
                    # Delta bar chart (average delta per model)
                    avg_delta_per_model = delta_pivot_df.mean(axis=1).reset_index(name='avg_delta')
                    avg_delta_per_model['model'] = avg_delta_per_model['model']
                    fig_avg_delta_bar = px.bar(
                        avg_delta_per_model,
                        x='model',
                        y='avg_delta',
                        title=f'Average Delta in {metric} Per Model/Cycle',
                        labels={'avg_delta': 'Average Delta', 'model': 'Model/Cycle'}
                    )
                    fig_avg_delta_bar.update_layout(height=500)
                    st.plotly_chart(fig_avg_delta_bar, use_container_width=True)
                
                # Additional comparative plots
                st.subheader("Bar Comparison of Key Overall Metrics Across Models")
                bar_melted = overall_df.reset_index().melt(id_vars=['model'], value_vars=['accuracy', 'macro_f1', 'weighted_f1'])
                fig_bar_comp = px.bar(
                    bar_melted,
                    x='model',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Overall Metrics Bar Comparison',
                    labels={'value': 'Score', 'variable': 'Metric', 'model': 'Model/Cycle'}
                )
                fig_bar_comp.update_layout(height=600)
                st.plotly_chart(fig_bar_comp, use_container_width=True)
            else:
                st.info("Upload multiple JSON files to enable cross-model comparisons and delta analyses.")
else:
    st.info("Upload JSON files to begin visualization.")