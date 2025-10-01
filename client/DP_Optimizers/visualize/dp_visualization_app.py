import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np

# Page config
st.set_page_config(
    page_title="DP Mechanisms Comparison",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all result files"""
    try:
        with open('client/DP_Optimizers/visualize/base_model.json', 'r') as f:
            base = json.load(f)
        with open('client/DP_Optimizers/visualize/detailed_results.json', 'r') as f:
            detailed = json.load(f)
        with open('client/DP_Optimizers/visualize/t_laplace_dp_results_exact_same_params.json', 'r') as f:
            trad_same = json.load(f)
        with open('client/DP_Optimizers/visualize/t_laplace_dp_results_equivalent_params.json', 'r') as f:
            trad_equiv = json.load(f)

        summary_df = pd.read_csv('client/DP_Optimizers/visualize/summary_results.csv')

        return base, detailed, trad_same, trad_equiv, summary_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

base, detailed, trad_same, trad_equiv, summary_df = load_data()

# Class names
CLASS_NAMES = [
    'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
    'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
    'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
]

# Main title
st.markdown('<div class="main-header">🔒 Differential Privacy Mechanisms Comparison</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Advanced Laplace vs Gaussian vs Traditional Laplace | Research Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")

    page = st.radio(
        "Select Analysis:",
        [
            "📊 Overview & Summary",
            "📈 Privacy-Utility Curves",
            "⚖️ Mechanism Comparison",
            "🔍 Traditional Laplace Analysis",
            "📉 Per-Class Performance",
            "🎯 Confusion Matrices",
            "⏱️ Training Efficiency",
            "📋 Detailed Metrics Table"
        ]
    )

    st.markdown("---")
    st.markdown("### Experiment Details")
    st.info("""
    **Dataset:** ML-Edge_IIoT  
    **Classes:** 15 attack types  
    **Samples:** 100,986 training  
    **Network:** 35→256→256→15  
    **Batch Size:** 1,024  
    **Epochs:** 20  
    **Privacy Budget:** ε ∈ {1, 3, 5, 10}
    """)

# =============================================================================
# PAGE 1: OVERVIEW & SUMMARY
# =============================================================================
if page == "📊 Overview & Summary":
    st.markdown('<div class="sub-header">Executive Summary</div>', unsafe_allow_html=True)

    # Key findings
    col1, col2, col3, col4 = st.columns(4)

    # Extract ε=3 results
    eps3_data = [d for d in detailed if d['target_epsilon'] == 3][0]

    # Base Model
    with col1:
        base_acc = base['test_metrics']['accuracy'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Base Model (No DP)</div>
            <div style="font-size: 2rem; font-weight: bold; color: #2ca02c; margin-bottom: 0.5rem;">{base_acc:.2f}%</div>
            <div style="background: #e8e8e8; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #2ca02c, #45b545); height: 100%; width: {base_acc}%; transition: width 0.3s;"></div>
            </div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 0.3rem;">Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    # Gaussian
    with col2:
        gauss_acc = eps3_data['gaussian']['accuracy'] * 100
        gauss_diff = gauss_acc - eps3_data['advanced_laplace']['accuracy'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Gaussian (ε=3)</div>
            <div style="font-size: 2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem;">{gauss_acc:.2f}%</div>
            <div style="background: #e8e8e8; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #1f77b4, #4a9fd6); height: 100%; width: {gauss_acc}%; transition: width 0.3s;"></div>
            </div>
            <div style="font-size: 0.8rem; color: #2ca02c; margin-top: 0.3rem;">+{gauss_diff:.1f}% vs Adv Laplace</div>
        </div>
        """, unsafe_allow_html=True)

    # Advanced Laplace
    with col3:
        adv_acc = eps3_data['advanced_laplace']['accuracy'] * 100
        adv_diff = base_acc - adv_acc
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Advanced Laplace (ε=3)</div>
            <div style="font-size: 2rem; font-weight: bold; color: #ff7f0e; margin-bottom: 0.5rem;">{adv_acc:.2f}%</div>
            <div style="background: #e8e8e8; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #ff7f0e, #ffaa5c); height: 100%; width: {adv_acc}%; transition: width 0.3s;"></div>
            </div>
            <div style="font-size: 0.8rem; color: #d62728; margin-top: 0.3rem;">-{adv_diff:.1f}% vs Base</div>
        </div>
        """, unsafe_allow_html=True)

    # Traditional Laplace
    with col4:
        trad_acc = trad_same['final_test_metrics']['accuracy'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Traditional Laplace (L1=3)</div>
            <div style="font-size: 2rem; font-weight: bold; color: #d62728; margin-bottom: 0.5rem;">{trad_acc:.2f}%</div>
            <div style="background: #e8e8e8; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #d62728, #ff5c5c); height: 100%; width: {trad_acc}%; transition: width 0.3s;"></div>
            </div>
            <div style="font-size: 0.8rem; color: #d62728; margin-top: 0.3rem;">Impractical ⚠️</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Summary comparison chart
    st.markdown("### Overall Accuracy Comparison (ε=3, δ=1e-5)")

    comparison_data = {
        'Mechanism': [
            'Base Model (No DP)',
            'Gaussian (L2=3)',
            'Advanced Laplace (L2=3)',
            'Traditional Laplace (L1=3)',
            'Traditional Laplace (L1=953)'
        ],
        'Accuracy': [
            base['test_metrics']['accuracy'] * 100,
            eps3_data['gaussian']['accuracy'] * 100,
            eps3_data['advanced_laplace']['accuracy'] * 100,
            trad_same['final_test_metrics']['accuracy'] * 100,
            trad_equiv['final_test_metrics']['accuracy'] * 100
        ],
        'Privacy Type': ['None', '(ε,δ)-DP', '(ε,δ)-DP', 'Pure ε-DP', 'Pure ε-DP'],
        'Color': ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    }

    fig = go.Figure()

    for i, mech in enumerate(comparison_data['Mechanism']):
        fig.add_trace(go.Bar(
            x=[mech],
            y=[comparison_data['Accuracy'][i]],
            name=mech,
            marker_color=comparison_data['Color'][i],
            text=[f"{comparison_data['Accuracy'][i]:.1f}%"],
            textposition='outside',
            showlegend=False,
            hovertemplate=f"<b>{mech}</b><br>Accuracy: {comparison_data['Accuracy'][i]:.2f}%<br>{comparison_data['Privacy Type'][i]}<extra></extra>"
        ))

    fig.update_layout(
        title="Accuracy Comparison at ε=3",
        yaxis_title="Test Accuracy (%)",
        yaxis_range=[0, 100],
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True, key="overview_accuracy_comparison")

    # Key insights
    st.markdown("### 🔑 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **✅ Gaussian Mechanism Wins**
        - **77.1%** accuracy at ε=3
        - **52.9%** better than Advanced Laplace
        - **67.2%** better than Traditional Laplace
        - RDP accounting provides 363× less noise
        """)

        st.info("""
        **📊 Advanced Laplace (L2)**
        - **24.2%** accuracy at ε=3
        - L2 sensitivity appropriate for DL
        - Still suffers from composition overhead
        - Sequential composition: noise ~1,199
        """)

    with col2:
        st.warning("""
        **⚠️ Traditional Laplace Fails**
        - **9.6%** with L1=3 (unfair comparison)
        - **7.9%** with L1=953 (fair but impractical)
        - L1 sensitivity incompatible with high-dim
        - Sequential composition: noise ~622,849
        """)

        st.error("""
        **❌ Privacy-Utility Tradeoff**
        - Base model: 81.2% (no privacy)
        - Gaussian loss: 4.1% (excellent tradeoff)
        - Adv Laplace loss: 57.0% (poor tradeoff)
        - Trad Laplace loss: 71.3% (catastrophic)
        """)

# =============================================================================
# PAGE 2: PRIVACY-UTILITY CURVES
# =============================================================================
elif page == "📈 Privacy-Utility Curves":
    st.markdown('<div class="sub-header">Privacy-Utility Tradeoff Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    This plot shows how model accuracy changes across different privacy budgets (ε).  
    **Key Finding:** Gaussian mechanism consistently outperforms Advanced Laplace across all privacy levels.
    """)

    # Extract data for curves
    epsilons = summary_df['Epsilon'].values
    adv_laplace_acc = summary_df['Adv_Laplace_Acc'].values * 100
    gaussian_acc = summary_df['Gaussian_Acc'].values * 100
    gaps = summary_df['Accuracy_Gap'].values * 100

    # Main privacy-utility curve
    fig = go.Figure()

    # Gaussian line
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=gaussian_acc,
        mode='lines+markers',
        name='Gaussian (RDP)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=12, symbol='square'),
        hovertemplate='<b>Gaussian</b><br>ε: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
    ))

    # Advanced Laplace line
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=adv_laplace_acc,
        mode='lines+markers',
        name='Advanced Laplace (L2)',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=12, symbol='circle'),
        hovertemplate='<b>Advanced Laplace</b><br>ε: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
    ))

    # Base model reference
    base_acc = base['test_metrics']['accuracy'] * 100
    fig.add_hline(y=base_acc, line_dash="dash", line_color="green",
                  annotation_text=f"Base Model (No DP): {base_acc:.1f}%",
                  annotation_position="right")

    # Random guessing reference
    fig.add_hline(y=100/15, line_dash="dot", line_color="red",
                  annotation_text=f"Random Guessing: {100/15:.1f}%",
                  annotation_position="left")

    fig.update_layout(
        title="Privacy-Utility Curves: Advanced Laplace vs Gaussian (δ=1e-5)",
        xaxis_title="Privacy Budget (ε)",
        yaxis_title="Test Accuracy (%)",
        height=600,
        template='plotly_white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True, key="privacy_utility_curves")

    # Detailed table
    st.markdown("### Detailed Results Table")

    table_df = pd.DataFrame({
        'ε': epsilons,
        'Gaussian Acc (%)': gaussian_acc,
        'Adv Laplace Acc (%)': adv_laplace_acc,
        'Gap (%)': gaps,
        'Noise Multiplier': summary_df['Noise_Multiplier'].values,
        'Gaussian ε_final': summary_df['Gaussian_Final_Eps'].round(4),
        'Adv Laplace ε_final': summary_df['Adv_Laplace_Final_Eps'].round(4)
    })

    st.dataframe(table_df, width='stretch')

# =============================================================================
# PAGE 3: MECHANISM COMPARISON
# =============================================================================
elif page == "⚖️ Mechanism Comparison":
    st.markdown('<div class="sub-header">Detailed Mechanism Comparison</div>', unsafe_allow_html=True)

    eps_select = st.selectbox("Select Privacy Budget (ε):", [1, 3, 5, 10], index=1, 
                            help="Choose privacy budget to compare mechanisms")

    eps_data = [d for d in detailed if d['target_epsilon'] == eps_select][0]

    # Metrics comparison with beautiful cards
    col1, col2, col3 = st.columns(3)

    # Gaussian Mechanism Card
    with col1:
        gauss_acc = eps_data['gaussian']['accuracy'] * 100
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 1rem; display: flex; align-items: center;">
                Gaussian
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem;">
                <div style="font-size: 0.9rem; opacity: 0.9;">Accuracy</div>
                <div style="font-size: 2.2rem; font-weight: bold;">{gauss_acc:.2f}%</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;">
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Final ε</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{eps_data['gaussian']['final_epsilon']:.4f}</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Time</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{eps_data['gaussian']['time_seconds']:.1f}s</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px; grid-column: span 2;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Macro F1-Score</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{eps_data['gaussian']['test_metrics']['macro_f1']*100:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Advanced Laplace Card
    with col2:
        adv_acc = eps_data['advanced_laplace']['accuracy'] * 100
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 1rem; display: flex; align-items: center;">
                Advanced Laplace
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem;">
                <div style="font-size: 0.9rem; opacity: 0.9;">Accuracy</div>
                <div style="font-size: 2.2rem; font-weight: bold;">{adv_acc:.2f}%</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;">
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Final ε</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{eps_data['advanced_laplace']['final_epsilon']:.4f}</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Time</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{eps_data['advanced_laplace']['time_seconds']:.1f}s</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px; grid-column: span 2;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Macro F1-Score</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{eps_data['advanced_laplace']['test_metrics']['macro_f1']*100:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Comparison Card
    with col3:
        gap = (eps_data['gaussian']['accuracy'] - eps_data['advanced_laplace']['accuracy']) * 100
        rel_improvement = (gap / eps_data['advanced_laplace']['accuracy']) * 100
        noise_mult = eps_data['noise_multiplier']
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 1rem; display: flex; align-items: center;">
                Comparison
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem;">
                <div style="font-size: 0.9rem; opacity: 0.9;">Gaussian Advantage</div>
                <div style="font-size: 2.2rem; font-weight: bold;">+{gap:.2f}%</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr; gap: 0.8rem;">
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Relative Improvement</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">+{rel_improvement:.1f}%</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Result</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">
                        Gaussian Wins
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Precision, Recall, F1 comparison
    st.markdown("### Precision, Recall, F1-Score Comparison")

    metrics_df = pd.DataFrame({
        'Metric': ['Macro Precision', 'Macro Recall', 'Macro F1', 
                   'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
        'Gaussian': [
            eps_data['gaussian']['test_metrics']['macro_precision'] * 100,
            eps_data['gaussian']['test_metrics']['macro_recall'] * 100,
            eps_data['gaussian']['test_metrics']['macro_f1'] * 100,
            eps_data['gaussian']['test_metrics']['weighted_precision'] * 100,
            eps_data['gaussian']['test_metrics']['weighted_recall'] * 100,
            eps_data['gaussian']['test_metrics']['weighted_f1'] * 100
        ],
        'Advanced Laplace': [
            eps_data['advanced_laplace']['test_metrics']['macro_precision'] * 100,
            eps_data['advanced_laplace']['test_metrics']['macro_recall'] * 100,
            eps_data['advanced_laplace']['test_metrics']['macro_f1'] * 100,
            eps_data['advanced_laplace']['test_metrics']['weighted_precision'] * 100,
            eps_data['advanced_laplace']['test_metrics']['weighted_recall'] * 100,
            eps_data['advanced_laplace']['test_metrics']['weighted_f1'] * 100
        ]
    })

    fig_metrics = go.Figure()

    fig_metrics.add_trace(go.Bar(
        x=metrics_df['Metric'],
        y=metrics_df['Gaussian'],
        name='Gaussian',
        marker_color='#1f77b4'
    ))

    fig_metrics.add_trace(go.Bar(
        x=metrics_df['Metric'],
        y=metrics_df['Advanced Laplace'],
        name='Advanced Laplace',
        marker_color='#ff7f0e'
    ))

    fig_metrics.update_layout(
        title=f"Performance Metrics Comparison (ε={eps_select})",
        yaxis_title="Score (%)",
        barmode='group',
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig_metrics, use_container_width=True, key="mechanism_comparison_metrics")

# =============================================================================
# PAGE 4: TRADITIONAL LAPLACE ANALYSIS
# =============================================================================
elif page == "🔍 Traditional Laplace Analysis":
    st.markdown('<div class="sub-header">Why Traditional Laplace Fails</div>', unsafe_allow_html=True)

    st.markdown("""
    This section demonstrates why Traditional Laplace mechanism is impractical for deep learning,
    comparing two scenarios: same numerical bounds (L1=3) and equivalent sensitivity (L1=953).
    """)
    # Extract ε=3 results
    eps3_data = [d for d in detailed if d['target_epsilon'] == 3][0]

    # Comparison chart
    trad_data = {
        'Configuration': [
            'Base Model (No DP)',
            'Gaussian (L2=3, ε=3)',
            'Advanced Laplace (L2=3, ε=3)',
            'Traditional Laplace (L1=3, ε=3)',
            'Traditional Laplace (L1=953, ε=3)'
        ],
        'Accuracy': [
            base['test_metrics']['accuracy'] * 100,
            eps3_data['gaussian']['accuracy'] * 100,
            eps3_data['advanced_laplace']['accuracy'] * 100,
            trad_same['final_test_metrics']['accuracy'] * 100,
            trad_equiv['final_test_metrics']['accuracy'] * 100
        ],
        'Clip Bound': ['N/A', 'L2=3', 'L2=3', 'L1=3', 'L1=953'],
        'Finding': [
            'Baseline',
            'RDP Accounting',
            'Advanced Composition',
            'Unfair: 280× aggressive clipping',
            'Fair but impractical'
        ]
    }

    fig = go.Figure()

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    for i, (conf, acc) in enumerate(zip(trad_data['Configuration'], trad_data['Accuracy'])):
        fig.add_trace(go.Bar(
            x=[conf],
            y=[acc],
            marker_color=colors[i],
            text=[f"{acc:.1f}%"],
            textposition='outside',
            name=conf,
            showlegend=False,
            hovertemplate=f"<b>{conf}</b><br>Accuracy: {acc:.2f}%<br>{trad_data['Clip Bound'][i]}<br>{trad_data['Finding'][i]}<extra></extra>"
        ))

    fig.update_layout(
        title="Traditional Laplace vs L2-based Mechanisms",
        yaxis_title="Test Accuracy (%)",
        yaxis_range=[0, 100],
        height=600,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True, key="traditional_laplace_comparison")

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Traditional Laplace (L1=3)")
        st.error(f"""
        **Accuracy:** {trad_same['final_test_metrics']['accuracy']*100:.2f}%

        **Problem:** L1 clipping at C=3 is ~280× more aggressive than L2 clipping

        **Why:** For gradient dimension n=78,863:
        - ||g||₁ ≈ √n × ||g||₂
        - L1=3 equivalent to L2≈0.01

        **Result:** Catastrophic signal loss → Random-level performance
        """)

    with col2:
        st.markdown("#### Traditional Laplace (L1=953)")
        st.warning(f"""
        **Accuracy:** {trad_equiv['final_test_metrics']['accuracy']*100:.2f}%

        **Approach:** L1 = L2 × √n = 3 × 280 ≈ 953 (fair scaling)

        **Problem:** Noise scales with sensitivity
        - Noise scale: ~622,849
        - 519× larger than Advanced Laplace (1,199)
        - 188,742× larger than Gaussian (3.3)

        **Result:** Fair comparison but still impractical
        """)

    st.markdown("---")

    st.markdown("### 💡 Key Takeaway")
    st.info("""
    **Traditional Laplace is fundamentally incompatible with deep learning:**

    1. **L1 Sensitivity Problem:** High-dimensional gradients have L1 norm ≈ √n × L2 norm
    2. **Composition Overhead:** Sequential composition requires ε_step = ε_total / T, making noise prohibitive
    3. **No Fair Comparison:** Either unfair (same bounds) or impractical (scaled bounds)

    **This validates the field's move to:**
    - L2 sensitivity (more appropriate for high-dim)
    - Advanced composition (for Laplace)
    - RDP accounting (for Gaussian) ✅
    """)

# =============================================================================
# PAGE 5: PER-CLASS PERFORMANCE
# =============================================================================
elif page == "📉 Per-Class Performance":
    st.markdown('<div class="sub-header">Per-Class Accuracy Analysis</div>', unsafe_allow_html=True)

    eps_select = st.selectbox("Select Privacy Budget (ε):", [1, 3, 5, 10], index=1, key='class_eps')

    eps_data = [d for d in detailed if d['target_epsilon'] == eps_select][0]

    # Extract per-class metrics
    gauss_precision = np.array(eps_data['gaussian']['test_metrics']['precision']) * 100
    gauss_recall = np.array(eps_data['gaussian']['test_metrics']['recall']) * 100
    gauss_f1 = np.array(eps_data['gaussian']['test_metrics']['f1_score']) * 100

    adv_precision = np.array(eps_data['advanced_laplace']['test_metrics']['precision']) * 100
    adv_recall = np.array(eps_data['advanced_laplace']['test_metrics']['recall']) * 100
    adv_f1 = np.array(eps_data['advanced_laplace']['test_metrics']['f1_score']) * 100

    base_precision = np.array(base['test_metrics']['precision']) * 100
    base_recall = np.array(base['test_metrics']['recall']) * 100
    base_f1 = np.array(base['test_metrics']['f1_score']) * 100

    # Metric selection
    metric_select = st.radio("Select Metric:", ['Precision', 'Recall', 'F1-Score'], horizontal=True)

    if metric_select == 'Precision':
        base_vals, gauss_vals, adv_vals = base_precision, gauss_precision, adv_precision
    elif metric_select == 'Recall':
        base_vals, gauss_vals, adv_vals = base_recall, gauss_recall, adv_recall
    else:
        base_vals, gauss_vals, adv_vals = base_f1, gauss_f1, adv_f1

    # Per-class comparison
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Base Model',
        x=CLASS_NAMES,
        y=base_vals,
        marker_color='#2ca02c'
    ))

    fig.add_trace(go.Bar(
        name='Gaussian',
        x=CLASS_NAMES,
        y=gauss_vals,
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        name='Advanced Laplace',
        x=CLASS_NAMES,
        y=adv_vals,
        marker_color='#ff7f0e'
    ))

    fig.update_layout(
        title=f"Per-Class {metric_select} Comparison (ε={eps_select})",
        xaxis_title="Attack Class",
        yaxis_title=f"{metric_select} (%)",
        barmode='group',
        height=600,
        template='plotly_white',
        xaxis={'tickangle': -45}
    )

    st.plotly_chart(fig, use_container_width=True, key="per_class_performance")

    # Class-wise gap
    st.markdown(f"### Gaussian vs Advanced Laplace Gap per Class")

    gap_vals = gauss_vals - adv_vals

    fig_gap = go.Figure()

    colors = ['#2ca02c' if g > 0 else '#d62728' for g in gap_vals]

    fig_gap.add_trace(go.Bar(
        x=CLASS_NAMES,
        y=gap_vals,
        marker_color=colors,
        text=[f'{g:+.1f}%' for g in gap_vals],
        textposition='outside'
    ))

    fig_gap.update_layout(
        title=f"Gaussian Advantage in {metric_select} per Class (ε={eps_select})",
        xaxis_title="Attack Class",
        yaxis_title=f"{metric_select} Gap (%)",
        height=500,
        template='plotly_white',
        xaxis={'tickangle': -45}
    )

    fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig_gap, use_container_width=True, key="per_class_gap")

# =============================================================================
# PAGE 6: CONFUSION MATRICES
# =============================================================================
elif page == "🎯 Confusion Matrices":
    st.markdown('<div class="sub-header">Confusion Matrix Analysis</div>', unsafe_allow_html=True)

    eps_select = st.selectbox("Select Privacy Budget (ε):", [1, 3, 5, 10], index=1, key='cm_eps')
    mechanism_select = st.radio("Select Mechanism:", 
                                ['Base Model', 'Gaussian', 'Advanced Laplace', 
                                 'Traditional Laplace (L1=3)', 'Traditional Laplace (L1=953)'],
                                horizontal=True)

    # Get confusion matrix
    eps_data = [d for d in detailed if d['target_epsilon'] == eps_select][0]

    if mechanism_select == 'Base Model':
        cm = np.array(base['test_metrics']['confusion_matrix'])
        title_suffix = "No DP"
    elif mechanism_select == 'Gaussian':
        cm = np.array(eps_data['gaussian']['test_metrics']['confusion_matrix'])
        title_suffix = f"ε={eps_select}"
    elif mechanism_select == 'Advanced Laplace':
        cm = np.array(eps_data['advanced_laplace']['test_metrics']['confusion_matrix'])
        title_suffix = f"ε={eps_select}"
    elif mechanism_select == 'Traditional Laplace (L1=3)':
        cm = np.array(trad_same['final_test_metrics']['confusion_matrix'])
        title_suffix = "L1=3, ε=3"
    else:
        cm = np.array(trad_equiv['final_test_metrics']['confusion_matrix'])
        title_suffix = "L1=953, ε=3"

    # Normalize option
    normalize = st.checkbox("Normalize by row (show percentages)", value=True)

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_display = cm_norm
        colorscale = 'Blues'
        text_suffix = '%'
    else:
        cm_display = cm
        colorscale = 'Blues'
        text_suffix = ''

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale=colorscale,
        text=[[f'{val:.1f}{text_suffix}' for val in row] for row in cm_display],
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="%" if normalize else "Count")
    ))

    fig.update_layout(
        title=f"Confusion Matrix: {mechanism_select} ({title_suffix})",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        xaxis={'tickangle': -45, 'side': 'bottom'},
        yaxis={'side': 'left'},
        height=700,
        width=800
    )

    st.plotly_chart(fig, use_container_width=True, key="confusion_matrix")

    # Diagonal accuracy
    diagonal = np.diag(cm)
    row_sums = cm.sum(axis=1)
    per_class_acc = (diagonal / row_sums * 100)

    st.markdown("### Per-Class Accuracy from Confusion Matrix")

    acc_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Correct': diagonal,
        'Total': row_sums,
        'Accuracy (%)': per_class_acc
    })

    st.dataframe(acc_df, width='stretch')

# =============================================================================
# PAGE 7: TRAINING EFFICIENCY
# =============================================================================
elif page == "⏱️ Training Efficiency":
    st.markdown('<div class="sub-header">Training Time & Efficiency Analysis</div>', unsafe_allow_html=True)

    # Extract training times
    times_data = []
    for d in detailed:
        times_data.append({
            'epsilon': d['target_epsilon'],
            'gaussian_time': d['gaussian']['time_seconds'],
            'adv_laplace_time': d['advanced_laplace']['time_seconds']
        })

    times_df = pd.DataFrame(times_data)

    # Training time comparison
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times_df['epsilon'],
        y=times_df['gaussian_time'],
        mode='lines+markers',
        name='Gaussian',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=times_df['epsilon'],
        y=times_df['adv_laplace_time'],
        mode='lines+markers',
        name='Advanced Laplace',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title="Training Time vs Privacy Budget",
        xaxis_title="Privacy Budget (ε)",
        yaxis_title="Training Time (seconds)",
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True, key="training_time_comparison")

    # Efficiency metric: Accuracy / Time
    st.markdown("### Efficiency Metric: Accuracy per Second")

    efficiency_data = []
    for d in detailed:
        efficiency_data.append({
            'epsilon': d['target_epsilon'],
            'gaussian_eff': d['gaussian']['accuracy'] * 100 / d['gaussian']['time_seconds'],
            'adv_laplace_eff': d['advanced_laplace']['accuracy'] * 100 / d['advanced_laplace']['time_seconds']
        })

    eff_df = pd.DataFrame(efficiency_data)

    fig_eff = go.Figure()

    fig_eff.add_trace(go.Bar(
        x=eff_df['epsilon'],
        y=eff_df['gaussian_eff'],
        name='Gaussian',
        marker_color='#1f77b4'
    ))

    fig_eff.add_trace(go.Bar(
        x=eff_df['epsilon'],
        y=eff_df['adv_laplace_eff'],
        name='Advanced Laplace',
        marker_color='#ff7f0e'
    ))

    fig_eff.update_layout(
        title="Training Efficiency (Accuracy % per Second)",
        xaxis_title="Privacy Budget (ε)",
        yaxis_title="Efficiency (% / second)",
        barmode='group',
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig_eff, use_container_width=True, key="training_efficiency")

    # Summary table
    st.markdown("### Training Time Summary")

    summary_time = pd.DataFrame({
        'ε': times_df['epsilon'],
        'Gaussian Time (s)': times_df['gaussian_time'].round(1),
        'Adv Laplace Time (s)': times_df['adv_laplace_time'].round(1),
        'Time Difference (s)': (times_df['gaussian_time'] - times_df['adv_laplace_time']).round(1),
        'Gaussian Efficiency': eff_df['gaussian_eff'].round(3),
        'Adv Laplace Efficiency': eff_df['adv_laplace_eff'].round(3)
    })

    st.dataframe(summary_time, width='stretch')

    st.info("""
    **Note:** Training time differences are minimal and primarily due to:
    - Different noise generation methods (Laplace vs Gaussian)
    - Privacy accounting overhead (RDP vs Advanced Composition)
    - Random variations in GPU scheduling

    **Conclusion:** Both mechanisms have similar computational cost; accuracy is the main differentiator.
    """)

# =============================================================================
# PAGE 8: DETAILED METRICS TABLE
# =============================================================================
elif page == "📋 Detailed Metrics Table":
    st.markdown('<div class="sub-header">Complete Metrics Overview</div>', unsafe_allow_html=True)

    eps_select = st.selectbox("Select Privacy Budget (ε):", [1, 3, 5, 10], index=1, key='table_eps')

    eps_data = [d for d in detailed if d['target_epsilon'] == eps_select][0]

    # Compile all metrics
    metrics_data = []

    for i, class_name in enumerate(CLASS_NAMES):
        metrics_data.append({
            'Class': class_name,
            'Base Precision': f"{base['test_metrics']['precision'][i]*100:.2f}%",
            'Base Recall': f"{base['test_metrics']['recall'][i]*100:.2f}%",
            'Base F1': f"{base['test_metrics']['f1_score'][i]*100:.2f}%",
            'Gaussian Precision': f"{eps_data['gaussian']['test_metrics']['precision'][i]*100:.2f}%",
            'Gaussian Recall': f"{eps_data['gaussian']['test_metrics']['recall'][i]*100:.2f}%",
            'Gaussian F1': f"{eps_data['gaussian']['test_metrics']['f1_score'][i]*100:.2f}%",
            'Adv Laplace Precision': f"{eps_data['advanced_laplace']['test_metrics']['precision'][i]*100:.2f}%",
            'Adv Laplace Recall': f"{eps_data['advanced_laplace']['test_metrics']['recall'][i]*100:.2f}%",
            'Adv Laplace F1': f"{eps_data['advanced_laplace']['test_metrics']['f1_score'][i]*100:.2f}%"
        })

    metrics_df = pd.DataFrame(metrics_data)

    st.dataframe(metrics_df, width='stretch', height=600)

    # Download button
    csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"detailed_metrics_eps{eps_select}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Aggregate metrics
    st.markdown("### Aggregate Metrics Summary")

    agg_data = {
        'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 
                   'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
        'Base Model': [
            f"{base['test_metrics']['accuracy']*100:.2f}%",
            f"{base['test_metrics']['macro_precision']*100:.2f}%",
            f"{base['test_metrics']['macro_recall']*100:.2f}%",
            f"{base['test_metrics']['macro_f1']*100:.2f}%",
            f"{base['test_metrics']['weighted_precision']*100:.2f}%",
            f"{base['test_metrics']['weighted_recall']*100:.2f}%",
            f"{base['test_metrics']['weighted_f1']*100:.2f}%"
        ],
        'Gaussian': [
            f"{eps_data['gaussian']['accuracy']*100:.2f}%",
            f"{eps_data['gaussian']['test_metrics']['macro_precision']*100:.2f}%",
            f"{eps_data['gaussian']['test_metrics']['macro_recall']*100:.2f}%",
            f"{eps_data['gaussian']['test_metrics']['macro_f1']*100:.2f}%",
            f"{eps_data['gaussian']['test_metrics']['weighted_precision']*100:.2f}%",
            f"{eps_data['gaussian']['test_metrics']['weighted_recall']*100:.2f}%",
            f"{eps_data['gaussian']['test_metrics']['weighted_f1']*100:.2f}%"
        ],
        'Advanced Laplace': [
            f"{eps_data['advanced_laplace']['accuracy']*100:.2f}%",
            f"{eps_data['advanced_laplace']['test_metrics']['macro_precision']*100:.2f}%",
            f"{eps_data['advanced_laplace']['test_metrics']['macro_recall']*100:.2f}%",
            f"{eps_data['advanced_laplace']['test_metrics']['macro_f1']*100:.2f}%",
            f"{eps_data['advanced_laplace']['test_metrics']['weighted_precision']*100:.2f}%",
            f"{eps_data['advanced_laplace']['test_metrics']['weighted_recall']*100:.2f}%",
            f"{eps_data['advanced_laplace']['test_metrics']['weighted_f1']*100:.2f}%"
        ],
        'Trad Laplace (L1=3)': [
            f"{trad_same['final_test_metrics']['accuracy']*100:.2f}%",
            f"{trad_same['final_test_metrics']['macro_precision']*100:.2f}%",
            f"{trad_same['final_test_metrics']['macro_recall']*100:.2f}%",
            f"{trad_same['final_test_metrics']['macro_f1']*100:.2f}%",
            f"{trad_same['final_test_metrics']['weighted_precision']*100:.2f}%",
            f"{trad_same['final_test_metrics']['weighted_recall']*100:.2f}%",
            f"{trad_same['final_test_metrics']['weighted_f1']*100:.2f}%"
        ],
        'Trad Laplace (L1=953)': [
            f"{trad_equiv['final_test_metrics']['accuracy']*100:.2f}%",
            f"{trad_equiv['final_test_metrics']['macro_precision']*100:.2f}%",
            f"{trad_equiv['final_test_metrics']['macro_recall']*100:.2f}%",
            f"{trad_equiv['final_test_metrics']['macro_f1']*100:.2f}%",
            f"{trad_equiv['final_test_metrics']['weighted_precision']*100:.2f}%",
            f"{trad_equiv['final_test_metrics']['weighted_recall']*100:.2f}%",
            f"{trad_equiv['final_test_metrics']['weighted_f1']*100:.2f}%"
        ]
    }

    agg_df = pd.DataFrame(agg_data)
    st.dataframe(agg_df, width='stretch')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Differential Privacy Mechanisms Comparison Dashboard</b></p>
    <p>Network: 35→256→256→15 | Dataset: ML-Edge_IIoT (15 classes)</p>
    <p>Mechanisms: Traditional Laplace (Pure ε-DP) | Advanced Laplace (Advanced Composition) | Gaussian (RDP)</p>
</div>
""", unsafe_allow_html=True)
