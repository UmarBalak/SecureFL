import streamlit as st
from pathlib import Path

# Get the directory where THIS file is located
CURRENT_DIR = Path(__file__).parent

# Page config
# st.set_page_config(
#     page_title="Fully Homomorphic Encryption",
#     page_icon="🔒",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

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

# Main title
st.markdown('<div class="main-header">🔒 Fully Homomorphic Encryption </div>', unsafe_allow_html=True)

tab_list = [
    "⏱️ Time Asymmetry Justification",
    "About"
]

tab_time_asymmetry_justification, tab_about = st.tabs(tab_list)

with tab_time_asymmetry_justification:
    # Read the markdown file
    with open(CURRENT_DIR / 'fhe_time_asymmetry_justification.md', 'r', encoding="utf-8") as f:
        md_content = f.read()

    # Display it with st.markdown
    st.markdown(md_content, unsafe_allow_html=True)

with tab_about:
    st.markdown("Nothing here")