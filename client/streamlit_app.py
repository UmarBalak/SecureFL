import streamlit as st
from pathlib import Path

# Get the directory where THIS file (streamlit_app.py) is located
BASE_DIR = Path(__file__).parent

# Must be the first Streamlit command
st.set_page_config(
    page_title="SecureFL",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define navigation pages pointing to your standalone apps
with st.sidebar:
    
    pages = [
        st.Page(
            str(BASE_DIR / "DP_Optimizers/visualize/dp_visualization_app.py"),
            title="Differential Privacy",
            icon="🔒",
            default=True
        ),
        st.Page(
            str(BASE_DIR / "tenseal_fhe/visualize/fhe_visualization_app.py"),
            title="Homomorphic Encryption",
            icon="🔐"
        ),
    ]
    

# Create navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()
