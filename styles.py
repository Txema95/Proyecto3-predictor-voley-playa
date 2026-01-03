import streamlit as st

import streamlit as st

def get_css_styles():
    """Retorna el string de CSS con todos los estilos personalizados."""
    return """
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .step-header {
        color: #ff7f0e;
        font-size: 1.5em;
        font-weight: bold;
        border-left: 4px solid #ff7f0e;
        padding-left: 10px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """

def apply_custom_styles():
    """Aplica los estilos CSS a la página de Streamlit."""
    st.markdown(get_css_styles(), unsafe_allow_html=True)

def init_page_config(title="IA Delivery"):
    """Configura los parámetros básicos de la página."""
    st.set_page_config(
        page_title="Climate Data Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
