import pandas as pd
import streamlit as st
from pathlib import Path
from config import BCN, DATE_START, DATE_END
from downloadData import download_data_from_meteostat
from viewData import viewDataAnalysis
from styles import apply_custom_styles, init_page_config
from viewDataTransform import viewDataTransform

# Configurar página Streamlit
init_page_config()

apply_custom_styles()

def main():
    """
    Aplicación Streamlit para analizar datos climáticos de Barcelona.
    Descarga datos de meteostat y muestra análisis completo.
    """
    
    tab1, tab2 = st.tabs([
        "Data Analysis",
        "Data Transform"
    ])
    
    with tab1:
        viewDataAnalysis()
    
    with tab2:
        viewDataTransform()


if __name__ == "__main__":
    main()
