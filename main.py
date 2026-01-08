import streamlit as st
from viewData import viewDataAnalysis
from styles import apply_custom_styles, init_page_config
from viewDataTransform import viewDataTransform
from modelo.model_predict import exec as execModelRF

# Configurar página Streamlit
init_page_config()

apply_custom_styles()

def main():
    """
    Aplicación Streamlit para analizar datos climáticos de Barcelona.
    Descarga datos de meteostat y muestra análisis completo.
    """
    
    tab1, tab2, tab3 = st.tabs([
        "Data Analysis",
        "Data Transform",
        "Model"
    ])
    
    with tab1:
        viewDataAnalysis()
    
    with tab2:
        viewDataTransform()
    with tab3:
        execModelRF()


if __name__ == "__main__":
    main()
