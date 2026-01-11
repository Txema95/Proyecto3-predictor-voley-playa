import streamlit as st
from viewData import viewDataAnalysis
from styles import apply_custom_styles, init_page_config
from viewDataTransform import viewDataTransform
from modelo.model_predict import execRFHourly as execModelRF,execRFDaily as execModelRFDaily, execRFWind as execModelRFWind, execXGboostWind as execModelXGBoostWind,execTempHours as execModelTempHours, execTempDaily as execModelTempDaily


# Configurar página Streamlit
init_page_config()

apply_custom_styles()

def main():
    """
    Aplicación Streamlit para analizar datos climáticos de Barcelona.
    Descarga datos de meteostat y muestra análisis completo.
    
    """
    # Recorte de datos hasta 31111,2018-07-20 07:00:00
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Data Analysis",
        "Data Transform",
        "Model hourly",
        "Model daily",
        "Model wind",
        "Model XGBoost wind",
        "Model predict temp hourly",
        "Model predict temp daily"
    ])
    
    with tab1:
        viewDataAnalysis()    
    with tab2:
        viewDataTransform()
    with tab3:
        st.write("Ejecutando modelo Random Forest Hourly...")
        #execModelRF() comentado porque se ejecuta solo
    with tab4:
        st.write("Ejecutando modelo Random Forest Daily...")
        #execModelRFDaily()
    with tab5:
        st.write("Ejecutando modelo Random Forest Wind...")
        #execModelRFWind()
    with tab6:
        execModelXGBoostWind()
    with tab7:
        execModelTempHours()
    with tab8:
        execModelTempDaily()

if __name__ == "__main__":
    main()
