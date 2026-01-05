
import streamlit as st
import pandas as pd
import algoritmos.algoritmo as algoritmo

def exec():
    st.write("Ejecutando modelo Random Forest...")
    df_ready = pd.read_csv('datos_modelo.csv')
    st.dataframe(df_ready)
    algoritmo.usar_random_forest(df_ready)
