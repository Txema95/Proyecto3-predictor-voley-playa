
import streamlit as st
import pandas as pd
import algoritmos.algoritmo as algoritmo

def execRFHourly():
    st.write("Ejecutando modelo Random Forest...")
    df_ready = pd.read_csv('datos_modelo.csv')
    st.dataframe(df_ready)
    algoritmo.usar_random_forest_target_manana(df_ready)

def execRFDaily():
    st.write("Ejecutando modelo Random Forest daily...")
    df_ready = pd.read_csv('datos_modelo.csv')
    st.dataframe(df_ready)
    algoritmo.usar_random_forest_target_manana(df_ready)
