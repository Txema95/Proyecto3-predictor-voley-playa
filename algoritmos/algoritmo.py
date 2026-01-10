
import time
import streamlit as st
from algoritmos import random_forest, random_forest_daily, random_forest_wind, xgBoost

# def usar_fuerza_bruta(df_matriz_distancias):
#     start = time.time()
#     with st.spinner("Calculando rutas...", show_time=True, width="content"):
#         rutas = fuerza_bruta.calcular(df_matriz_distancias)
#     elapsed = time.time() - start
#     st.title(f"Rutas (calculado en {elapsed:.2f} segundos): ")    
#     st.dataframe(rutas)
#     st.title(f"Rutas generadas y guardadas en app/data/rutas.csv")    
#     rutas.to_csv("app/data/rutas.csv", index=False)

def usar_random_forest(df):
    random_forest.ejecutar(df)

def usar_random_forest_target_manana(df):
    X_train_m, X_test_m, y_train_m, y_test_m, features_m = random_forest.preparar_datasets(df, 'manana_next_day')
    model_manana =random_forest.entrenar_modelo(X_train_m, y_train_m)
    results_manana = random_forest.evaluate_model(model_manana, X_train_m, X_test_m, 
                                   y_train_m, y_test_m, features_m, 
                                   "MAÑANA 8-12h")
    
    #entrenar modelo tarde
    X_train_t, X_test_t, y_train_t, y_test_t, features_t = random_forest.preparar_datasets(df, 'tarde_next_day')
    model_tarde = random_forest.entrenar_modelo(X_train_t, y_train_t)
    results_tarde = random_forest.evaluate_model(model_tarde, X_train_t, X_test_t, 
                                  y_train_t, y_test_t, features_t, 
                                  "TARDE 12-20h")
    
    
    random_forest.show_results(results_manana, results_tarde)

def usar_random_forest_daily_target(df):
    random_forest_daily.ejecutar(df)

def usar_random_forest_wind_target(df):
    random_forest_wind.ejecutar(df)
    
def usar_xgboost_wind_target(df):
    xgBoost.ejecutar(df)