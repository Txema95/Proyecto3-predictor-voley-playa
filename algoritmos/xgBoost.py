


import numpy as np
import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



def ejecutar(df):

    # 4. SELECCIÓN DE COLUMNAS PARA EL MODELO
    # No incluimos 'time' ni 'target_wspd' en las X
    features = ['tavg', 'prcp', 'wspd', 'pres', 'wspd_lag1', 'pres_diff', 'day_of_year', 'day_sin', 'day_cos','t_diff','pres_lag2']
    df = df.dropna()
    X = df[features]
    y = df['target_wspd']

    print("Split cronológico del dataset para entrenamiento y prueba")
    # 3. Split cronológico (80% tren, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Configuramos el modelo para que sea agresivo con los errores
    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01, # Paso lento para no sobreajustar
        max_depth=7,        # Un poco de profundidad para capturar ráfagas
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    st.write(f"Árboles óptimos: {model_xgb.best_iteration}")
    
    preds = model_xgb.predict(X_test)
    
    # Métricas
    st.write(mean_absolute_error(y_test, preds))
    st.write(np.sqrt(mean_squared_error(y_test, preds)))
