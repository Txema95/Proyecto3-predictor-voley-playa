import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def ejecutar(df):
    """
    Modelo de Random Forest para predecir temperatura
    Utiliza datos históricos de días previos para mejorar predicciones
    """
    
    # 1. PREPARACIÓN DE DATOS HISTÓRICOS
    df_model = df.copy()
    df_model = df_model.sort_values('time').reset_index(drop=True)
    
    # Features de días previos (lag features)
    df_model['temp_ayer'] = df_model['temp'].shift(24)
    df_model['temp_2dias'] = df_model['temp'].shift(48)
    df_model['temp_3dias'] = df_model['temp'].shift(72)
    
    # Promedios móviles
    df_model['temp_media_24h'] = df_model['temp'].shift(1).rolling(window=24).mean()
    df_model['temp_media_48h'] = df_model['temp'].shift(1).rolling(window=48).mean()
    df_model['temp_media_72h'] = df_model['temp'].shift(1).rolling(window=72).mean()
    
    # Desviación estándar
    df_model['temp_std_24h'] = df_model['temp'].shift(1).rolling(window=24).std()
    
    # Máximos y mínimos recientes
    df_model['temp_max_24h'] = df_model['temp'].shift(1).rolling(window=24).max()
    df_model['temp_min_24h'] = df_model['temp'].shift(1).rolling(window=24).min()
    
    # Variables meteorológicas previas
    df_model['dwpt_ayer'] = df_model['dwpt'].shift(24)
    df_model['rhum_ayer'] = df_model['rhum'].shift(24)
    df_model['pres_ayer'] = df_model['pres'].shift(24)
    df_model['prcp_ayer'] = df_model['prcp'].shift(24)
    df_model['wspd_ayer'] = df_model['wspd'].shift(24)
    df_model['pres_media_24h'] = df_model['pres'].shift(1).rolling(window=24).mean()
    
    # Componentes temporales
    df_model['hour'] = pd.to_datetime(df_model['time']).dt.hour
    df_model['month'] = pd.to_datetime(df_model['time']).dt.month
    df_model['day_of_year'] = pd.to_datetime(df_model['time']).dt.dayofyear
    
    df_model = df_model.dropna()
    
    st.write(f"### Dataset preparado: {len(df_model)} registros")
    
    # 2. SELECCIÓN DE FEATURES
    features = [
        'dwpt', 'rhum', 'pres', 'prcp', 'wspd', 'wdir', 'wpgt', 'coco',
        'temp_ayer', 'temp_2dias', 'temp_3dias',
        'temp_media_24h', 'temp_media_48h', 'temp_media_72h',
        'temp_std_24h', 'temp_max_24h', 'temp_min_24h',
        'dwpt_ayer', 'rhum_ayer', 'pres_ayer', 'prcp_ayer', 'wspd_ayer',
        'pres_media_24h', 'hour', 'month', 'day_of_year'
    ]
    
    X = df_model[features]
    y = df_model['temp']
    
    # 3. DIVISIÓN TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 4. ENTRENAMIENTO
    st.write("\n### Entrenando Random Forest...")
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # 5. PREDICCIONES
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # 6. MÉTRICAS
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    # VISUALIZACIONES
    
    st.write("\n## Visualización de Resultados")
    
    # --- GRÁFICO 1: Métricas de Rendimiento ---
    st.write("### Métricas del Modelo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE (Test)", f"{mae_test:.2f}°C", 
                 delta=f"{mae_test - mae_train:.2f}°C vs Train")
    with col2:
        st.metric("RMSE (Test)", f"{rmse_test:.2f}°C",
                 delta=f"{rmse_test - rmse_train:.2f}°C vs Train")
    with col3:
        st.metric("R² (Test)", f"{r2_test:.4f}",
                 delta=f"{r2_test - r2_train:.4f} vs Train")
    
    # --- GRÁFICO 2: Predicciones vs Valores Reales ---
    st.write("\n### Predicciones vs Valores Reales")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tomamos una muestra para mejor visualización
    n_samples = min(500, len(y_test))
    indices = range(n_samples)
    
    ax.plot(indices, y_test.values[:n_samples], 
            label='Temperatura Real', color='#2E86AB', linewidth=2, alpha=0.8)
    ax.plot(indices, y_pred_test[:n_samples], 
            label='Temperatura Predicha', color='#A23B72', linewidth=2, alpha=0.8)
    
    ax.fill_between(indices, y_test.values[:n_samples], y_pred_test[:n_samples], 
                     alpha=0.3, color='gray', label='Error')
    
    ax.set_xlabel('Muestra', fontsize=12)
    ax.set_ylabel('Temperatura (°C)', fontsize=12)
    ax.set_title('Comparación: Temperatura Real vs Predicha', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # --- GRÁFICO 3: Scatter Plot con Línea de Regresión ---
    st.write("\n### Diagrama de Dispersión")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='#06A77D', edgecolors='black', linewidth=0.5)
    
    # Línea de predicción perfecta
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Predicción Perfecta')
    
    # Línea de regresión
    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)
    ax.plot(y_test, p(y_test), 
            'b-', linewidth=2, alpha=0.8, label=f'Regresión: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Temperatura Real (°C)', fontsize=12)
    ax.set_ylabel('Temperatura Predicha (°C)', fontsize=12)
    ax.set_title(f'Scatter Plot - R² = {r2_test:.4f}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # --- GRÁFICO 4: Distribución de Errores ---
    st.write("\n### Distribución de Errores")
    
    errores = y_test.values - y_pred_test
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    axes[0].hist(errores, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    axes[0].axvline(errores.mean(), color='green', linestyle='-', linewidth=2, 
                    label=f'Media = {errores.mean():.2f}°C')
    axes[0].set_xlabel('Error (°C)', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_title('Histograma de Errores', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q Plot para normalidad
    stats.probplot(errores, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normalidad de Errores)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # --- ESTADÍSTICAS DE ERRORES ---
    st.write("\n### Estadísticas de Errores")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Error Medio", f"{errores.mean():.3f}°C")
    with col2:
        st.metric("Error Mediano", f"{np.median(errores):.3f}°C")
    with col3:
        st.metric("Desv. Estándar", f"{errores.std():.3f}°C")
    with col4:
        # Test de normalidad Shapiro-Wilk
        _, p_value = stats.shapiro(errores[:5000] if len(errores) > 5000 else errores)
        st.metric("P-value (Shapiro)", f"{p_value:.4f}")
    
    # --- GRÁFICO 5: Importancia de Features ---
    st.write("\n### Top 15 Features Más Importantes")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
    
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title('Features Más Importantes del Modelo', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    st.pyplot(fig)
    plt.close()
    
    # --- GRÁFICO 6: Distribución de Errores por Rango ---
    st.write("\n### Distribución de Errores Absolutos")
    
    error_ranges = pd.cut(np.abs(errores), 
                          bins=[0, 1, 2, 3, 5, np.inf],
                          labels=['<1°C', '1-2°C', '2-3°C', '3-5°C', '>5°C'])
    
    error_counts = error_ranges.value_counts().sort_index()
    error_pcts = (error_counts / len(errores) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(error_counts.index, error_counts.values, 
                  color=['#06D6A0', '#118AB2', '#EF476F', '#FFD166', '#FF6B6B'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Rango de Error', fontsize=12)
    ax.set_ylabel('Número de Predicciones', fontsize=12)
    ax.set_title('Distribución de Errores Absolutos por Rango', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir porcentajes
    for i, (bar, pct) in enumerate(zip(bars, error_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()
    
    # --- TABLA DE P-VALUES DE CORRELACIÓN ---
    st.write("\n### P-values de Correlación con Temperatura")
    
    p_values = []
    correlations = []
    
    for feat in features:
        if feat in X_test.columns:
            corr, p_val = stats.pearsonr(X_test[feat], y_test)
            p_values.append(p_val)
            correlations.append(corr)
    
    corr_df = pd.DataFrame({
        'Feature': features,
        'Correlación': correlations,
        'P-value': p_values,
        'Significativo': ['✓' if p < 0.05 else '✗' for p in p_values]
    }).sort_values('P-value')
    
    st.dataframe(corr_df.head(20).style.background_gradient(subset=['P-value'], cmap='RdYlGn_r'))
    
    # Resumen final
    st.write("\n### Resumen")
    st.write(f"""
    - El modelo predice la temperatura promedio diaria con un error promedio de **{mae_test:.2f}°C**
    - El **{error_pcts.iloc[0]:.1f}%** de las predicciones tienen un error menor a 0.5°C
    - El **{error_pcts.iloc[:2].sum():.1f}%** de las predicciones tienen un error menor a 1°C
    - R² de **{r2_test:.4f}** indica un {'excelente' if r2_test > 0.9 else 'buen' if r2_test > 0.8 else 'aceptable'} ajuste
    """)
    
    return {
        'model': rf,
        'features': features,
        'metrics': {
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test
        },
        'X_test': X_test,
        'y_test': y_test,
        'predictions': y_pred_test
    }