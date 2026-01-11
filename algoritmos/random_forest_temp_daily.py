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
    Modelo de Random Forest para predecir temperatura promedio diaria (tavg)
    Utiliza datos históricos de días previos para mejorar predicciones
    """
    
    # 1. PREPARACIÓN DE DATOS
    df_model = df.copy()
    df_model = df_model.sort_values('time').reset_index(drop=True)
    
    st.write(f"### Dataset original: {len(df_model)} días")
    
    # 2. CREAR FEATURES ADICIONALES (si no existen)
    # Ya tenemos muchas features lag en el dataset, pero podemos crear algunas adicionales
    
    # Temperaturas de días anteriores (adicionales)
    if 'tavg_lag_2' not in df_model.columns:
        df_model['tavg_lag_2'] = df_model['tavg'].shift(2)
    if 'tavg_lag_3' not in df_model.columns:
        df_model['tavg_lag_3'] = df_model['tavg'].shift(3)
    if 'tavg_lag_7' not in df_model.columns:
        df_model['tavg_lag_7'] = df_model['tavg'].shift(7)  # Semana anterior
    
    # Rango térmico del día anterior
    if 'tmax_ayer' not in df_model.columns:
        df_model['tmax_ayer'] = df_model['tmax'].shift(1)
    if 'tmin_ayer' not in df_model.columns:
        df_model['tmin_ayer'] = df_model['tmin'].shift(1)
    
    df_model['rango_termico_ayer'] = df_model['tmax_ayer'] - df_model['tmin_ayer']
    
    # Medias móviles adicionales
    if 'tavg_media_7d' not in df_model.columns:
        df_model['tavg_media_7d'] = df_model['tavg'].shift(1).rolling(window=7).mean()
    if 'tavg_media_14d' not in df_model.columns:
        df_model['tavg_media_14d'] = df_model['tavg'].shift(1).rolling(window=14).mean()
    
    # Desviación estándar de temperatura (variabilidad)
    df_model['tavg_std_7d'] = df_model['tavg'].shift(1).rolling(window=7).std()
    
    # Tendencia de temperatura (cambio respecto a hace 3 días)
    df_model['tavg_tendencia_3d'] = df_model['tavg_ayer'] - df_model['tavg_lag_3']
    
    # Presión: tendencia
    if 'pres_lag_2' not in df_model.columns:
        df_model['pres_lag_2'] = df_model['pres'].shift(2)
    df_model['pres_tendencia'] = df_model['pres_ayer'] - df_model['pres_lag_2']
    
    # Precipitación acumulada últimos 3 días
    if 'prcp_sum_3d' not in df_model.columns:
        df_model['prcp_sum_3d'] = df_model['prcp'].shift(1).rolling(window=3).sum()
    
    # Eliminar filas con NaN
    df_model = df_model.dropna()
    
    st.write(f"### Dataset preparado: {len(df_model)} días (después de lags)")
    
    # 3. SELECCIÓN DE FEATURES
    # Usamos las features existentes + las nuevas creadas
    features = [
        # Temperatura histórica
        'tavg_lag_1', 'tavg_ayer',  # Día anterior (pueden ser duplicados, el modelo lo manejará)
        'tavg_lag_2', 'tavg_lag_3', 'tavg_lag_7',
        
        # Estadísticas móviles de temperatura
        'temp_roll_3', 'tavg_media_3d', 'tavg_media_7d', 'tavg_media_14d',
        'tavg_std_7d', 'tavg_tendencia_3d',
        
        # Rango térmico
        'tmax_ayer', 'tmin_ayer', 'rango_termico_ayer',
        
        # Presión
        'pres', 'pres_lag_1', 'pres_ayer', 'pres_lag_2',
        'pres_media_3d', 'pres_tendencia',
        
        # Precipitación
        'prcp', 'prcp_lag_1', 'prcp_ayer',
        'prcp_sum_3d', 'prcp_sum_7',
        
        # Componentes temporales
        'month', 'day_of_year', 'month_sin', 'month_cos'
    ]
    
    # Filtrar solo las features que existen
    features = [f for f in features if f in df_model.columns]
    
    # Variable objetivo: temperatura promedio del día
    X = df_model[features]
    y = df_model['tavg']
    
    st.write(f"**Features utilizadas:** {len(features)}")
    
    # 4. DIVISIÓN TRAIN/TEST (temporal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    st.write(f"- Datos de entrenamiento: {len(X_train)} días")
    st.write(f"- Datos de prueba: {len(X_test)} días")
    
    # 5. ENTRENAMIENTO
    st.write("\n### Entrenando Random Forest...")
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # 6. PREDICCIONES
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # 7. MÉTRICAS
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    # VISUALIZACIONES
    
    st.write("\n## Visualización de Resultados")
    
    # --- MÉTRICAS EN TARJETAS ---
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
    
    # --- GRÁFICO 1: Serie Temporal ---
    st.write("\n### Predicciones vs Valores Reales (Últimos 180 días de Test)")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_samples = min(180, len(y_test))
    indices = range(n_samples)
    
    ax.plot(indices, y_test.values[:n_samples], 
            label='Temperatura Real', color='#2E86AB', linewidth=2.5, marker='o', 
            markersize=4, alpha=0.8)
    ax.plot(indices, y_pred_test[:n_samples], 
            label='Temperatura Predicha', color='#A23B72', linewidth=2.5, marker='s',
            markersize=4, alpha=0.8)
    
    ax.fill_between(indices, y_test.values[:n_samples], y_pred_test[:n_samples], 
                     alpha=0.25, color='gray', label='Error')
    
    ax.set_xlabel('Día', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperatura (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación: Temperatura Real vs Predicha (Datos Diarios)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    st.pyplot(fig)
    plt.close()
    
    # --- GRÁFICO 2: Scatter Plot ---
    st.write("\n### Diagrama de Dispersión")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=40, color='#06A77D', 
               edgecolors='black', linewidth=0.5)
    
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2.5, label='Predicción Perfecta', alpha=0.8)
    
    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)
    ax.plot(sorted(y_test), p(sorted(y_test)), 
            'b-', linewidth=2.5, alpha=0.8, 
            label=f'Regresión: y={z[0]:.3f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Temperatura Real (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperatura Predicha (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Scatter Plot - R² = {r2_test:.4f}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    st.pyplot(fig)
    plt.close()
    
    # --- GRÁFICO 3: Distribución de Errores ---
    st.write("\n### Análisis de Errores")
    
    errores = y_test.values - y_pred_test
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    axes[0].hist(errores, bins=40, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2.5, label='Error = 0')
    axes[0].axvline(errores.mean(), color='green', linestyle='-', linewidth=2.5, 
                    label=f'Media = {errores.mean():.3f}°C')
    axes[0].set_xlabel('Error (°C)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    axes[0].set_title('Histograma de Errores', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Q-Q Plot
    stats.probplot(errores, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normalidad de Errores)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].get_lines()[0].set_markerfacecolor('#06A77D')
    axes[1].get_lines()[0].set_markersize(6)
    
    plt.tight_layout()
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
        _, p_value = stats.shapiro(errores if len(errores) <= 5000 else errores[:5000])
        st.metric("P-value (Shapiro)", f"{p_value:.4f}")
    
    # --- GRÁFICO 4: Importancia de Features ---
    st.write("\n### Top 15 Features Más Importantes")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(feature_importance)))
    bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                   color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_title('Features Más Importantes del Modelo', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.invert_yaxis()
    
    for i, (bar, row) in enumerate(zip(bars, feature_importance.itertuples())):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # --- GRÁFICO 5: Distribución de Errores por Rango ---
    st.write("\n### Distribución de Errores Absolutos")
    
    error_ranges = pd.cut(np.abs(errores), 
                          bins=[0, 0.5, 1, 2, 3, np.inf],
                          labels=['<0.5°C', '0.5-1°C', '1-2°C', '2-3°C', '>3°C'])
    
    error_counts = error_ranges.value_counts().sort_index()
    error_pcts = (error_counts / len(errores) * 100)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    bars = ax.bar(error_counts.index, error_counts.values, 
                  color=['#06D6A0', '#118AB2', '#FFD166', '#EF476F', '#9D4EDD'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Rango de Error', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Días', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Errores Absolutos por Rango', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for i, (bar, pct, count) in enumerate(zip(bars, error_pcts, error_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{int(count)} días\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # --- TABLA DE P-VALUES ---
    st.write("\n### P-values de Correlación con Temperatura")
    
    p_values = []
    correlations = []
    
    for feat in features:
        corr, p_val = stats.pearsonr(X_test[feat], y_test)
        p_values.append(p_val)
        correlations.append(corr)
    
    corr_df = pd.DataFrame({
        'Feature': features,
        'Correlación': correlations,
        'P-value': p_values,
        'Significativo': ['✓ Sí' if p < 0.05 else '✗ No' for p in p_values]
    }).sort_values('P-value')
    
    st.dataframe(
        corr_df.head(20).style.background_gradient(subset=['P-value'], cmap='RdYlGn_r')
        .format({'Correlación': '{:.4f}', 'P-value': '{:.6f}'})
    )
    
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
            'r2_test': r2_test,
            'mae_train': mae_train,
            'rmse_train': rmse_train,
            'r2_train': r2_train
        },
        'X_test': X_test,
        'y_test': y_test,
        'predictions': y_pred_test,
        'feature_importance': feature_importance
    }