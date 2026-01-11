import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def ejecutar(df):
    # 4. SELECCIÓN DE COLUMNAS PARA EL MODELO
    # No incluimos 'time' ni 'target_wspd' en las X
    features = ['tavg', 'prcp', 'wspd', 'pres', 'wspd_lag1', 'pres_diff', 'day_of_year', 'day_sin', 'day_cos']
    df = df.dropna()
    X = df[features]
    y = df['target_wspd']

    print("Split cronológico del dataset para entrenamiento y prueba")
    # 3. Split cronológico (80% tren, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("Entrenando el modelo Random Forest Regressor")
    # 4. Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluando el modelo")
    # 5. Evaluación
    preds = model.predict(X_test)
    st.write(f"MAE: {mean_absolute_error(y_test, preds)}")
    st.write(f"RMSE: {root_mean_squared_error(y_test, preds)}")
    #mostrar_importancia(model, features)
    plot_residuos(y_test, preds)
    

def mostrar_importancia(model, features):
    st.subheader("🎯 Importancia de las Variables")
    
    # Crear DataFrame de importancia
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Variable': features,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=True)

    # Crear gráfico con Plotly
    fig = px.bar(
        feature_df, 
        x='Importancia', 
        y='Variable', 
        orientation='h',
        title="Impacto de cada variable en la predicción del viento",
        labels={'Importancia': 'Peso en el Modelo', 'Variable': 'Feature'},
        color='Importancia',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(showlegend=False, height=450)
    
    # Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Añadir una explicación dinámica
    top_feature = feature_df.iloc[-1]['Variable']
    st.info(f"💡 La variable más influyente es **{top_feature}**. "
            "Esto sugiere que el modelo se apoya principalmente en este dato para predecir el día siguiente.")

def plot_residuos(y_real, y_pred):
    st.subheader("📊 Análisis de Residuos (Errores)")
    
    # Aseguramos que sean series de pandas para evitar problemas de índices
    residuos_df = pd.DataFrame({
        'Real': y_real.values if hasattr(y_real, 'values') else y_real,
        'Predicho': y_pred,
    })
    residuos_df['Residuo'] = residuos_df['Real'] - residuos_df['Predicho']
    residuos_df['Error_Absoluto'] = residuos_df['Residuo'].abs()

    # Gráfico de Dispersión: Real vs Residuo
    fig_scatter = px.scatter(
        residuos_df, 
        x='Real', 
        y='Residuo',
        color='Error_Absoluto',
        color_continuous_scale='Viridis',
        title="Dispersión de Errores (Real vs Residuo)",
        labels={'Real': 'Velocidad Real (km/h)', 'Residuo': 'Error (Real - Predicho)'},
        opacity=0.7
    )
    
    # Línea horizontal en 0 (El objetivo ideal)
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Ideal (Sin error)")
    
    # Mostrar el scatter
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Gráfico 2: Distribución del error (Histograma)
    # Esto es fundamental en IA para ver si el error sigue una distribución normal
    fig_hist = px.histogram(
        residuos_df, 
        x='Residuo', 
        nbins=30,
        title="Distribución de los Errores",
        color_discrete_sequence=['#636EFA'],
        marginal="violin" # Aquí el violin funciona bien por separado
    )
    st.plotly_chart(fig_hist, use_container_width=True)

