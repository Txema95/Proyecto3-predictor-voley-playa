
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


import pandas as pd
import numpy as np
import plotly.graph_objects as go


def ejecutar(df_ready):

    # El target es lo que queremos predecir (mañana)
    y = df_ready['target_24h']

    # Las características (features) son todo lo demás, menos las fechas y el target
    # Asegúrate de quitar 'time', 'date' y cualquier columna de 'target'
    X = df_ready.drop(columns=['target_24h', 'target_lluvia', 'time', 'date', 'Unnamed: 0','Unnamed: 0.1'])
    # Definimos el punto de corte (80% para entrenar)
    split_point = int(len(df_ready) * 0.8)

    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]

    st.write(f"Entrenando con {len(X_train)} horas y probando con {len(X_test)} horas.")
    

    # Creamos el modelo
    # n_estimators=100 significa que usaremos 100 árboles de decisión
    model = RandomForestClassifier(n_estimators=100,
                                   class_weight={0:1,1:10},  # Damos 5 veces mas importancia a la lluvia
                                   random_state=42)

    # Entrenamos (esto puede tardar unos segundos)
    model.fit(X_train, y_train)

    st.write("¡Modelo entrenado!")

    # Obtenemos las probabilidades para el set de prueba
    # [:, 1] nos da la probabilidad de que SEA lluvia (clase 1)
    probabilidades = model.predict_proba(X_test)[:, 1]

    # Veamos las primeras 5 predicciones en porcentaje
    for i in range(5):
        st.write(f"Predicción: {probabilidades[i]*100:.2f}% de probabilidad de lluvia.")

    # Matriz de confusión
    matriz_de_confusion(probabilidades, y_test,df_ready)

def matriz_de_confusion(probabilidades, y_test,df_ready):

    # 1. Convertimos probabilidades en 0 o 1 (umbral del 20%)
    y_pred = (probabilidades > 0.15).astype(int)

    # 2. Creamos la matriz
    cm = confusion_matrix(y_test, y_pred)

    # 3. Creamos la figura usando matplotlib/seaborn (igual que antes)
    # Usamos 'fig, ax = plt.subplots()' que es más limpio para Streamlit
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicción del Modelo (0=Sol, 1=Lluvia)', fontsize=12)
    ax.set_ylabel('Realidad (0=Sol, 1=Lluvia)', fontsize=12)
    ax.set_title('Matriz de Confusión', fontsize=15)

    st.write("Matriz de confusión")
    st.pyplot(fig)

    st.write("Reporte de clasificación:")
    report = classification_report(y_test, y_pred)
    st.text(report)

    st.header("Análisis de Separación: Humedad vs Presión")
    fig_sc, ax_sc = plt.subplots()
    # Dibujamos una muestra de 2000 puntos para no colapsar el gráfico
    sample = df_ready.sample(2000)
    sns.scatterplot(data=sample, x='rhum', y='pres', hue='target_lluvia', alpha=0.5, ax=ax_sc)
    st.pyplot(fig_sc)



def preparar_datasets(df, target_col):     
    #Prepara X, y y divide en train/test
     
    
    # Eliminar filas con NaN en el target o en features críticas
    df_clean = df.dropna(subset=[target_col])
    
    # Seleccionar features (excluir columnas no numéricas y targets)
    exclude_cols = ['time', 'date', 'franja', 'lluvia', 'manana_next_day', 
                    'tarde_next_day', 'prcp', 'snow', 'coco', 'tsun', 'wpgt']
    
    feature_cols = [col for col in df_clean.columns 
                    if col not in exclude_cols and df_clean[col].dtype in ['float64', 'int64']]
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Rellenar NaN restantes con 0 (de los lags y rolling)
    X = X.fillna(0)
    
    # Split temporal: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Dataset preparado para: {target_col}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    print(f"   Proporción lluvia en train: {y_train.mean():.2%}")
    print(f"   Proporción lluvia en test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def entrenar_modelo(X_train, y_train):
    """Entrena Random Forest con hiperparámetros optimizados"""
    
    print("\n🌲 Entrenando Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        class_weight={0: 1, 1: 20},  # Más peso a la clase minoritaria (lluvia)s
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    print("✅ Modelo entrenado")
    
    return rf



# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================

# ============================================================================
# FUNCIÓN DE EVALUACIÓN
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols, target_name):
    """Evalúa el modelo y retorna resultados"""
    
    # Predicciones
    #y_train_pred = model.predict(X_train)
    #y_test_pred = model.predict(X_test)

    
    # Predicciones con threshold personalizado
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Classification Report
    report_train = classification_report(y_train, y_train_pred, 
                                        target_names=['No Lluvia', 'Lluvia'],
                                        output_dict=True)
    
    report_test = classification_report(y_test, y_test_pred, 
                                       target_names=['No Lluvia', 'Lluvia'],
                                       output_dict=True)
    
    # Confusion Matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    return {
        'model': model,
        'report_train': report_train,
        'report_test': report_test,
        'cm_train': cm_train,
        'cm_test': cm_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# ============================================================================
# VISUALIZACIÓN MATRIZ DE CONFUSIÓN
# ============================================================================

def plot_confusion_matrix(cm, title):
    """Matriz de confusión con Plotly"""
    
    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear anotaciones con valores y porcentajes
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)"
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicho: No Lluvia', 'Predicho: Lluvia'],
        y=['Real: No Lluvia', 'Real: Lluvia'],
        colorscale='Blues',
        text=np.array(annotations).reshape(cm.shape),
        texttemplate='%{text}',
        textfont={"size": 14},
        showscale=True,
        colorbar=dict(title="Cantidad")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicción',
        yaxis_title='Valor Real',
        height=400,
        font=dict(size=12)
    )
    
    return fig

# ============================================================================
# VISUALIZACIÓN CLASSIFICATION REPORT
# ============================================================================

def show_classification_report(report, dataset_name):
    """Muestra el classification report en formato tabla"""
    
    # Convertir a DataFrame
    df_report = pd.DataFrame(report).transpose()
    
    # Filtrar solo las clases y métricas agregadas
    df_display = df_report.loc[['No Lluvia', 'Lluvia', 'accuracy', 'macro avg', 'weighted avg']]
    
    # Formatear
    df_styled = df_display.style.format({
        'precision': '{:.3f}',
        'recall': '{:.3f}',
        'f1-score': '{:.3f}',
        'support': '{:.0f}'
    }).background_gradient(
        subset=['precision', 'recall', 'f1-score'], 
        cmap='RdYlGn', 
        vmin=0, 
        vmax=1
    )
    
    st.dataframe(df_styled, use_container_width=True)

    
def show_results(results_manana, results_tarde):
    """Muestra matrices de confusión y classification reports"""
    
    # Crear tabs para cada modelo
    tab1, tab2 = st.tabs(["🌅 Modelo Mañana (8-12h)", "🌆 Modelo Tarde (12-20h)"])
    
    # ========== TAB 1: MODELO MAÑANA ==========
    with tab1:
        st.header("🌅 Evaluación: Franja Mañana (8-12h)")
        
        st.markdown("---")
        
        # TRAIN
        st.subheader("📚 Conjunto de Entrenamiento (Train)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Matriz de Confusión - Train**")
            fig_cm_train = plot_confusion_matrix(
                results_manana['cm_train'],
                "Confusion Matrix - Mañana (Train)"
            )
            st.plotly_chart(fig_cm_train, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Report - Train**")
            show_classification_report(results_manana['report_train'], "Train")
        
        st.markdown("---")
        
        # TEST
        st.subheader("🧪 Conjunto de Prueba (Test)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Matriz de Confusión - Test**")
            fig_cm_test = plot_confusion_matrix(
                results_manana['cm_test'],
                "Confusion Matrix - Mañana (Test)"
            )
            st.plotly_chart(fig_cm_test, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Report - Test**")
            show_classification_report(results_manana['report_test'], "Test")
    
    # ========== TAB 2: MODELO TARDE ==========
    with tab2:
        st.header("🌆 Evaluación: Franja Tarde (12-20h)")
        
        st.markdown("---")
        
        # TRAIN
        st.subheader("📚 Conjunto de Entrenamiento (Train)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Matriz de Confusión - Train**")
            fig_cm_train = plot_confusion_matrix(
                results_tarde['cm_train'],
                "Confusion Matrix - Tarde (Train)"
            )
            st.plotly_chart(fig_cm_train, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Report - Train**")
            show_classification_report(results_tarde['report_train'], "Train")
        
        st.markdown("---")
        
        # TEST
        st.subheader("🧪 Conjunto de Prueba (Test)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Matriz de Confusión - Test**")
            fig_cm_test = plot_confusion_matrix(
                results_tarde['cm_test'],
                "Confusion Matrix - Tarde (Test)"
            )
            st.plotly_chart(fig_cm_test, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Report - Test**")
            show_classification_report(results_tarde['report_test'], "Test")
