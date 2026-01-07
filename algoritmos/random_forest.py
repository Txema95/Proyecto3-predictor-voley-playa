
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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