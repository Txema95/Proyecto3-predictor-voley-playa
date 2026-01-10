import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def ejecutar(df):
    # 4. SELECCIÓN DE COLUMNAS PARA EL MODELO
    # No incluimos 'time' ni 'target' en las X
    features = ['tavg', 'tmin', 'tmax', 'prcp', 'pres', 
                'month', 'pres_ayer', 'tavg_ayer', 'prcp_ayer', 
                'tavg_media_3d', 'pres_media_3d']

    X = df[features]
    y = df['target']

    # 5. ENTRENAMIENTO
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Usamos 200 árboles para mayor estabilidad
    rf = RandomForestClassifier(n_estimators=200,
                                max_depth=10,
                                random_state=42
                                ,class_weight={0: 1, 1: 10}
                                )
    rf.fit(X_train, y_train)

    # 6. EVALUACIÓN
    # En lugar de .predict(), usamos .predict_proba()
    probabilidades = rf.predict_proba(X_test)[:, 1]

    # Si la probabilidad es > 0.3, decimos que lloverá
    nuevas_preds = (probabilidades > 0.6).astype(int)

    st.write(f"Precisión General: {accuracy_score(y_test, nuevas_preds):.2f}")
    st.write("\nInforme de Clasificación:")
    st.write(classification_report(y_test, nuevas_preds))

    # preds = rf.predict(X_test)
    # st.write(f"Precisión General: {accuracy_score(y_test, preds):.2f}")
    # st.write("\nInforme de Clasificación:")
    # st.write(classification_report(y_test, preds))