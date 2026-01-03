import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from dataTransform import ClimateDataTransformer, compare_distributions, get_optimal_transformation
from scipy.stats import skew, kurtosis

def viewDataTransform():
    """
    Vista de Streamlit para transformación y visualización de datos.
    """
    st.markdown('<div class="main-header">Data Transformation & Scaling</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Esta herramienta te permite **transformar y escalar** los datos climáticos para:
    - Mejorar la visualización de distribuciones sesgadas
    - Normalizar variables con diferentes escalas
    - Preparar datos para modelos de machine learning
    """)
    
    st.markdown("---")
    
    # Cargar datos
    csv_path = "clima_barcelona_10anos.csv"
    
    if not Path(csv_path).exists():
        st.error(f"File not found: {csv_path}")
        st.info("Please download the data first.")
        return
    
    df = pd.read_csv(csv_path, parse_dates=['time'])
    
    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quick Transform",
        "Custom Transform",
        "Compare Distributions",
        "Climate Features"
    ])
    
    # ==================== TAB 1: QUICK TRANSFORM ====================
    with tab1:
        st.markdown("### Apply Recommended Transformations")
        st.info("Aplica automáticamente las mejores transformaciones para cada variable climática")
        
        if st.button("✨ Apply All Recommended Transformations", type="primary", use_container_width=True):
            with st.spinner("Transforming data..."):
                transformer = ClimateDataTransformer(df)
                transformer.apply_recommended_transformations()
                transformed_df = transformer.get_transformed_data()
                
                # Guardar en session state
                st.session_state['transformed_df'] = transformed_df
                st.session_state['transformer'] = transformer
                
                st.success("Transformations applied successfully!")
                
                # Mostrar resumen
                st.markdown("#### Transformation Summary")
                summary = transformer.get_transformation_summary()
                st.dataframe(summary, use_container_width=True, hide_index=True)
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Columns", len(df.columns))
                with col2:
                    st.metric("Transformed Columns", len(transformed_df.columns))
                with col3:
                    st.metric("New Features", len(transformed_df.columns) - len(df.columns))
        
        # Mostrar recomendaciones
        st.markdown("---")
        st.markdown("#### Recommended Transformations by Variable")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['target_lluvia']]
        
        recommendations = []
        for col in numeric_cols:
            recommendation = get_optimal_transformation(df, col)
            skewness = abs(skew(df[col].dropna()))
            recommendations.append({
                'Variable': col,
                'Skewness': f"{skewness:.2f}",
                'Recommended': recommendation
            })
        
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 2: CUSTOM TRANSFORM ====================
    with tab2:
        st.markdown("### Custom Transformations")
        
        transformer = ClimateDataTransformer(df)
        
        # Selección de columna
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['target_lluvia']]
        
        selected_col = st.selectbox("Select variable to transform:", numeric_cols)
        
        if selected_col:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Select Transformation Method")
                
                transform_type = st.radio(
                    "Transformation type:",
                    [
                        "Standard Scaling (Z-score)",
                        "MinMax Scaling (0-1)",
                        "Robust Scaling (IQR)",
                        "Log Transform (log1p)",
                        "Square Root",
                        "Box-Cox"
                    ]
                )
                
                if st.button("Apply Transformation", type="primary"):
                    if transform_type == "Standard Scaling (Z-score)":
                        transformer.standard_scale([selected_col])
                    elif transform_type == "MinMax Scaling (0-1)":
                        transformer.minmax_scale([selected_col])
                    elif transform_type == "Robust Scaling (IQR)":
                        transformer.robust_scale([selected_col])
                    elif transform_type == "Log Transform (log1p)":
                        transformer.log_transform([selected_col], method='log1p')
                    elif transform_type == "Square Root":
                        transformer.sqrt_transform([selected_col])
                    elif transform_type == "Box-Cox":
                        transformer.boxcox_transform([selected_col])
                    
                    st.session_state['custom_transformer'] = transformer
                    st.success(f"{transform_type} applied to {selected_col}!")
            
            with col2:
                st.markdown("#### Transformation Guide")
                
                guides = {
                    "Standard Scaling (Z-score)": "**Mejor para:** Variables con distribución normal (temp, pres)\n\n**Fórmula:** (x - μ) / σ",
                    "MinMax Scaling (0-1)": "**Mejor para:** Variables con rango conocido (rhum: 0-100%)\n\n**Fórmula:** (x - min) / (max - min)",
                    "Robust Scaling (IQR)": "**Mejor para:** Variables con outliers (wspd, prcp)\n\n**Fórmula:** (x - median) / IQR",
                    "Log Transform (log1p)": "**Mejor para:** Datos muy sesgados con muchos ceros (prcp, snow)\n\n**Fórmula:** log(1 + x)",
                    "Square Root": "**Mejor para:** Sesgo moderado (wspd, prcp)\n\n**Fórmula:** √x",
                    "Box-Cox": "**Mejor para:** Normalizar distribución\n\n**Fórmula:** (xᵏ - 1) / λ"
                }
                
                st.info(guides.get(transform_type, ""))
    
    # ==================== TAB 3: COMPARE DISTRIBUTIONS ====================
    with tab3:
        st.markdown("### Distribution Comparison")
        
        if 'transformed_df' in st.session_state:
            transformed_df = st.session_state['transformed_df']
            transformer = st.session_state['transformer']
            
            # Seleccionar variable original
            original_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            original_cols = [col for col in original_cols if col not in ['target_lluvia']]
            
            selected_original = st.selectbox("Select original variable:", original_cols)
            
            # Encontrar columnas transformadas correspondientes
            transformed_cols = [col for col in transformed_df.columns if col.startswith(selected_original) and col != selected_original]
            
            if transformed_cols:
                selected_transformed = st.selectbox("Select transformed variable:", transformed_cols)
                
                # Crear visualización comparativa
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Distribution Comparison: {selected_original} vs {selected_transformed}', 
                            fontsize=16, fontweight='bold')
                
                # Original - Histogram
                axes[0, 0].hist(df[selected_original].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
                axes[0, 0].set_title(f'Original: {selected_original}')
                axes[0, 0].set_xlabel('Value')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Transformed - Histogram
                axes[0, 1].hist(transformed_df[selected_transformed].dropna(), bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
                axes[0, 1].set_title(f'Transformed: {selected_transformed}')
                axes[0, 1].set_xlabel('Value')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Original - Box plot
                axes[1, 0].boxplot(df[selected_original].dropna(), vert=True)
                axes[1, 0].set_title(f'Box Plot: {selected_original}')
                axes[1, 0].set_ylabel('Value')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Transformed - Box plot
                axes[1, 1].boxplot(transformed_df[selected_transformed].dropna(), vert=True)
                axes[1, 1].set_title(f'Box Plot: {selected_transformed}')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Estadísticas comparativas
                st.markdown("#### Statistical Comparison")
                
                comparison = compare_distributions(transformed_df, selected_original, selected_transformed)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Distribution**")
                    orig_stats = comparison['original']
                    st.metric("Mean", f"{orig_stats['mean']:.4f}")
                    st.metric("Std Dev", f"{orig_stats['std']:.4f}")
                    st.metric("Skewness", f"{orig_stats['skewness']:.4f}")
                    st.metric("Kurtosis", f"{orig_stats['kurtosis']:.4f}")
                
                with col2:
                    st.markdown("**Transformed Distribution**")
                    trans_stats = comparison['transformed']
                    st.metric("Mean", f"{trans_stats['mean']:.4f}")
                    st.metric("Std Dev", f"{trans_stats['std']:.4f}")
                    st.metric("Skewness", f"{trans_stats['skewness']:.4f}")
                    st.metric("Kurtosis", f"{trans_stats['kurtosis']:.4f}")
                
                # Interpretación
                st.markdown("---")
                st.markdown("#### Interpretation")
                
                orig_skew = abs(comparison['original']['skewness'])
                trans_skew = abs(comparison['transformed']['skewness'])
                
                if trans_skew < orig_skew:
                    improvement = ((orig_skew - trans_skew) / orig_skew) * 100
                    st.success(f"La transformación redujo el sesgo en un {improvement:.1f}% (de {orig_skew:.2f} a {trans_skew:.2f})")
                else:
                    st.warning("La transformación no redujo el sesgo significativamente")
                
                if abs(comparison['transformed']['skewness']) < 0.5:
                    st.success("La distribución transformada es aproximadamente simétrica")
                elif abs(comparison['transformed']['skewness']) < 1:
                    st.info("La distribución transformada tiene sesgo moderado")
                else:
                    st.warning("La distribución transformada aún tiene sesgo significativo")
                
            else:
                st.warning("No hay transformaciones disponibles para esta variable")
        else:
            st.info("Primero aplica las transformaciones en la pestaña 'Quick Transform'")
    
    # ==================== TAB 4: CLIMATE FEATURES ====================
    with tab4:
        st.markdown("### Climate-Specific Features")
        
        st.info("Características derivadas basadas en conocimiento meteorológico")
        
        if st.button("Create Climate Features", type="primary", use_container_width=True):
            with st.spinner("Creating features..."):
                transformer = ClimateDataTransformer(df)
                transformer.create_climate_features()
                feature_df = transformer.get_transformed_data()
                
                st.session_state['feature_df'] = feature_df
                st.success("Climate features created!")
        
        if 'feature_df' in st.session_state:
            feature_df = st.session_state['feature_df']
            
            # Mostrar nuevas características
            st.markdown("#### New Features Created")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'rain_intensity' in feature_df.columns:
                    st.markdown("**Rain Intensity Categories**")
                    rain_counts = feature_df['rain_intensity'].value_counts()
                    st.bar_chart(rain_counts)
                
                if 'heat_index' in feature_df.columns:
                    st.markdown("**Heat Index Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    feature_df['heat_index'].dropna().hist(bins=50, ax=ax, 
                                                            edgecolor='black', alpha=0.7, color='orange')
                    ax.set_xlabel('Heat Index')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Heat Index Distribution')
                    st.pyplot(fig)
            
            with col2:
                if 'wind_category' in feature_df.columns:
                    st.markdown("**Wind Categories (Beaufort)**")
                    wind_counts = feature_df['wind_category'].value_counts()
                    st.bar_chart(wind_counts)
                
                if 'wind_chill' in feature_df.columns:
                    st.markdown("**Wind Chill Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    feature_df['wind_chill'].dropna().hist(bins=50, ax=ax,
                                                            edgecolor='black', alpha=0.7, color='lightblue')
                    ax.set_xlabel('Wind Chill')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Wind Chill Distribution')
                    st.pyplot(fig)
            
            # Opciones de descarga
            st.markdown("---")
            st.markdown("#### Export Transformed Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = feature_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="barcelona_climate_transformed.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.info(f"Total columns: {len(feature_df.columns)}\n\nNew features: {len(feature_df.columns) - len(df.columns)}")
    
    st.markdown("---")
    st.caption("Data Transformation Tool - Barcelona Climate Data")


if __name__ == "__main__":
    viewDataTransform()