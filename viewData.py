import pandas as pd
import streamlit as st
from pathlib import Path
from config import BCN, DATE_START, DATE_END
from downloadData import download_data_from_meteostat
from dataAnalysis import inspect_dataset, summarize_nulls, summarize_zeros, column_statistics

def viewDataAnalysis():
        # Título principal
    st.markdown('<div class="main-header">Climate Data Analyzer - Barcelona Beach Volleyball</div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PASO 1: Descargar datos
    st.markdown('<div class="step-header">Step 1: Download Climate Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info(f"Location: Barcelona | Period: {DATE_START.year} - {DATE_END.year}")
    
    with col2:
        if st.button("Download Data", type="primary", use_container_width=True):
            with st.spinner("Downloading data from meteostat..."):
                download_data_from_meteostat(BCN, DATE_START, DATE_END)
                st.success("Data downloaded successfully!")
    
    st.markdown("---")
    
    # PASO 2: Cargar datos
    st.markdown('<div class="step-header">Step 2: Load Data</div>', unsafe_allow_html=True)
    
    csv_path = "clima_barcelona_10anos.csv"
    
    if not Path(csv_path).exists():
        st.error(f"File not found: {csv_path}")
        st.info("Please download the data first using the button above.")
        return
    
    df = pd.read_csv(csv_path, parse_dates=['time'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Date Range", f"{df['time'].min().year}-{df['time'].max().year}")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.markdown("---")
    
    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "Complete Inspection",
        "Null Values Analysis", 
        "Zero Values Analysis",
        "Column Statistics"
    ])
    
    # TAB 1: Complete Inspection
    with tab1:
        st.markdown('<div class="step-header">Step 3: Complete Dataset Inspection</div>', unsafe_allow_html=True)
        
        # Obtener información de inspección
        inspection_result = inspect_dataset(df, verbose=False)
        
        # Crear tabla con información general
        inspection_data = []
        for col_name, stats in inspection_result.items():
            row = {
                "Column": col_name,
                "Type": stats['dtype'],
                "Non-Null": stats['non_null_count'],
                "Null Count": stats['null_count'],
                "Null %": f"{stats['null_percentage']}%"
            }
            
            if 'mean' in stats:
                row.update({
                    "Mean": stats['mean'],
                    "Min": stats['min'],
                    "Max": stats['max'],
                    "Std": stats['std']
                })
            
            inspection_data.append(row)
        
        inspection_df = pd.DataFrame(inspection_data)
        st.dataframe(inspection_df, use_container_width=True, hide_index=True)
        
        # Estadísticas generales
        st.subheader("General Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            completeness = ((total_cells - null_cells) / total_cells * 100)
            st.metric("Completeness", f"{completeness:.1f}%", delta="of data filled")
        
        with col2:
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_rows)
        
        with col3:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            st.metric("Numeric Columns", len(numeric_cols))
    
    # TAB 2: Null Values Analysis
    with tab2:
        st.markdown('<div class="step-header">Step 4: Null Values Summary</div>', unsafe_allow_html=True)
        
        null_summary = summarize_nulls(df, sort_by="count")
        
        if len(null_summary) > 0:
            st.dataframe(null_summary, use_container_width=True, hide_index=True)
            
            # Gráfico de valores nulos
            st.subheader("Null Values Visualization")
            null_chart = null_summary.copy()
            null_chart = null_chart.sort_values("Null_Count", ascending=True)
            st.bar_chart(null_chart.set_index("Column")["Null_Count"])
        else:
            st.success("No null values found in the dataset!")
    
    # TAB 3: Zero Values Analysis
    with tab3:
        st.markdown('<div class="step-header">Step 5: Zero Values Summary</div>', unsafe_allow_html=True)
        
        zero_summary = summarize_zeros(df, sort_by="count")
        
        if len(zero_summary) > 0:
            st.dataframe(zero_summary, use_container_width=True, hide_index=True)
            
            # Gráfico de valores cero
            st.subheader("Zero Values Visualization")
            zero_chart = zero_summary.copy()
            zero_chart = zero_chart.sort_values("Zero_Count", ascending=True)
            st.bar_chart(zero_chart.set_index("Column")["Zero_Count"])
        else:
            st.success("No zero values found in the dataset!")
    
    # TAB 4: Column Statistics
    with tab4:
        st.markdown('<div class="step-header">Step 6: Detailed Column Statistics</div>', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select a column to view detailed statistics:", numeric_cols)
            
            if selected_col:
                stats = column_statistics(df, selected_col)
                
                # Crear columnas para mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Count", stats['non_null_count'])
                    st.metric("Mean", f"{stats['mean']:.4f}")
                    st.metric("Median", f"{stats['q2']:.4f}")
                
                with col2:
                    st.metric("Null Count", stats['null_count'])
                    st.metric("Zero Count", stats['zero_count'])
                    st.metric("Std Dev", f"{stats['std']:.4f}")
                
                with col3:
                    st.metric("Min", f"{stats['min']:.4f}")
                    st.metric("Max", f"{stats['max']:.4f}")
                    st.metric("Range", f"{stats['range']:.4f}")
                
                # Quartiles
                st.subheader("Quartile Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Q1 (25%)", f"{stats['q1']:.4f}")
                with col2:
                    st.metric("Q2 (50%)", f"{stats['q2']:.4f}")
                with col3:
                    st.metric("Q3 (75%)", f"{stats['q3']:.4f}")
                with col4:
                    st.metric("IQR", f"{stats['iqr']:.4f}")
                
                # Distribución
                st.subheader("Distribution")
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 4))
                df[selected_col].dropna().hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {selected_col}")
                st.pyplot(fig)
        else:
            st.warning("No numeric columns found in the dataset.")
    
    # Footer
    st.markdown("---")
    st.caption("Climate Data Analysis Tool - Barcelona Weather Data Analysis 2005-2025")
