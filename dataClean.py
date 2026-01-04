import pandas as pd
import numpy as np
from datetime import datetime

def find_first_complete_row(df, columns_to_check):
    """
    Encuentra la primera fila donde todas las columnas especificadas tienen valores no nulos
    """
    for idx in df.index:
        if df.loc[idx, columns_to_check].notna().all():
            return idx
    return 0

def fill_missing_values(df):
    """
    Rellena valores faltantes usando diferentes estrategias según la columna
    """
    df_filled = df.copy()
    
    # Columnas numéricas que requieren interpolación
    numeric_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
    
    for col in numeric_cols:
        if col in df_filled.columns and df_filled[col].isna().any():
            # Convertir 'date' y 'hour' a datetime si están disponibles
            if 'date' in df_filled.columns and 'hour' in df_filled.columns:
                # Crear índice temporal para mejor interpolación
                df_filled['datetime_temp'] = pd.to_datetime(df_filled['date']) + pd.to_timedelta(df_filled['hour'], unit='h')
                df_filled = df_filled.sort_values('datetime_temp')
                
                # Guardar índice original
                original_index = df_filled.index
                
                # Establecer datetime como índice temporal para interpolación
                df_filled = df_filled.set_index('datetime_temp')
                
                # Estrategia 1: Interpolación temporal
                df_filled[col] = df_filled[col].interpolate(method='time', limit_direction='both')
                
                # Restaurar índice original
                df_filled = df_filled.reset_index()
                
                # Estrategia 2: Rellenar con valores del mismo día/hora de años anteriores
                if df_filled[col].isna().any():
                    for idx in df_filled[df_filled[col].isna()].index:
                        try:
                            date_val = df_filled.loc[idx, 'date']
                            hour_val = df_filled.loc[idx, 'hour']
                            
                            current_date = pd.to_datetime(date_val)
                            month = current_date.month
                            day = current_date.day
                            
                            # Filtrar por mismo mes, día y hora
                            similar_rows = df_filled[
                                (pd.to_datetime(df_filled['date']).dt.month == month) &
                                (pd.to_datetime(df_filled['date']).dt.day == day) &
                                (df_filled['hour'] == hour_val) &
                                (df_filled[col].notna())
                            ]
                            
                            if len(similar_rows) > 0:
                                df_filled.loc[idx, col] = similar_rows[col].median()
                        except:
                            pass
                
                # Limpiar columna temporal
                if 'datetime_temp' in df_filled.columns:
                    df_filled = df_filled.drop('datetime_temp', axis=1)
            else:
                # Si no hay información temporal, usar interpolación lineal simple
                df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both')
            
            # Estrategia 3: Rellenar cualquier valor restante con la mediana
            if df_filled[col].isna().any():
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    
    return df_filled

def clean_clima_data(input_file, output_file):
    """
    Limpia los datos del clima según los criterios especificados
    """
    print(f"Leyendo archivo: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Datos originales: {len(df)} filas, {len(df.columns)} columnas")
    
    # 1. Borrar columnas "snow" y "tsun"
    columns_to_drop = ['snow', 'tsun']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"Columnas eliminadas: {existing_cols_to_drop}")
    
    # 2. Encontrar primera fila con valores completos
    columns_to_check = ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
    existing_cols_to_check = [col for col in columns_to_check if col in df.columns]
    
    first_complete_row = find_first_complete_row(df, existing_cols_to_check)
    print(f"Primera fila completa encontrada en índice: {first_complete_row}")
    
    df = df.iloc[first_complete_row:].reset_index(drop=True)
    print(f"Datos después del filtrado: {len(df)} filas")
    
    # 3. Rellenar valores faltantes
    missing_before = df.isna().sum().sum()
    print(f"Valores faltantes antes de rellenar: {missing_before}")
    
    df = fill_missing_values(df)
    
    missing_after = df.isna().sum().sum()
    print(f"Valores faltantes después de rellenar: {missing_after}")
    
    # Guardar el CSV limpio
    df.to_csv(output_file, index=False)
    print(f"\nArchivo limpio guardado en: {output_file}")
    print(f"Dimensiones finales: {len(df)} filas, {len(df.columns)} columnas")
    
    # Mostrar resumen de columnas con valores faltantes (si quedan)
    missing_summary = df.isna().sum()
    if missing_summary.sum() > 0:
        print("\nColumnas con valores faltantes restantes:")
        print(missing_summary[missing_summary > 0])
    
    return df

