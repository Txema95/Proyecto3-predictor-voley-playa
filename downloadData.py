from datetime import datetime
from meteostat import Hourly
import pandas as pd

def download_data_from_meteostat(point, date_start, date_end):
    """
        Descargar datos horarios de Meteostat para un punto y rango de fechas dados.

        -Obtiene datos de meteostat.
        -Limpia datos marcando campos a cero o null.
        -Crea interpolaciones para huecos pequeños en las columnas wspd, pres.
        -Crea columnas target_lluvia, parts_of_day, avg_temp_parts_of_day, max_temp_parts_of_day, min_temp_parts_of_day, avg_temp_24, max_temp_24, min_temp_24.
        -Exporta los datos a un archivo CSV.
        
        Args:
            point (_type_): _description_
            date_start (_type_): _description_
            date_end (_type_): _description_
    """
    print("Iniciando descarga de datos para Barcelona...")

    # 3. Obtener datos
    data = Hourly(point, date_start, date_end)
    df = data.fetch()

    # 4. Limpieza técnica
    df['prcp'] = df['prcp'].fillna(0)
    df['wspd'] = df['wspd'].interpolate()
    df['pres'] = df['pres'].interpolate()
    df['temp'] = df['temp'].interpolate()

    # 5. Crear etiquetas (Targets)
    # IMPORTANTE: Movemos el reset_index después de los cálculos de tiempo 
    # o usamos la columna generada.
    df = df.reset_index()
    
    # Creamos columnas auxiliares de tiempo
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour

    df['target_lluvia'] = (df['prcp'] > 0.1).astype(int)

    def get_part_of_day(hour):
        if 0 <= hour < 6:
            return 'madrugada'
        elif 6 <= hour < 12:
            return 'manana'
        elif 12 <= hour < 20:
            return 'tarde'
        else:
            return 'noche'

    # CORRECCIÓN 1: Usamos la columna 'hour' que acabamos de crear, no el index
    df['parts_of_day'] = df['hour'].apply(get_part_of_day)

    # CORRECCIÓN 2: La columna 'date' ya existe (línea 25), no necesitamos sacarla del index otra vez
    # df['date'] = df.index.date  <-- Esta línea causaba error y es redundante

    temp_parts = (
        df.groupby(['date', 'parts_of_day'])['temp']
        .agg(
            avg_temp_parts_of_day='mean',
            max_temp_parts_of_day='max',
            min_temp_parts_of_day='min'
        )
        .reset_index()
    )

    df = df.merge(
        temp_parts,
        on=['date', 'parts_of_day'],
        how='left'
    )

    temp_daily = (
        df.groupby('date')['temp']
        .agg(
            avg_temp_24='mean',
            max_temp_24='max',
            min_temp_24='min'
        )
        .reset_index()
    )

    df = df.merge(
        temp_daily,
        on='date',
        how='left'
    )

    # 6. Exportar a CSV
    df.to_csv('clima_barcelona_10anos.csv') # Asegúrate de que el nombre coincida con viewData.py

    print(f"Descarga completada. Se han guardado {len(df)} registros.")