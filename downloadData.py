from datetime import datetime
from meteostat import Hourly
import pandas as pd

def download_data_from_meteostat(point, date_start, date_end):
    """
        Descargar datos horarios de Meteostat para un punto y rango de fechas dados.

        -Obtiene datos de meteostat.
        -Limpia datos marcando campos a cero o null.
        -Crea interpolaciones para huecos pequeños en las columnas wspd, pres.
        -Crea una columna target_lluvia: 1 si llueve más de 0.1mm, 0 si no.
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

    # 4. Limpieza técnica para el modelo predictivo
    # Si no hay registro de lluvia, asumimos 0
    df['prcp'] = df['prcp'].fillna(0)

    # Para el viento (wspd) y presión (pres), rellenamos huecos pequeños por interpolación
    # Esto evita que el modelo falle por celdas vacías
    df['wspd'] = df['wspd'].interpolate()
    df['pres'] = df['pres'].interpolate()
    df['temp'] = df['temp'].interpolate()

    # 5. Crear etiquetas (Targets)
    # 'target_lluvia': 1 si llueve más de 0.1mm, 0 si no.
    df['target_lluvia'] = (df['prcp'] > 0.1).astype(int)

    # 6. Exportar a CSV
    df.to_csv('clima_barcelona_5anos.csv')

    print(f"Descarga completada. Se han guardado {len(df)} registros.")
    print("Archivo: clima_barcelona_5anos.csv")
