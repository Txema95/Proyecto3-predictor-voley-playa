from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd
def main():
    

    # 1. Configurar ubicación: Barcelona (Latitud, Longitud, Altitud)
    # Coordenadas: 41.3851, 2.1734. Altitud aprox: 12m
    bcn = Point(41.3851, 2.1734, 12)

    # 2. Definir periodo (5 años hasta el cierre de 2024 para datos completos)
    inicio = datetime(2019, 1, 1)
    fin = datetime(2025, 12, 17)

    print("Iniciando descarga de datos para Barcelona...")

    # 3. Obtener datos
    data = Hourly(bcn, inicio, fin)
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


if __name__ == "__main__":
    main()
