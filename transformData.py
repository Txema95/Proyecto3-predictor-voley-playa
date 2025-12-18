
import pandas as pd
def transform_data():

    # Cargar el archivo que ya descargaste
    df = pd.read_csv('clima_barcelona.csv', index_col=0, parse_dates=True)

    # Calcular la media, min y max de las últimas 24 horas (ventana móvil)
    df['temp_max_24h'] = df['temp'].rolling(window=24).max()
    df['temp_min_24h'] = df['temp'].rolling(window=24).min()
    df['temp_avg_24h'] = df['temp'].rolling(window=24).mean()

    # Guardar de nuevo con las nuevas columnas
    df.to_csv('clima_barcelona_extendido.csv')