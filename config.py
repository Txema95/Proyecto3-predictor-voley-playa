from meteostat import Point
from datetime import datetime

# Configuración de Barcelona
# Coordenadas: 41.3851, 2.1734. Altitud aprox: 12m
BCN = Point(41.3851, 2.1734, 12)

# Definir periodo de análisis (5 años hasta el 17 de diciembre de 2024)
DATE_START = datetime(2015, 1, 1)
DATE_END = datetime(2025, 12, 31)
