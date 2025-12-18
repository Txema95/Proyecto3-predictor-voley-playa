from config import BCN, DATE_START, DATE_END
from downloadData import download_data_from_meteostat


def main():
    """
        Uso de la aplicación para predecir el clima en Barcelona para poder hacer deporte (voley playa).
        
        -Usar def para descargar datos de meteostat.
    """
    
    download_data_from_meteostat(BCN, DATE_START, DATE_END)

if __name__ == "__main__":
    main()
