import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series):
    """Realiza la prueba de Dickey-Fuller para verificar si la serie es estacionaria."""
    result = adfuller(series)
    return result[1]  # Devuelve el p-valor

def make_stationary(series, alpha=0.05):
    """
    Aplica diferenciación a la serie hasta que sea estacionaria según la prueba ADF.
    Devuelve la serie transformada y el número de diferencias aplicadas (d).
    """
    d = 0
    while test_stationarity(series) > alpha:  # Si la serie no es estacionaria
        series = series.diff().dropna()  # Aplica la diferenciación
        d += 1
        if len(series) < 2:  # Evitar errores en series muy cortas
            break
    return series, d

# Ejemplo de uso con una serie de ejemplo
if __name__ == "__main__":
    # np.random.seed(42)
    # n = 100
    # time_series = np.cumsum(np.random.randn(n))  # Serie no estacionaria (caminata aleatoria)
    
    FILE_NAME = "grafcan_la_laguna_features.csv"
    DATASET_PATH = "../1_data_preprocessing/processed_data/"  + FILE_NAME
    DATASET = "relative_humidity"
    df = pd.read_csv(DATASET_PATH, parse_dates=['time'])

    
    # Visualización de la serie original
    plt.figure(figsize=(10, 4))
    plt.plot(df[DATASET], label="Serie Original")
    plt.title("Serie Temporal No Estacionaria")
    plt.legend()
    plt.show()
    
    # Aplicar la diferenciación para hacerla estacionaria
    stationary_series, d = make_stationary(df[DATASET])
    
    # Visualización de la serie diferenciada
    plt.figure(figsize=(10, 4))
    plt.plot(stationary_series, label=f"Serie Diferenciada (d={d})")
    plt.title("Serie Temporal Estacionaria")
    plt.legend()
    plt.show()
    
    print(f"Número óptimo de diferencias (d): {d}")
