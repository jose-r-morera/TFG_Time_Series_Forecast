import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

def determinar_orden_ma_acf(serie, max_lags=20, alpha=0.05, min_conf=0.05):
    """
    Determina el mejor orden MA usando la función de autocorrelación (ACF).
    
    Parámetros:
    serie (array-like): Serie temporal.
    max_lags (int): Máximo número de retardos a considerar.
    alpha (float): Nivel de significancia para las bandas de confianza en la ACF.
    
    Retorna:
    int: El mejor orden MA basado en la ACF.
    """
    # Graficar la ACF
    plot_acf(serie, lags=max_lags, alpha=alpha)
    plt.title('Función de Autocorrelación (ACF)')
    plt.show()

    # Calcular la ACF hasta max_lags retardos
    acf_values = sm.tsa.acf(serie, nlags=max_lags, fft=False)
    
    # Encontrar el primer lag donde la ACF es no significativa
    conf_interval = max(1.96 / np.sqrt(len(serie)), min_conf)
    # conf_interval = 1.96 / np.sqrt(len(serie))  # Intervalo de confianza a 95%
    
    for lag in range(1, max_lags + 1):
        if abs(acf_values[lag]) < conf_interval:
            print(f"\nEl mejor orden MA sugerido por la ACF es: {lag-1}")
            return lag-1
    
    # Si todos los retardos son significativos, sugerir el máximo retardo
    print(f"\nTodos los retardos hasta {max_lags} son significativos. Se sugiere el orden MA máximo: {max_lags}")
    return max_lags

# # Ejemplo de uso con una serie temporal simulada
# np.random.seed(123)
# serie = sm.tsa.arma_generate_sample(ar=[1], ma=[1, 0.75, 0.52], nsample=100)  # Simula una serie MA(2)

FILE_NAME = "grafcan_cuesta_features.csv"
DATASET_PATH = "../1_tratamiento_datos/processed_data/" + FILE_NAME
serie = pd.read_csv(DATASET_PATH, parse_dates=['time'])["air_temperature"].values

plt.figure(1)
plt.plot(serie)
plt.show()

# Determinar el mejor orden MA basado en la ACF
mejor_orden_ma = determinar_orden_ma_acf(serie, max_lags=40, alpha=0.05)

print("mejor_orden_ma", mejor_orden_ma)