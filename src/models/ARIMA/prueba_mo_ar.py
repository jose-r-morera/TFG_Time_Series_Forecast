#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AR component test
===========

Test to determine the best AR order using PACF.

Example:
        $ python file.py

"""

__author__ = "José Ramón Morera Campos"
__version__ = "1.0.1"
#######################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

def determinar_orden_ar_pacf(serie, max_lags=20, alpha=0.05):
    """
    Determina el mejor orden AR usando la función de autocorrelación parcial (PACF).
    
    Parámetros:
    serie (array-like): Serie temporal.
    max_lags (int): Máximo número de retardos a considerar.
    alpha (float): Nivel de significancia para las bandas de confianza en la PACF.
    
    Retorna:
    int: El mejor orden AR basado en la PACF.
    """
    # Graficar la PACF
    plot_pacf(serie, lags=max_lags, alpha=alpha, method='ywm')
    plt.title('Función de Autocorrelación Parcial (PACF)')
    plt.show()

    # Calcular la PACF hasta max_lags retardos
    pacf_values = sm.tsa.pacf(serie, nlags=max_lags, method='ywm')
    
    # Encontrar el primer retardo donde la PACF es no significativa
    conf_interval = 1.96 / np.sqrt(len(serie))  # Intervalo de confianza a 95%
    
    for lag in range(1, max_lags + 1):
        print("lag: ", lag, "pacf_values[lag]:", pacf_values[lag], "conf_interval:", conf_interval)
        if abs(pacf_values[lag]) < conf_interval:
            print(f"\nEl mejor orden AR sugerido por la PACF es: {lag-1}")
            return lag-1
    
    # Si todos los retardos son significativos, sugerir el máximo retardo
    print(f"\nTodos los retardos hasta {max_lags} son significativos. Se sugiere el orden AR máximo: {max_lags}")
    return max_lags

# # Ejemplo de uso con una serie temporal simulada
# np.random.seed(123)
# serie = sm.tsa.arma_generate_sample(ar=[1, -0.75, 0.3], ma=[1], nsample=100)  # Simula una serie AR(2)

FILE_NAME = "grafcan_la_laguna_features.csv"
DATASET_PATH = "../1_data_preprocessing/processed_data/" + FILE_NAME
DATASET = "relative_humidity" # atmospheric_pressure or relative_humidity or air_temperature
serie = pd.read_csv(DATASET_PATH, parse_dates=['time'])[DATASET].values

plt.figure(1)
plt.plot(serie)
plt.show()

pacf_values = sm.tsa.pacf(serie, nlags=20, method='ywm')
plt.figure(2)
plt.plot(abs(pacf_values),'*')
plt.show()


# Determinar el mejor orden AR basado en la PACF
mejor_orden_ar = determinar_orden_ar_pacf(serie, max_lags=60)

print("mejor_orden_ar", mejor_orden_ar)