import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ejemplo de un array numpy que representa una serie temporal
datos = np.array([120, 125, 130, 128, 140, 150, 148, 145, 155, 160, 158, 162, 170, 165, 175, 180, 178])

# Crear una serie temporal con las fechas correspondientes (opcional)
fechas = pd.date_range(start='2023-01-01', periods=len(datos), freq='D')
serie_temporal = pd.Series(datos, index=fechas)

# Graficar la serie temporal
plt.figure(figsize=(10, 4))
plt.plot(serie_temporal, label='Serie Temporal', color='blue')
plt.title('Serie Temporal')
plt.grid(True)
plt.legend()
plt.show()

# Definir el modelo ARIMA(p, d, q)
# Aquí elegimos (p=2, d=2, q=2) como ejemplo, pero se pueden ajustar los valores según ACF, PACF y calculando el numero de diferenciaciones necesarias para hacer la serie estacionaria
modelo_arima = ARIMA(serie_temporal, order=(2, 2, 2))

# Ajustar el modelo ARIMA a los datos
resultado = modelo_arima.fit()

# Ver lo bueno que es el ajuste
print(resultado.summary())

# Hacer predicciones usando el modelo ajustado
predicciones = resultado.forecast(steps=5)

# Graficar la serie original y las predicciones
plt.figure(figsize=(10, 4))
plt.plot(serie_temporal, label='Serie Original', color='blue')
plt.plot(pd.date_range(start=fechas[-1], periods=6, freq='D')[1:], predicciones, label='Predicciones', color='orange', linestyle='--')
plt.title('Modelo ARIMA: Serie Temporal y Predicciones')
plt.grid(True)
plt.legend()
plt.show()
