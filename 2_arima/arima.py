import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

# ARIMA model
from statsmodels.tsa.arima.model import ARIMA
###################################################


FILE_NAME = "grafcan_cuesta_features.csv"
DATASET_PATH = "../1_tratamiento_datos/processed_data/" + FILE_NAME
df = pd.read_csv(DATASET_PATH, parse_dates=['time'])

target_day = '2025-02-01'
indices = df[df['time'].dt.date == pd.to_datetime(target_day).date()].index
first_2025_day_index = indices[0]

train = df.iloc[0:first_2025_day_index]["air_temperature"]
test = df.iloc[first_2025_day_index:]["air_temperature"].reset_index(drop=True)
#####################################
model_temp = ARIMA(train, order=(4,0,0))
model_fit = model_temp.fit()

# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())
##################################################
history = [x for x in train]
predictions = list()
true_vals = list()

# walk-forward validation
for t in range(0, len(test)-2, 3):
    print("it number: ", t)
    model = ARIMA(history, order=(4, 0, 0))
    model_fit = model.fit()

    output = model_fit.forecast(steps=3)
    predictions.extend(output)
    true_vals.extend(test.iloc[t:t+3])
    
    # Add next 3 real values to history
    history.extend(test.iloc[t:t+3])

    
# evaluate forecasts
import math 
rmse = math.sqrt(mean_squared_error(true_vals, predictions))

print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test[:24])
plt.plot(predictions[:24], color='red')
plt.show()

plt.plot(test[:72])
plt.plot(predictions[:73], color='red')
plt.show()

