import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
###################################################


FILE_NAME = "grafcan_la_laguna_features.csv"
DATASET_PATH = "../1_data_preprocessing/processed_data/" + FILE_NAME
df = pd.read_csv(DATASET_PATH, parse_dates=['time'])

TEST_SPLIT_DAY = '2025-02-01'  # Test data starts from this date

ARIMA_P = 4  # Autoregressive order
ARIMA_D = 0  # Integrated order
ARIMA_Q = 0  # Moving average order

FORECAST_STEPS = 3  # Number of steps to forecast

####################
indices = df[df['time'].dt.date == pd.to_datetime(TEST_SPLIT_DAY).date()].index
first_2025_day_index = indices[0]

train = df.iloc[0:first_2025_day_index]["air_temperature"]
test = df.iloc[first_2025_day_index:]["air_temperature"].reset_index(drop=True)
#####################################
model_temp = ARIMA(train, order=(ARIMA_P, ARIMA_D, ARIMA_Q))
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
for t in range(0, len(test)-FORECAST_STEPS+1, FORECAST_STEPS):
    print("iteration", t%FORECAST_STEPS, "of", len(test)//FORECAST_STEPS)
    model = ARIMA(history, order=(ARIMA_P, ARIMA_D, ARIMA_Q))
    model_fit = model.fit()

    output = model_fit.forecast(steps=FORECAST_STEPS)
    predictions.extend(output)
    true_vals.extend(test.iloc[t:t+FORECAST_STEPS])

    # Add next forecast_steps real values to history
    history.extend(test.iloc[t:t+FORECAST_STEPS])

# evaluate forecasts
rmse = math.sqrt(mean_squared_error(true_vals, predictions))

print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test[:24])
plt.plot(predictions[:24], color='red')
plt.show()

plt.plot(test[:72])
plt.plot(predictions[:73], color='red')
plt.show()
