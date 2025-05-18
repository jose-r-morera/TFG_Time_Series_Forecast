import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
###################################################

# FILE = "openmeteo_la_orotava_features.csv"
FILES = [
    "openmeteo_arona_features.csv", "openmeteo_la_orotava_features.csv",
    "openmeteo_la_laguna_features.csv", "openmeteo_punta_hidalgo_features.csv",
    "openmeteo_garachico_features.csv", "openmeteo_santa_cruz_features.csv",
    "grafcan_arona_features.csv", "grafcan_la_orotava_features.csv",
    "grafcan_la_laguna_features.csv", "grafcan_punta_hidalgo_features.csv",
    "grafcan_garachico_features.csv", "grafcan_santa_cruz_features.csv"]
DATASET_PATH = "../1_data_preprocessing/processed_data/"

PLOT = False
TEST_SPLIT_DAY = '2025-02-01'  # Test data starts from this date
DATASET = "atmospheric_pressure"  # atmospheric_pressure or relative_humidity or air_temperature

FORECAST_STEPS = 3  # Number of steps to forecast
# ARIMA parameters
ARIMA_P = 4  # Autoregressive order
ARIMA_D = 0  # Integrated order
ARIMA_Q = 0  # Moving average order

####################
for file in FILES:
    print("Loading data from", file)
    df = pd.read_csv(DATASET_PATH + file, parse_dates=['time'])
    indices = df[df['time'].dt.date == pd.to_datetime(TEST_SPLIT_DAY).date()].index
    first_2025_day_index = indices[0]

    train = df.iloc[0:first_2025_day_index][DATASET]
    test = df.iloc[first_2025_day_index:][DATASET].reset_index(drop=True)
    #####################################
    model_temp = ARIMA(train, order=(ARIMA_P, ARIMA_D, ARIMA_Q))
    model_fit = model_temp.fit()

    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    if PLOT: 
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
        print("iteration", t//FORECAST_STEPS, "of", len(test)//FORECAST_STEPS)
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

    if PLOT: 
        # plot forecasts against actual outcomes
        plt.plot(test[:24])
        plt.plot(predictions[:24], color='red')
        plt.show()

        plt.plot(test[:72])
        plt.plot(predictions[:73], color='red')
        plt.show()
