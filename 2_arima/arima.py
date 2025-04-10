
FILE_NAME = "grafcan_cuesta_features.csv"
DATASET_PATH = "../1_tratamiento_datos/processed_data/" + FILE_NAME


model_temp = ARIMA(df["air_temperature"], order=(30,1,0))
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