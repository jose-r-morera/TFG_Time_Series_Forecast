import pandas as pd
import numpy as np
import pickle

# VARIABLES #
FILE_NAME = "grafcan_cuesta_features.csv"
DATASET_PATH = "../1_tratamiento_datos/processed_data/" + FILE_NAME

past_n = 30 # Number of past time steps to use as input
future_n = 3 # Number of future time steps to predict
STEP = 6 # Number of time steps to skip between each input sequence

# Output is stored as JSON
OUTPUT_PATH = "paquetes_futuro_s6.pkl"

#############
# Load data #
#############
df = pd.read_csv(DATASET_PATH, parse_dates=['time']).drop(columns=['artificial_value_flag', 'outlier_flag'])

# Test/Train split #
####################
target_day = '2025-01-01'

indices = df[df['time'].dt.date == pd.to_datetime(target_day).date()].index
first_2025_index = indices[0]

###################################################
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return [(data - data_mean) / data_std, data_mean, data_std]
#################################################

no_normalize = ["sin_day", "cos_day", "sin_year", "cos_year"]

features = df.drop(columns=["time", *no_normalize]).astype("float32")

features.index = df["time"]

normalized_features, mean, std = normalize(features.values, first_2025_index)
normalized_features = pd.DataFrame(normalized_features, columns=features.columns)

for col in no_normalize:
    normalized_features[col] = df[col].astype("float32")

train_data = normalized_features.loc[0 : first_2025_index].reset_index(drop=True)
val_data = normalized_features.loc[first_2025_index:].reset_index(drop=True)

# Training data structure #
###########################


def create_dataset(target, features, past_n=1, future_n=1, step=1):
  label_start = past_n
  label_end = label_start + first_2025_index
  
  y_train_full = normalized_features.iloc[label_start:label_end][target]

  x_train = []
  future_train = []
  y_train_multi = []
  for i in range(0, len(train_data) - past_n - future_n + 1, step):
    x_window = train_data.iloc[i:i + past_n][features].values
    x_train.append(x_window)
    future_n_window = normalized_features.iloc[label_start + i:label_start + i + future_n][features].drop(columns=[target]).values
    future_train.append(future_n_window)
    y_window = y_train_full.iloc[i:i + future_n].values
    y_train_multi.append(y_window)
  x_train = np.array(x_train)
  future_train = np.array(future_train)
  y_train_multi = np.array(y_train_multi)

  # Validation data structure #
  #############################
  val_label_start = first_2025_index + past_n

  x_val = []
  future_val = []
  y_val_multi = []
  for i in range(0, len(val_data) - past_n - future_n + 1, step):
    x_window = val_data.iloc[i:i + past_n][features].values
    x_val.append(x_window)
    future_n_window = normalized_features.iloc[val_label_start + i:val_label_start + i + future_n][features].drop(columns=[target]).values
    future_val.append(future_n_window)
    y_window = normalized_features.iloc[val_label_start + i: val_label_start + i + future_n][target].values
    y_val_multi.append(y_window)
  x_val = np.array(x_val)
  future_val = np.array(future_val)
  y_val_multi = np.array(y_val_multi)
  
  return x_train, future_train, y_train_multi, x_val, future_val, y_val_multi

data = {"mean": mean, "std": std} # Save data statistics from normaliation

for target in ["air_temperature", "relative_humidity", "atmospheric_pressure"]:
  features = ["sin_day", "cos_day", "sin_year", "cos_year", target]
  x_train, future_train, y_train_multi, x_val, future_val, y_val_multi = create_dataset(target, features, past_n, future_n, STEP)
  data[target] = {
    "x_train": x_train,
    "future_train": future_train,
    "y_train": y_train_multi,
    "x_val": x_val,
    "future_val": future_val,
    "y_val": y_val_multi
  }


with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(data, f)
