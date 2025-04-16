import pandas as pd
import numpy as np
import pickle
import random

# VARIABLES #
FILES = ["grafcan_arona_features.csv", "grafcan_la_laguna_features.csv", "grafcan_la_orotava_features.csv", 
        "openmeteo_arona_features.csv", "openmeteo_la_laguna_features.csv", "openmeteo_la_orotava_features.csv"]
DATASETS_PATH = "../1_tratamiento_datos/processed_data/"

past_n = 30 # Number of past time steps to use as input
future_n = 3 # Number of future time steps to predict
STEP = 6 # Number of time steps to skip between each input sequence

# Output is stored as JSON
OUTPUT_PATH = "paquetes_futuro_s6.pkl"

####################################################################################################
def create_df_windows(df, features, target, train_split=0.85):
  label_start = past_n
  y = df.iloc[label_start:][target]

  train_past_variables, test_past_variables = [], []
  train_future_variables, test_future_variables = [], []
  train_y, test_y = [], []
  i = 0
  while (i  < len(df) - past_n - future_n + 1):
    if random.random() < train_split: 
      train_past_variables.append(df.iloc[i:i + past_n][features].values)
      future_n_window = df.iloc[label_start + i:label_start + i + future_n][features].drop(columns=[target]).values
      train_future_variables.append(future_n_window)
      train_y.append(y.iloc[i:i + future_n].values)
      i += STEP
    else: 
      # Test data
      test_past_variables.append(df.iloc[i:i + past_n][features].values)
      future_n_window = df.iloc[label_start + i:label_start + i + future_n][features].drop(columns=[target]).values
      test_future_variables.append(future_n_window)
      test_y.append(y.iloc[i:i + future_n].values)
      # Avoid train / test data leakage
      i += past_n + future_n 
      
  train = {
    "past": np.array(train_past_variables),
    "future": np.array(train_future_variables),
    "y": np.array(train_y)
  }
  test = {
    "past": np.array(test_past_variables),
    "future": np.array(test_future_variables),
    "y": np.array(test_y)
  }
      
  return train, test
#####################################################################################################
  

# Data structure #
targets = ["air_temperature", "relative_humidity", "atmospheric_pressure"]
data = {}
for target in targets:
  data[target] = {}

# Load data #
for file in FILES:
  df = pd.read_csv(DATASETS_PATH + file, parse_dates=['time']).drop(columns=['artificial_value_flag', 'outlier_flag'])
  
  for target in targets:
    features = ["sin_day", "cos_day", "sin_year", "cos_year", target]
    train, test = create_df_windows(df, features, target)
    # Store data
    data[target][file] = {}
    data[target][file]["train"] = train
    data[target][file]["test"] = test

#######################
# Dataset Aggregation #
#######################
train_data = {}
test_data = {}
template = {
      "past_variables": [],
      "future_variables": [],
      "y": [],
    }

for target in targets:
  # Initialize empty (avoid copying because of reference)
  train_data[target] = {key: [] for key in template}
  test_data[target] =  {key: [] for key in template}
  # Test has 10% of the data in each dataset source
  for dataset, samples in data[target].items():
    train_data[target]["past_variables"].extend(samples["train"]["past"])
    train_data[target]["future_variables"].extend(samples["train"]["future"])
    train_data[target]["y"].extend(samples["train"]["y"])
    # Test data
    test_data[target]["past_variables"].extend(samples["test"]["past"])
    test_data[target]["future_variables"].extend(samples["test"]["future"])
    test_data[target]["y"].extend(samples["test"]["y"])

print("Train data shape:")
for target in targets:
  # Convert to numpy array
  train_data[target]["past_variables"] = np.array(train_data[target]["past_variables"])
  train_data[target]["future_variables"] = np.array(train_data[target]["future_variables"])
  train_data[target]["y"] = np.array(train_data[target]["y"])
  # Test
  test_data[target]["past_variables"] = np.array(test_data[target]["past_variables"])
  test_data[target]["future_variables"] = np.array(test_data[target]["future_variables"])
  test_data[target]["y"] = np.array(test_data[target]["y"])
  print(f"{target}: {train_data[target]['past_variables'].shape} samples")
  print(f"TEST {target}: {test_data[target]['past_variables'].shape} samples")

# Normalization
for target in targets:
  # Mean across every window and every timestep
  mean = train_data[target]["past_variables"].mean(axis=(0,1))
  std = train_data[target]["past_variables"].std(axis=(0,1))
  train_data[target]["past_variables"] = (train_data[target]["past_variables"] - mean) / std
  test_data[target]["past_variables"] = (test_data[target]["past_variables"] - mean) / std
  # Last feature is the target variable
  train_data[target]["y"] = (train_data[target]["y"] - mean[-1]) / std[-1]
  test_data[target]["y"] = (test_data[target]["y"] - mean[-1]) / std[-1]
  # Save data
  train_data[target]["mean"] = mean
  train_data[target]["std"] = std

# Save train and test data
with open(OUTPUT_PATH, "wb") as f:
  pickle.dump({"train": train_data, "test": test_data}, f)
print(f"Data saved to {OUTPUT_PATH}")