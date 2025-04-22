import pandas as pd
import numpy as np
import pickle
import random
import math

# VARIABLES #
FILES = ["grafcan_arona_features.csv", "grafcan_la_laguna_features.csv", "grafcan_la_orotava_features.csv", 
        "openmeteo_arona_features.csv", "openmeteo_la_laguna_features.csv", "openmeteo_la_orotava_features.csv"]
DATASETS_PATH = "../1_tratamiento_datos/processed_data/"

past_n = 30 # Number of past time steps to use as input
future_n = 3 # Number of future time steps to predict
STEP = 6 # Number of time steps to skip between each input sequence
TRAIN_SPLIT = 0.9 # Percentage of data to use for training (0.85 = 85% train, 15% test)

RANDOM_SAMPLE_RATE = 0.5 # Probability of adding a new synthetic sample to the training set
RANDOM_STD = 0.1 # Standard deviation of the noise to add to the samples

# Output is stored as JSON
OUTPUT_PATH = "paquetes_s6_augmented.pkl"

####################################################################################################
def create_df_windows(df, past_features, future_features, target, past_n, future_n, step, train_split=0.85):
  label_start = past_n
  y = df.iloc[label_start:][target]

  past_variables_windows = []
  future_variables_windows = []
  y_windows = []
  
  for i in range(0, len(df) - past_n - future_n + 1, step):
    past_variables_windows.append(df.iloc[i:i + past_n][past_features].values)
    future_n_window = df.iloc[label_start + i:label_start + i + future_n][future_features].values
    future_variables_windows.append(future_n_window)
    y_windows.append(y.iloc[i:i + future_n].values)
    
  total_windows = len(past_variables_windows)
  overlap_windows = math.ceil((past_n + future_n)/step)
  test_indexes = random.sample(range(total_windows), int(total_windows * (1 - train_split)))
  
  forbidden = set()
  for idx in test_indexes:
      # forbid idx itself and the next `overlap_windows - 1` windows
      for j in range(0, overlap_windows):
          forbidden.add(idx + j)

  # Clip to valid window numbers
  forbidden = sorted(i for i in forbidden if 0 <= i < total_windows)
  train_idxs = sorted(set(range(total_windows)) - set(forbidden))

  # Gather train/test splits
  train_past_variables = [past_variables_windows[i] for i in train_idxs]
  train_future_variables = [future_variables_windows[i] for i in train_idxs]
  train_y = [y_windows[i] for i in train_idxs]

  test_past_variables = [past_variables_windows[i] for i in test_indexes]
  test_future_variables = [future_variables_windows[i] for i in test_indexes]
  test_y = [y_windows[i] for i in test_indexes]
    
  # Convert to numpy arrays and store in dictionary
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
    future_features = ["sin_day", "cos_day", "sin_year", "cos_year"]
    train, test = create_df_windows(df, features, future_features, target, past_n, future_n, STEP, TRAIN_SPLIT)
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
  
#####################
# Data Augmentation #
#####################
# Add new samples to train data using noise on the existing data
for target in targets:
  for i in range(len(train_data[target]["past_variables"])):
    if random.random() < RANDOM_SAMPLE_RATE:
      # Noise uses normal distribution centered around 0, like the normalized data
      noise = np.random.normal(0, RANDOM_STD, train_data[target]["past_variables"][i].shape)
      new_sample = train_data[target]["past_variables"][i] + noise
      # Do the same for y
      noise = np.random.normal(0, RANDOM_STD, train_data[target]["y"][i].shape)
      new_y = train_data[target]["y"][i] + noise
      # Append the new sample to the train data
      train_data[target]["past_variables"] = np.append(train_data[target]["past_variables"], [new_sample], axis=0)
      # Future data is not modified
      train_data[target]["future_variables"] = np.append(train_data[target]["future_variables"], [train_data[target]["future_variables"][i]], axis=0)
      train_data[target]["y"] = np.append(train_data[target]["y"], [new_y], axis=0)

# print shapes
print("Train data shape after augmentation:")
for target in targets:
  print(f"{target}: {train_data[target]['past_variables'].shape} samples")
  print(f"TEST {target}: {test_data[target]['past_variables'].shape} samples")
# Save train and test data
with open(OUTPUT_PATH, "wb") as f:
  pickle.dump({"train": train_data, "test": test_data}, f)
print(f"Data saved to {OUTPUT_PATH}")