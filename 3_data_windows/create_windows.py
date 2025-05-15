import pandas as pd
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt

# VARIABLES #

# FILES = { "santa_cruz": ["grafcan_santa_cruz_features.csv", "openmeteo_santa_cruz_features.csv"]}
#FILES = {"arona": ["grafcan_arona_features.csv", "openmeteo_arona_features.csv"],}
#FILES = {"la_orotava": ["grafcan_la_orotava_features.csv", "openmeteo_la_orotava_features.csv"],}
#FILES = {"la_laguna": ["grafcan_la_laguna_features.csv", "openmeteo_la_laguna_features.csv"],}
# FILES = {"punta_hidalgo": ["grafcan_punta_hidalgo_features.csv", "openmeteo_punta_hidalgo_features.csv"],}

# FILES = {"arona": ["grafcan_arona_features.csv", "openmeteo_arona_features.csv"],
#         "la_orotava": ["grafcan_la_orotava_features.csv", "openmeteo_la_orotava_features.csv"],
#         # "la_laguna": ["grafcan_la_laguna_features.csv", "openmeteo_la_laguna_features.csv"],
# #          "punta_hidalgo": ["grafcan_punta_hidalgo_features.csv", "openmeteo_punta_hidalgo_features.csv"],
# }

FILES = {
  "arona": ["grafcan_arona_features.csv", "openmeteo_arona_features.csv"],
  "la_laguna": ["grafcan_la_laguna_features.csv","openmeteo_la_laguna_features.csv"],
  "la_orotava": ["grafcan_la_orotava_features.csv", "openmeteo_la_orotava_features.csv"],
  "punta_hidalgo": ["grafcan_punta_hidalgo_features.csv", "openmeteo_punta_hidalgo_features.csv"],
}

# FILES =
DATASETS_PATH = "../1_data_preprocessing/processed_data/"

#PAST_N = 13 # Number of past time steps to use as input
FUTURE_N = 6 # Number of future time steps to predict
STEP = 6 # Number of time steps to skip between each input sequence
TRAIN_SPLIT = 0.9 # Percentage of data to use for training (0.85 = 85% train, 15% test)

USE_COVARIATES = True # Use other features as covariates (ej, temperature and pressure to predict humidity)

NOISE_SAMPLE_RATE = 0 # Probability of adding a new synthetic sample to the training set
NOISE_STD = 0.02 # Standard deviation of the noise to add to the samples

# Output is stored as JSON
OUTPUT_PATH = "f6/paquetes_s6_cov_full_"
#OUTPUT_PATH = "paquetes_s6_augmented.pkl"

##########################
# Random Seed #
np.random.seed(17)
random.seed(17)

####################################################################################################
def df_raw_windows(df, past_features, future_features, target, past_n, future_n, step):
  past_variables_windows = []
  future_variables_windows = []
  y_windows = []
  
  for i in range(0, len(df) - past_n - future_n + 1, step):
    past_variables_windows.append(df.iloc[i:i + past_n][past_features].values)
    future_n_window = df.iloc[past_n + i:past_n + i + future_n][future_features].values
    future_variables_windows.append(future_n_window)
    y_windows.append(df.iloc[past_n + i:past_n+ i + future_n][target].values)
  
  data = {
    "past_variables": past_variables_windows,
    "future_variables": future_variables_windows,
    "y": y_windows
  }
  return data
#####################################################################################################

# Data structure for windows
template = {
  "past_variables": [],
  "future_variables": [],
  "y": [],
}

def create_processed_windows(past_n, future_n, step, train_percent, use_covariates):
  # Data structure #
  targets = ["air_temperature", "relative_humidity", "atmospheric_pressure"]
  train_data, test_data = {}, {}
  for target in targets:
    train_data[target] = {}

  # Load data #
  for i in range(len(targets)):
    # print(f"Loading {FILES[location]} for {targets[i]} in {location}")
    target = targets[i]
    train_data[target] = {key: [] for key in template}
    test_data[target] = {key: [] for key in template}
    
    for location in FILES:
      location_data = [] 
      
      ###############################################
      # Aggregate all the datasets for the location #
      ###############################################
      for dataset in FILES[location]:
        df = pd.read_csv(DATASETS_PATH + dataset, parse_dates=['time']).drop(columns=['artificial_value_flag', 'outlier_flag'])

        if use_covariates: # set target as last feature
          features = ["sin_day", "cos_day", "sin_year", "cos_year", *(targets[:i] + targets[i+1:]), targets[i]]
        else: 
          features = ["sin_day", "cos_day", "sin_year", "cos_year", target]
        future_features = ["sin_day", "cos_day", "sin_year", "cos_year"]
        windows_data = df_raw_windows(df, features, future_features, target, past_n, future_n, step)
        
        # Store data
        location_data.append(windows_data)
        
      total_windows = len(location_data[0]["past_variables"])
      overlap_windows = math.ceil((past_n + future_n)/step) # number of windows that overlap with the current one
      test_indexes = random.sample(range(total_windows), int(total_windows * (1 - train_percent)))
    
      forbidden = set()
      for idx in test_indexes:
          # forbid idx itself and the next `overlap_windows - 1` windows
          for j in range(0, overlap_windows + 1):
              forbidden.add(idx + j)

      # Clip to valid window numbers
      forbidden = sorted(i for i in forbidden if 0 <= i < total_windows)
      train_idxs = sorted(set(range(total_windows)) - set(forbidden))

      train_split = {key: [] for key in template}
      test_split = {key: [] for key in template}
      for dataset in range(len(location_data)):
        past_variables_windows = location_data[dataset]["past_variables"]
        future_variables_windows = location_data[dataset]["future_variables"]
        y_windows = location_data[dataset]["y"]
        # Gather train/test splits
        train_past_variables = np.array([past_variables_windows[i] for i in train_idxs])
        train_future_variables = np.array([future_variables_windows[i] for i in train_idxs])
        train_y = np.array([y_windows[i] for i in train_idxs])

        test_past_variables = np.array([past_variables_windows[i] for i in test_indexes])
        test_future_variables = np.array([future_variables_windows[i] for i in test_indexes])
        test_y = np.array([y_windows[i] for i in test_indexes])
        
        train_split["past_variables"].extend(train_past_variables)
        train_split["future_variables"].extend(train_future_variables)
        train_split["y"].extend(train_y)
        
        test_split["past_variables"].extend(test_past_variables)
        test_split["future_variables"].extend(test_future_variables)
        test_split["y"].extend(test_y)
      
      train_data[target]["past_variables"].extend(train_split["past_variables"])
      train_data[target]["future_variables"].extend(train_split["future_variables"])
      train_data[target]["y"].extend(train_split["y"])
      
      test_data[target]["past_variables"].extend(test_split["past_variables"])
      test_data[target]["future_variables"].extend(test_split["future_variables"])
      test_data[target]["y"].extend(test_split["y"])
        
  # Finished data aggregation #
  print("Data shape:")
  
  for target in targets:
    # Convert to numpy array
    train_data[target]["past_variables"] = np.array(train_data[target]["past_variables"])
    train_data[target]["future_variables"] = np.array(train_data[target]["future_variables"])
    train_data[target]["y"] = np.array(train_data[target]["y"])
    # Test
    test_data[target]["past_variables"] = np.array(test_data[target]["past_variables"])
    test_data[target]["future_variables"] = np.array(test_data[target]["future_variables"])
    test_data[target]["y"] = np.array(test_data[target]["y"])
    print(f"Train {target}: {train_data[target]['past_variables'].shape} samples")
    print(f"TEST {target}: {test_data[target]['past_variables'].shape} samples")

  #################
  # Normalization #
  #################
  TARGET_INDEX = 4 # 0,1,2,3 = sin/cos
  COVARIATES_INDEXES = [5, 6] # we are using 3 features
  if USE_COVARIATES:
    normalization_indexes = [TARGET_INDEX] + COVARIATES_INDEXES
  else:
    normalization_indexes = [TARGET_INDEX] 
  for target in targets:
    # Mean across every window and every timestep
    mean = train_data[target]["past_variables"][:, :, normalization_indexes].mean(axis=(0,1))
    std = train_data[target]["past_variables"][:, :, normalization_indexes].std(axis=(0,1))
    train_data[target]["past_variables"][:, :, normalization_indexes] = (train_data[target]["past_variables"][:, :, normalization_indexes] - mean) / std
    test_data[target]["past_variables"][:, :, normalization_indexes] = (test_data[target]["past_variables"][:, :, normalization_indexes] - mean) / std
    # For target use the data of the last feature, which is the target variable
    train_data[target]["y"] = (train_data[target]["y"] - mean[-1]) / std[-1]
    test_data[target]["y"] = (test_data[target]["y"] - mean[-1]) / std[-1]
    # Save data
    train_data[target]["mean"] = mean
    train_data[target]["std"] = std
  
#####################
# Data Augmentation #
#####################
# Add new samples to train data using noise on the existing data
# for target in targets:
#   new_past, new_future, new_y = [], [], []
#   for i in range(len(train_data[target]["past_variables"])):
#     if random.random() < NOISE_SAMPLE_RATE:
#       # Noise uses normal distribution centered around 0, like the normalized data
#       noise = np.random.normal(0, NOISE_STD, train_data[target]["past_variables"][i][:, normalization_indexes].shape)
#       new_sample = train_data[target]["past_variables"][i].copy()
#       new_sample[:, normalization_indexes] += noise
#       new_past.append(new_sample)
      
#       # Plot signal before and after
#       # plt.plot(train_data[target]["past_variables"][i][:, normalization_indexes], label="original")
#       # plt.plot(new_sample[:, normalization_indexes], label="new sample")
#       # plt.legend()
#       # plt.show()
      
#       # Do the same for y
#       noise = np.random.normal(0, NOISE_STD, train_data[target]["y"][i].shape)
#       new_y.append(train_data[target]["y"][i] + noise)
#       # plt.plot(train_data[target]["y"][i], label="original y")
#       # plt.plot(new_y, label="new sample y")
#       # plt.legend()
#       # plt.show()
#       new_future.append(train_data[target]["future_variables"][i])
      
#   # Add new samples to train data
#   train_data[target]["past_variables"] = np.concatenate((train_data[target]["past_variables"], np.array(new_past)), axis=0)
#   train_data[target]["future_variables"] = np.concatenate((train_data[target]["future_variables"], np.array(new_future)), axis=0)
#   train_data[target]["y"] = np.concatenate((train_data[target]["y"], np.array(new_y)), axis=0)
        


  # print shapes
  print("Data shape after augmentation:")
  for target in targets:
    print(f"Train {target}: {train_data[target]['past_variables'].shape} samples")
    print(f"TEST {target}: {test_data[target]['past_variables'].shape} samples")
    
  # print mean and std
  for target in targets:
    print(f"{target} mean: {train_data[target]['mean']}")
    print(f"{target} std: {train_data[target]['std']}")
    
  # Save train and test data
  out_path = OUTPUT_PATH + "p"+ str(past_n) + ".pkl"
  with open(out_path, "wb") as f:
    pickle.dump({"train": train_data, "test": test_data}, f)
  print(f"Data saved to {out_path}")
  
for past_n in range(17, 18):
  np.random.seed(17)
  random.seed(17)
  print("Generating windows for past_n = ", past_n)
  create_processed_windows(past_n, FUTURE_N, STEP, TRAIN_SPLIT, USE_COVARIATES)
