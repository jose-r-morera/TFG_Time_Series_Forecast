import pandas as pd
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt

# VARIABLES #

# FILES = { "santa_cruz": ["grafcan_santa_cruz_features.csv"]}
# FILES = { "santa_cruz": ["openmeteo_santa_cruz_features.csv"]}
# FILES = {"garachico": ["grafcan_garachico_features.csv"]}
# FILES = {"garachico": ["openmeteo_garachico_features.csv"]}

# FILES = {"arona": ["grafcan_arona_features.csv"],}
# FILES = {"arona": ["openmeteo_arona_features.csv"]}
# FILES = {"la_orotava": ["grafcan_la_orotava_features.csv"]}
# FILES = {"la_orotava": ["openmeteo_la_orotava_features.csv"],}
# FILES = {"la_laguna": ["grafcan_la_laguna_features.csv"]}
# FILES = {"la_laguna": ["openmeteo_la_laguna_features.csv"],}
# FILES = {"punta_hidalgo": ["grafcan_punta_hidalgo_features.csv"]}
# FILES = {"punta_hidalgo": ["openmeteo_punta_hidalgo_features.csv"],}

DATASETS_PATH = "../1_data_preprocessing/test_data/"

#PAST_N = 13 # Number of past time steps to use as input
FUTURE_N = 3 # Number of future time steps to predict
STEP = 6 # Number of time steps to skip between each input sequence

USE_COVARIATES = True # Use other features as covariates (ej, temperature and pressure to predict humidity)

# Output is stored as JSON
OUTPUT_PATH = "f3_val/paquetes_s6_cov_open_meteo_garachico_"

MEAN = {
    "air_temperature": [ 70.01980283, 983.22582669,  20.09918054],
    "relative_humidity": [ 20.23295666, 983.24144877,  69.89585863],
    "atmospheric_pressure": [ 20.24945787, 69.79472259, 983.25164999]}
STD = {
    "air_temperature": [16.52443657, 36.1337416,   4.02342859],
    "relative_humidity": [ 4.01299522, 36.01805655, 17.01302257],
    "atmospheric_pressure": [ 4.04825457, 16.9311768,  36.02897078]}

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

def create_processed_windows(past_n, future_n, step, use_covariates):
  # Data structure #
  targets = ["air_temperature", "relative_humidity", "atmospheric_pressure"]
  data = {}

  # Load data #
  for i in range(len(targets)):
    # print(f"Loading {FILES[location]} for {targets[i]} in {location}")
    target = targets[i]
    data[target] = {key: [] for key in template}
    
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
        
      for dataset in range(len(location_data)):
        past_variables_windows = location_data[dataset]["past_variables"]
        future_variables_windows = location_data[dataset]["future_variables"]
        y_windows = location_data[dataset]["y"]
      
        data[target]["past_variables"].extend(past_variables_windows)
        data[target]["future_variables"].extend(future_variables_windows)
        data[target]["y"].extend(y_windows)
  
  for target in targets:
    # Convert to numpy array
    data[target]["past_variables"] = np.array(data[target]["past_variables"])
    data[target]["future_variables"] = np.array(data[target]["future_variables"])
    data[target]["y"] = np.array(data[target]["y"])
        
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
    mean = MEAN[target]
    std = STD[target]
    data[target]["past_variables"][:, :, normalization_indexes] = (data[target]["past_variables"][:, :, normalization_indexes] - mean) / std
    # For target use the data of the last feature, which is the target variable
    data[target]["y"] = (data[target]["y"] - mean[-1]) / std[-1]
  
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
        

    
  # Save train and test data
  out_path = OUTPUT_PATH + "p"+ str(past_n) + ".pkl"
  with open(out_path, "wb") as f:
    pickle.dump(data, f)
  print(f"Data saved to {out_path}")
  
for past_n in range(17, 18):
  np.random.seed(17)
  random.seed(17)
  print("Generating windows for past_n = ", past_n)
  create_processed_windows(past_n, FUTURE_N, STEP, USE_COVARIATES)
