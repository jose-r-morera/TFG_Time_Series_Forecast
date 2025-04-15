import pandas as pd
import numpy as np
import pickle

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
def create_df_windows(df, features, target):
  label_start = past_n
  y = df.iloc[label_start:][target]

  past_variables = []
  future_variables = []
  y_multi_step = []
  for i in range(0, len(df) - past_n - future_n + 1, STEP):
    past_variables_window = df.iloc[i:i + past_n][features].values
    past_variables.append(past_variables_window)
    future_n_window = df.iloc[label_start + i:label_start + i + future_n][features].drop(columns=[target]).values
    future_variables.append(future_n_window)
    y_window = y.iloc[i:i + future_n].values
    y_multi_step.append(y_window)

  past_variables = np.array(past_variables)
  future_variables = np.array(future_variables)
  y_multi_step = np.array(y_multi_step)
    
  return past_variables, future_variables, y
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
    past_variables, future_variables, y = create_df_windows(df, features, target)
    # Store data
    data[target][file] = {}
    data[target][file]["past_variables"] = past_variables
    data[target][file]["future_variables"] = future_variables
    data[target][file]["y"] = y

####################
# Train-test split #
####################
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
    print("samples length: ", len(samples["past_variables"]))
    # Split data
    TRAIN_PERCENTAGE = 0.9
    train_split = int(len(samples["past_variables"]) * TRAIN_PERCENTAGE)
    train_data[target]["past_variables"].extend(samples["past_variables"][:train_split])
    train_data[target]["future_variables"].extend(samples["future_variables"][:train_split])
    train_data[target]["y"].extend(samples["y"][:train_split])
    # Test data
    test_data[target]["past_variables"].extend(samples["past_variables"][train_split:])
    test_data[target]["future_variables"].extend(samples["future_variables"][train_split:])
    test_data[target]["y"].extend(samples["y"][train_split:])
  
print("Train data shape:")
for target in targets:
  print(f"{target}: {len(train_data[target]['past_variables'])} samples")
print("Test data shape:")
for target in targets:
  print(f"{target}: {len(test_data[target]['past_variables'])} samples")
  
###################################################
def normalize(train, test, no_normalize):
  data_mean = train.mean(axis=0)
  data_std = train.std(axis=0)
  return [(train - data_mean) / data_std, data_mean, data_std]
#################################################

# no_normalize = ["sin_day", "cos_day", "sin_year", "cos_year"]


#   # Normalization #
#   # (After data split)
#   features = df.drop(columns=["time", *no_normalize]).astype("float32")
#   normalized_features, mean, std = normalize(features.values, first_2025_indepast_variables)
#   normalized_features = pd.DataFrame(normalized_features, columns=features.columns)
  
#     for col in no_normalize:
#       normalized_features[col] = df[col].astype("float32")
  
#       data = {"mean": mean, "std": std} # Save data statistics from normaliation



# with open(OUTPUT_PATH, "wb") as f:
#     pickle.dump(data, f)
