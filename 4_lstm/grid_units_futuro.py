import pandas as pd
import numpy as np
import pickle

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import keras_tuner as kt
from keras import backend as K
import gc # garbage collector


######################################################################

# VARIABLES #
#DATA_PATH = "./paquetes_s6.pkl"
#DATA_PATH = "../3_data_windows/paquetes_s6_augmented.pkl"
DATA_PATH = "../3_data_windows/f3/paquetes_s6_cov_full_p24.pkl"

BATCH_SIZE = 64
SHUFFLE = True

PRINT = False

learning_rate = 0.002
EPOCHS = 700
DATASET = "relative_humidity"  # atmospheric_pressure or relative_humidity or air_temperature

#################################################################
# Avoid memory issues with TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
#############
# Load data #
#############
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

x_train = data["train"][DATASET]["past_variables"]
future_train = data["train"][DATASET]["future_variables"]
y_train = data["train"][DATASET]["y"]

x_val = data["test"][DATASET]["past_variables"]
future_val = data["test"][DATASET]["future_variables"]
y_val = data["test"][DATASET]["y"]

#############
# BATCH AND SHUFFLE
#############
dataset_train = tf.data.Dataset.from_tensor_slices(((x_train, future_train), y_train))
if SHUFFLE:
  dataset_train = dataset_train.shuffle(buffer_size=dataset_train.cardinality())
dataset_train = dataset_train.batch(BATCH_SIZE)

dataset_val = tf.data.Dataset.from_tensor_slices(((x_val, future_val), y_val))
if SHUFFLE:
  dataset_val = dataset_val.shuffle(buffer_size=dataset_val.cardinality())
dataset_val = dataset_val.batch(BATCH_SIZE)

if PRINT:
    for batch_inputs, batch_y in dataset_train.take(1):  # Take the first batch
        batch_x, batch_future = batch_inputs
        print("First batch X:", batch_x.numpy()[:2])
        print("First batch future:", batch_future.numpy()[:2])
        print("First batch Y:", batch_y.numpy()[:2])
    for batch_inputs, batch_y in dataset_val.take(1):  # Take the first batch
        batch_x, batch_future = batch_inputs
        print("First batch X:", batch_x.numpy()[:2])
        print("First batch future:", batch_future.numpy()[:2])
        print("First batch Y:", batch_y.numpy()[:2])
        
############################################

for batch in dataset_train.take(1):
    inputs, targets = batch
    past_data, future_data = inputs
    
# Define the model
past_shape = (past_data.shape[1], past_data.shape[2])
future_shape = (future_data.shape[1], future_data.shape[2])    
target_dim = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)


####################################################

def build_model(hp):
    K.clear_session() # clear memory
    gc.collect() # garbage collector
    # Encoder part (LSTM for past data)
    past_in   = tf.keras.layers.Input(shape=past_shape,   name="past_data")
    future_in = tf.keras.layers.Input(shape=future_shape, name="future_data")

    # Past data 
   
    past_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(42, return_sequences=False))(past_in)
    
    # Future exogenous features
    future_dense = tf.keras.layers.Flatten()(future_in)
    future_units = hp.Int('future_units', min_value=1, max_value=8, step=1)
    future_dense = tf.keras.layers.Dense(future_units)(future_dense)

    # Combine the outputs of past and future
    future_residue = tf.keras.layers.Flatten()(future_in)
    merged = tf.keras.layers.concatenate([past_lstm, future_dense, future_residue])

    # Final output layer
    merged = tf.keras.layers.Dense(6* target_dim)(merged) 
    out = tf.keras.layers.Dense(target_dim)(merged)

    model = tf.keras.Model([past_in, future_in], out)
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=["mse"],
    )
    # model.summary()
    return model

# tuner = kt.BayesianOptimization(
#     build_model,
#     objective='val_mse',
#     executions_per_trial=3,
#     directory='../output/tuner',
#     project_name='lstm_future_bayesian'
# )

# Grid search
tuner = kt.GridSearch(
    build_model,
    objective='val_mse',
    max_trials=500,
    executions_per_trial=6,
    directory='../output/tuner',
    project_name='lstm_fut_grid_h2'
)

# Define tunable patience and min_delta for ReduceLROnPlateau
def get_callbacks(hp):
    patience = 3
    factor = 0.4

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        min_delta=0.0001,  # Minimum change to be considered an improvement
        monitor='val_loss', 
        factor=factor,  # Reduce LR by a factor of 0.5
        patience=patience,  # Wait for `patience` epochs before reducing
        verbose=1,
        min_lr=5e-7
    )
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True)

    return [reduce_lr_callback, es_callback]


###############################################################################
print("Search space hyperparameters:")
for hp in tuner.oracle.get_space().space:
    print(f"  {hp.name}: {getattr(hp, 'values', getattr(hp, 'sampling', ''))}")

# Run the hyperparameter search
tuner.search(dataset_train, epochs=EPOCHS, validation_data=dataset_val, callbacks=get_callbacks(tuner.oracle.hyperparameters))

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print(f"Past LSTM units: {best_hyperparameters.get('past_units')}")
