import pandas as pd
import numpy as np
import pickle

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from keras import backend as K
import gc # garbage collector

######################################################################

# VARIABLES #
#DATA_PATH = "./paquetes_s6.pkl"
#DATA_PATH = "../3_data_windows/paquetes_s6_augmented.pkl"
DATA_PATH = "../3_data_windows/paquetes_s6_covariates_augmented.pkl"

BATCH_SIZE = 64
SHUFFLE = True

PRINT = False

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

x_train = data["train"]["air_temperature"]["past_variables"]
y_train = data["train"]["air_temperature"]["y"]

x_val = data["test"]["air_temperature"]["past_variables"]
y_val = data["test"]["air_temperature"]["y"]

#############
# BATCH AND SHUFFLE
#############
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
if SHUFFLE:
  dataset_train = dataset_train.shuffle(buffer_size=15000)
dataset_train = dataset_train.batch(BATCH_SIZE)

dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
if SHUFFLE:
  dataset_val = dataset_val.shuffle(buffer_size=15000)
dataset_val = dataset_val.batch(BATCH_SIZE)

if PRINT:
    for batch_inputs, batch_y in dataset_train.take(1):  # Take the first batch
        print("First batch X:", batch_inputs.numpy()[:2])
        print("First batch Y:", batch_y.numpy()[:2])
    for batch_inputs, batch_y in dataset_val.take(1):  # Take the first batch
        print("First batch X:", batch_inputs.numpy()[:2])
        print("First batch Y:", batch_y.numpy()[:2])
        
############################################

for batch in dataset_train.take(1):
    inputs, targets = batch
    
# Define the model
past_data_shape = (inputs.shape[1], inputs.shape[2])
target_shape = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)

learning_rate = 0.002
EPOCHS = 700

####################################################

# Encoder part (LSTM for past data)
past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
#past_data_layer = layers.SpatialDropout1D(0.1)(past_data_layer)
past_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, return_sequences=True))(past_data_layer)
#past_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True, recurrent_dropout=0.1))(past_data_layer)
past_lstm = tf.keras.layers.LSTM(16, return_sequences=False)(past_lstm)
# dropout
#past_lstm = tf.keras.layers.Dropout(0.2)(past_lstm)

# Final output layer
output_units = target_shape # Output shape should match the target sequence
outputs = tf.keras.layers.Dense(output_units)(past_lstm)

# Create the model
model = tf.keras.Model(inputs=past_data_layer, outputs=outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

# Define the callbacks
path_checkpoint = "lstm_future_checkpoint.weights.h5"
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.4,         
    patience=3,         
    verbose=1,
    min_lr=1e-7
)

# Train the model
history = model.fit(
    dataset_train,
    epochs=EPOCHS,
    validation_data=dataset_val,
    callbacks=[modelckpt_callback, reduce_lr_callback, es_callback],
)