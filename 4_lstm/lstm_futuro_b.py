import pandas as pd
import numpy as np
import pickle

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from custom_attention import CustomAttention

######################################################################

# VARIABLES #
#DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_arona_p17.pkl"
# DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_la_orotava_p17.pkl"
# DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_la_laguna_p17.pkl"
# DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_punta_hidalgo_p17.pkl"

# DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_arona_orotava_p17.pkl"
# DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_arona_laguna_orotava_p17.pkl"

DATA_PATH = "../3_data_windows/f3/paquetes_s6_cov_full_p17.pkl"

DATASET = "relative_humidity"  # atmospheric_pressure or relative_humidity or air_temperature

BATCH_SIZE = 64
SHUFFLE = True

PRINT = False

# Model #
learning_rate = 0.002
EPOCHS = 700

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
def load_data(dataset):
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    x_train = data["train"][dataset]["past_variables"]
    future_train = data["train"][dataset]["future_variables"]
    y_train = data["train"][dataset]["y"]

    x_val = data["test"][dataset]["past_variables"]
    future_val = data["test"][dataset]["future_variables"]
    y_val = data["test"][dataset]["y"]

    # BATCH AND SHUFFLE
    dataset_train = tf.data.Dataset.from_tensor_slices(((x_train, future_train), y_train))
    if SHUFFLE:
        dataset_train = dataset_train.shuffle(buffer_size=dataset_train.cardinality())
    dataset_train = dataset_train.batch(BATCH_SIZE)

    dataset_val = tf.data.Dataset.from_tensor_slices(((x_val, future_val), y_val))
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
    return dataset_train, dataset_val

#####################################################
def build_and_train_model(dataset_train):
    # Define the model
    for batch in dataset_train.take(1):
        inputs, targets = batch
        past_data, future_data = inputs
        
    past_data_shape = (past_data.shape[1], past_data.shape[2])
    future_data_shape = (future_data.shape[1], future_data.shape[2])    
    output_units = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)
    ########################################################################################
    # Encoder part (LSTM for past data)
    past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
    encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(42, return_sequences=False))(past_data_layer)

    # Decoder part (LSTM for future exogenous features)
    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    decoder_lstm = tf.keras.layers.Flatten()(future_data_layer)
    decoder_lstm = tf.keras.layers.Dense(4)(decoder_lstm)

    # Combine the outputs of encoder and decoder (you can concatenate or merge them)
    #future_residue = tf.keras.layers.Flatten()(future_data_layer)
    merged = tf.keras.layers.concatenate([encoder_lstm, decoder_lstm])#, future_residue])

    # Final output layer
    merged = tf.keras.layers.Dense(6* output_units)(merged) 
    outputs = tf.keras.layers.Dense(output_units)(merged)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    
    return model

#######################################################

# train_data, val_data = load_data("atmospheric_pressure")
train_data, val_data = load_data(DATASET)
# Ejecutar n veces y promediar el val_loss
n_runs = 10
val_losses = []

## Callbacks
path_checkpoint = "lstm_future_checkpoint.weights.h5"
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.4,         
    patience=3,         
    verbose=0,
    min_lr=1e-7
)

for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    model = build_and_train_model(train_data)
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=[reduce_lr_callback, es_callback],
        verbose=1
    )
    train_val_loss = min(history.history["val_loss"])
    val_losses.append(train_val_loss)
    print(f"Best val_loss in run {i+1}: {train_val_loss:.6f}")
    if i == 0 or train_val_loss < best_val_loss:
        best_val_loss = train_val_loss
        model.save_weights(path_checkpoint)
        print(f"Model saved with val_loss: {best_val_loss:.6f}")

avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
min_val_loss = np.min(val_losses)
print(f"\nAverage val_loss over {n_runs} runs: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
print(f"Minimum val_loss over {n_runs} runs: {min_val_loss:.6f}")
