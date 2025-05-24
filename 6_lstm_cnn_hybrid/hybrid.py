import pandas as pd
import numpy as np
import pickle

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


######################################################################

# VARIABLES #
DATA_PATH = "../3_data_windows/f3/paquetes_s6_cov_full_p20.pkl"
DATASET = "atmospheric_pressure"  # atmospheric_pressure or relative_humidity or air_temperature

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

x_train = data["train"][DATASET]["past_variables"]
future_train = data["train"][DATASET]["future_variables"]
y_train = data["train"][DATASET]["y"]

x_val = data["test"][DATASET]["past_variables"]
future_val = data["test"][DATASET]["future_variables"]
y_val = data["test"][DATASET]["y"]

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
        
############################################

for batch in dataset_train.take(1):
    inputs, targets = batch
    past_data, future_data = inputs
    
# Define the model
past_data_shape = (past_data.shape[1], past_data.shape[2])
future_data_shape = (future_data.shape[1], future_data.shape[2])    
target_shape = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)
output_units = target_shape # Output shape should match the target sequence

learning_rate = 0.002
EPOCHS = 300
####################################################
def build_and_train_model(filters):
    # Reconstruye el modelo y entrena (todo igual que antes)
    past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
        # CNN before LSTM
    past = tf.keras.layers.Conv1D(filters=filters, kernel_size=2, activation="relu", padding="causal")(past_data_layer)
    past = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(49, return_sequences=False))(past)

    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    decoder_lstm = tf.keras.layers.Flatten()(future_data_layer)
    decoder_lstm = tf.keras.layers.Dense(4, activation='relu')(decoder_lstm)

    merged = tf.keras.layers.concatenate([past, decoder_lstm])
    outputs = tf.keras.layers.Dense(output_units)(merged)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    path_checkpoint = "lstm_future_checkpoint.weights.h5"
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)
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
        verbose=0,
        min_lr=1e-7
    )

    history = model.fit(
        dataset_train,
        epochs=EPOCHS,
        validation_data=dataset_val,
        callbacks=[modelckpt_callback, reduce_lr_callback, es_callback],
        verbose=0
    )
    best_val_loss = min(history.history["val_loss"])
    return best_val_loss

# Ejecutar n veces y promediar el val_loss
n_runs = 6
for filters in range(32, 33):
    print(f"Filters: {filters}")
    val_losses = []
    for i in range(n_runs):
        print(f"Run {i+1}/5")
        val_loss = build_and_train_model(filters)
        val_losses.append(val_loss)
        print(f"Best val_loss in run {i+1}: {val_loss:.6f}")

    avg_val_loss = np.mean(val_losses)
    min_val_loss = min(val_losses)
    std_val_loss = np.std(val_losses)
    print(f"\nAverage val_loss over {n_runs} runs: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"Minimum val_loss over {n_runs} runs: {min_val_loss:.6f}")
    