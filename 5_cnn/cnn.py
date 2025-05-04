import pandas as pd
import numpy as np
import pickle

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
######################################################################

# VARIABLES #
DATA_PATH = "../3_data_windows/processed_windows/paquetes_s6_cov_p17.pkl"

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
    return dataset_train, dataset_val

#####################################################
def build_and_train_model(dataset_train, dataset_val):
    # Define the model
    for batch in dataset_train.take(1):
        inputs, targets = batch
        past_data, future_data = inputs
        
    past_data_shape = (past_data.shape[1], past_data.shape[2])
    future_data_shape = (future_data.shape[1], future_data.shape[2])    
    output_units = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)
    ########################################################################################
    past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
    x1 = tf.keras.layers.Conv1D(63, 2, activation='relu', padding='causal')(past_data_layer)
    #x1 = tf.keras.layers.AveragePooling1D(pool_size=3)(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    # Future data: Flatten + Dense compression
    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    x2 = tf.keras.layers.Flatten()(future_data_layer)
    x2 = tf.keras.layers.Dense(4, activation='relu')(x2)

    # Combine and predict
    y = tf.keras.layers.concatenate([x1, x2])
    outputs = tf.keras.layers.Dense(output_units, name='outputs')(y)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="huber", metrics=['mse'])

    path_checkpoint = "lstm_future_checkpoint.weights.h5"
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
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
        verbose=1
    )
    best_val_loss = min(history.history["val_loss"])
    return best_val_loss
#######################################################

# train_data, val_data = load_data("atmospheric_pressure")
train_data, val_data = load_data("relative_humidity")
# Ejecutar n veces y promediar el val_loss
n_runs = 10
val_losses = []

for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    val_loss = build_and_train_model(train_data, val_data)
    val_losses.append(val_loss)
    print(f"Best val_loss in run {i+1}: {val_loss:.6f}")

avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
min_val_loss = np.min(val_losses)
print(f"\nAverage val_loss over {n_runs} runs: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
print(f"Minimum val_loss over {n_runs} runs: {min_val_loss:.6f}")
