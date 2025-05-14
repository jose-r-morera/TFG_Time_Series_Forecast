import numpy as np
import pickle
import matplotlib.pyplot as plt

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


######################################################################

# VARIABLES #
DATA_PATH = "../3_data_windows/f3/paquetes_s6_cov_full_p17.pkl"
dataset = "air_temperature"  # Change this to the dataset you want to use

BATCH_SIZE = 64

PLOT = False  # If plotting, set shuffle to False
SHUFFLE = True
PRINT = False

learning_rate = 0.002
EPOCHS = 700

NOISE_STD = 0.05
print("NOISE_STD", NOISE_STD)

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
    
x_train = data["train"][dataset]["past_variables"]
future_train = data["train"][dataset]["future_variables"]
y_train = data["train"][dataset]["y"]

x_val = data["test"][dataset]["past_variables"]
future_val = data["test"][dataset]["future_variables"]
y_val = data["test"][dataset]["y"]

#############
# BATCH AND SHUFFLE
#############
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
def tent_noise(x, R):
    u1 = tf.random.uniform(tf.shape(x), minval=0.0, maxval=1.0, dtype=x.dtype)
    u2 = tf.random.uniform(tf.shape(x), minval=0.0, maxval=1.0, dtype=x.dtype)
    # (u1 + u2)/2  is triangular on [0,1], subtract 0.5 and scale to get [−R, R]:
    target_noise = ((u1 + u2) / 2.0 - 0.5) * 2.0 * R
    return target_noise

COVARIATE_NOISE_FEATURES = [4,5,6]
def add_noise_to_target(x, y):
    past, fut = x
    # Add noise to the covariables
    covariates_noise = tent_noise(past, NOISE_STD)
    
    # Create a mask: 1 for features you want noise on, 0 otherwise
    n_features = tf.shape(past)[-1]
    mask = tf.zeros((n_features,), dtype=past.dtype)
    updates = tf.ones((len(COVARIATE_NOISE_FEATURES),), dtype=past.dtype)
    indices = tf.expand_dims(tf.constant(COVARIATE_NOISE_FEATURES, dtype=tf.int32), axis=1)
    mask = tf.tensor_scatter_nd_update(mask, indices, updates)
    mask = tf.reshape(mask, (1,1,n_features))  # Broadcast to (batch, steps, features)

    # Apply noise only to selected features
    past_noise = covariates_noise * mask
    #tf.print("past_noise", past_noise)    
    # Add noise to the target variable
    target_noise = tent_noise(y, NOISE_STD/2)
    
    return (past + past_noise, fut), y #+ target_noise

dataset_train_noisy = dataset_train.map(add_noise_to_target)
####################################################

if PLOT:
    # now grab one batch *outside* of the pipeline and plot it
    for (x_noisy, y_noisy), (x_orig, y_orig) in zip(dataset_train_noisy.take(1),
                                        dataset_train.take(1)):
        # convert to numpy
        y_orig = y_orig.numpy()        # shape = (batch_size, target_length)
        y_noisy = y_noisy.numpy()      # same shape

        # how many to plot
        n_plot = min(5, y_orig.shape[0])

        # slice out the first n_plot samples
        y_orig = y_orig[:n_plot]
        y_noisy = y_noisy[:n_plot]

        fig, axes = plt.subplots(n_plot, 1, figsize=(8, 3*n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(np.arange(y_orig.shape[1]), y_orig[i],    label='Original y')
            ax.plot(np.arange(y_orig.shape[1]), y_noisy[i],  '--', label='Noisy y')
            ax.set_ylabel(f"Sample {i}")
            ax.legend()

        axes[-1].set_xlabel("Time step")
        fig.suptitle(f"First {n_plot} targets before and after noise")
        plt.tight_layout()
        plt.show()
        
        ## Do the same for the past data
        past, fut = x_orig
        past_noisy, fut_noisy = x_noisy
        past = past.numpy()        # shape = (batch_size, target_length, n_features)
        past_noisy = past_noisy.numpy()      # same shape
        # how many to plot
        n_plot = min(5, past.shape[0])
        # slice out the first n_plot samples
        past = past[:n_plot]
        past_noisy = past_noisy[:n_plot]
        fig, axes = plt.subplots(n_plot, 1, figsize=(8, 3*n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(np.arange(past.shape[1]), past[i],    label='Original past')
            ax.plot(np.arange(past.shape[1]), past_noisy[i],  '--', label='Noisy past')
            ax.set_ylabel(f"Sample {i}")
            ax.legend()
        axes[-1].set_xlabel("Time step")
        fig.suptitle(f"First {n_plot} past data before and after noise")
        plt.tight_layout()
        plt.show()

        break   # only do it
####################################

for batch in dataset_train_noisy.take(1):
    inputs, targets = batch
    past_data, future_data = inputs
    
# Define the model
past_data_shape = (past_data.shape[1], past_data.shape[2])
future_data_shape = (future_data.shape[1], future_data.shape[2])    
target_shape = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)
output_units = target_shape # Output shape should match the target sequence

##################################################################

def build_and_train_model():
    # Reconstruye el modelo y entrena (todo igual que antes)
    past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
    encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(65, return_sequences=False))(past_data_layer)
    
    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    decoder_lstm = tf.keras.layers.LSTM(4, return_sequences=False)(future_data_layer)

    merged = tf.keras.layers.concatenate([encoder_lstm, decoder_lstm])
    outputs = tf.keras.layers.Dense(output_units)(merged)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

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
        dataset_train_noisy,
        epochs=EPOCHS,
        validation_data=dataset_val,
        callbacks=[modelckpt_callback, reduce_lr_callback, es_callback],
        verbose=0
    )
    best_val_loss = min(history.history["val_loss"])
    return best_val_loss

# Ejecutar n veces y promediar el val_loss
n_runs = 10
val_losses = []

for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    val_loss = build_and_train_model()
    val_losses.append(val_loss)
    print(f"Best val_loss in run {i+1}: {val_loss:.6f}")

avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
min_val_loss = np.min(val_losses)
print(f"\nAverage val_loss over {n_runs} runs: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
print(f"Minimum val_loss over {n_runs} runs: {min_val_loss:.6f}")
