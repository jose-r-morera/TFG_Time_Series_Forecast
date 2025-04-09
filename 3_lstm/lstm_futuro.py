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
DATA_PATH = "paquetes_futuro.pkl"

BATCH_SIZE = 64
SHUFFLE = False

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

x_train = data["air_temperature"]["x_train"]
future_train = data["air_temperature"]["future_train"]
y_train = data["air_temperature"]["y_train"]

x_val = data["air_temperature"]["x_val"]
future_val = data["air_temperature"]["future_val"]
y_val = data["air_temperature"]["y_val"]

#############
# BATCH AND SHUFFLE
#############
dataset_train = tf.data.Dataset.from_tensor_slices(((x_train, future_train), y_train))
if SHUFFLE:
  dataset_train = dataset_train.shuffle(buffer_size=1000)
dataset_train = dataset_train.batch(BATCH_SIZE)

dataset_val = tf.data.Dataset.from_tensor_slices(((x_val, future_val), y_val))
if SHUFFLE:
  dataset_val = dataset_val.shuffle(buffer_size=1000)
dataset_val = dataset_val.batch(BATCH_SIZE)

for batch_inputs, batch_y in dataset_train.take(1):  # Take the first batch
    batch_x, batch_future = batch_inputs
    print("First batch X:", batch_x.numpy()[:2])
    print("First batch future:", batch_future.numpy()[:2])
    print("First batch Y:", batch_y.numpy()[:2])
for batch_inputs, batch_y in dataset_val.take(1):  # Take the first batch
    print("First batch X:", batch_x.numpy()[:2])
    print("First batch future:", batch_future.numpy()[:2])
    print("First batch Y:", batch_y.numpy()[:2])
    
###########################################
class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        # These are trainable layers that compute the attention score
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, hidden_units)
        # Compute score for each time step
        score = tf.nn.tanh(self.W(inputs))             # (batch_size, time_steps, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, time_steps, 1)
        # Multiply each hidden state by its attention weight and sum over time steps
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)  # (batch_size, hidden_units)
        return context_vector
############################################

for batch in dataset_train.take(1):
    inputs, targets = batch
    past_data, future_data = inputs
    
# Example placeholder values – update these with your actual data dimensions
past_data_shape = (past_data.shape[1], past_data.shape[2])
future_data_shape = (future_data.shape[1], future_data.shape[2])    
output_dim = targets.shape[1]     # e.g., How many values to predict (e.g., 3-hour forecast)
EPOCHS = 700

####################################################

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        # These are trainable layers that compute the attention score
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, hidden_units)
        # Compute score for each time step
        score = tf.nn.tanh(self.W(inputs))             # (batch_size, time_steps, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, time_steps, 1)
        # Multiply each hidden state by its attention weight and sum over time steps
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)  # (batch_size, hidden_units)
        return context_vector

################################################

def build_model(hp):
    K.clear_session() # clear memory
    gc.collect() # garbage collector

    ### MODEL DEFINITION ####
    # General parameters
    use_second_lstm = hp.Boolean('use_second_lstm', default=False) # Second lstm on past data
    use_attention_1 = hp.Boolean('use_attention_1', default=False) # Attention on past data
    use_attention_2 = False # Attention on merged data
    if not use_attention_1:
        use_attention_2 = hp.Boolean('use_attention_2', default=False)
    
    #################################
    # --- PAST DATA Input layer --- #
    #################################
    past_input = layers.Input(shape=(past_data_shape), name ="past")
    
    # --- First LSTM layer ---
    units_1 = hp.Int('units_1', min_value=16, max_value=640, step=24)
    bidir_1 = hp.Boolean('bidir_1', default=False)
    
    if bidir_1:
        units_1 //= 2  # Halve the units for bidirectional LSTM
        past_layers = layers.Bidirectional(
            layers.LSTM(units_1, return_sequences=use_second_lstm or use_attention_1) 
        ) (past_input)
    else:
        past_layers = layers.LSTM(units_1, return_sequences=use_second_lstm or use_attention_1 or use_attention_2)(past_input)
    past_out_dim = units_1 # save past data dimesion
    
    # --- Optional second LSTM layer ---
    if use_second_lstm:
        units_2 = hp.Int('units_2', min_value=16, max_value=640, step=24)
        past_out_dim = units_2
        bidir_2 = hp.Boolean('bidir_2', default=False)
        
        if bidir_2:
            past_layers = layers.Bidirectional(layers.LSTM(units_2, return_sequences=use_attention_1 or use_attention_2))(past_layers)
        else:
            past_layers = layers.LSTM(units_2, return_sequences=use_attention_1)(past_layers)

    # --- Optional Attention Layer ---
    if use_attention_1:
        if use_second_lstm:
            attn_units_1 = hp.Int('attn_units_1', min_value=units_2//2, max_value=units_2*2, step=24)
        else:
            attn_units_1 = hp.Int('attn_units_1', min_value=units_1//2, max_value=units_1*2, step=24)
        past_out_dim = attn_units_1
        past_layers = Attention(attn_units_1)(past_layers)  # Apply attention before dense layer
 
    ##################
    # --- FUTURE --- #
    ##################
    future_input = layers.Input(shape=(future_data_shape), name ="future_data")

    units_f = hp.Int('units_f', min_value=16, max_value=640, step=24)
    bidir_f = hp.Boolean('bidir_f', default=False)
    
    if bidir_f:
        units_f //= 2  # Halve the units for bidirectional LSTM
        future_layers = layers.Bidirectional(
            layers.LSTM(units_f, name="lstm_future", return_sequences=False) 
        )(future_input)
    else:
        future_layers = layers.LSTM(units_f, name="lstm_future", return_sequences=False)(future_input)

    #################
    # --- MERGE --- #
    #################
    merged = layers.Concatenate(name='concatenate')([past_layers, future_layers])
    merged_dim = past_out_dim + units_f # past + future units

    if use_attention_2:
        merged = layers.Reshape((1, -1), name="att_reshape")(merged)
        attn_units_2 = hp.Int('attn_units_2', min_value=merged_dim//2, max_value=round(merged_dim*1.5), step=24)
        merged = Attention(attn_units_2)(merged)
    
    dense_2 = hp.Boolean('dense_2', default=False)
    if dense_2: 
        units_dense_2 = hp.Int('units_dense_2', min_value=merged_dim//4, max_value=round(merged_dim*1.5), step=24)
        merged = layers.Dense(units_dense_2, activation="relu")(merged)

    # --- Dense output layer ---
    output = layers.Dense(output_dim)(merged)

    ########################################################
    model = keras.Model(inputs=[past_input, future_input], outputs=output, name='past_future_model')
    # --- Learning rate tuning ---
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=["mse"],
    )
    
    model.summary()
    return model

# Create a tuner instance
# tuner = kt.Hyperband(
#     build_model,
#     objective='val_mse',
#     max_epochs=EPOCHS,
#     executions_per_trial=3,
#     factor=3,
#     directory='tuner',
#     project_name='lstm_tuner'
# )
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mse',
    executions_per_trial=3,
    directory='tuner',
    project_name='lstm_future_bayesian'
)

# Define tunable patience and min_delta for ReduceLROnPlateau
def get_callbacks(hp):
    patience = hp.Int('reduce_lr_patience', min_value=2, max_value=5, step=1)
    factor = hp.Choice('reduce_lr_factor', values=[0.4, 0.5, 0.6, 0.7])

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

# Print best hyperparameters found
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print(f"First LSTM units: {best_hp.get('units_1')}")
print(f"First LSTM bidirectional: {best_hp.get('bidir_1')}")
print(f"Use second LSTM layer: {best_hp.get('use_second_lstm')}")
if best_hp.get('use_second_lstm'):
    print(f"Second LSTM units: {best_hp.get('units_2')}")
    print(f"Second LSTM bidirectional: {best_hp.get('bidir_2')}")
print(f"Use Attention: {best_hp.get('use_attention_1')}")
if best_hp.get('use_attention_1'):
    print(f"Attention layer units: {best_hp.get('attn_units_1')}")
print(f"Learning rate: {best_hp.get('learning_rate')}")
print(f"Reduce LR patience: {best_hp.get('reduce_lr_patience')}")
print(f"Reduce LR factor: {best_hp.get('reduce_lr_factor')}")