#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid search for the size of the past window in LSTM with future covariates
===========

Compares different sizes of the past window using grid search and valitdation loss.
Averages results over multiple runs for each configuration.

Example:
        $ python file.py

"""

__author__ = "José Ramón Morera Campos"
__version__ = "1.0.1"
#######################################################################


import os
import pickle
import numpy as np
import tensorflow as tf

# Disable TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Hyperparams & callbacks
BATCH_SIZE    = 64
SHUFFLE       = True
LEARNING_RATE = 0.002
EPOCHS        = 300

DATASET = "relative_humidity"  # air_temperature, "atmospheric_pressure" or "relative_humidity"

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, min_delta=0.000001, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.4, patience=3, min_lr=1e-7, verbose=0
)

def build_model(past_shape, future_shape, target_dim):
    """Returns a compiled Keras model (fresh weights)."""
    past_in   = tf.keras.layers.Input(shape=past_shape,   name="past_data")
    future_in = tf.keras.layers.Input(shape=future_shape, name="future_data")

    # Past data 
    past_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(42, return_sequences=False))(past_in)
    past_lstm = tf.keras.layers.Dense(20)(past_lstm)
    
    # Future exogenous features
    future_dense = tf.keras.layers.Flatten()(future_in)
    future_dense = tf.keras.layers.Dense(4)(future_dense)

    # Combine the outputs of past and future
    future_residue = tf.keras.layers.Flatten()(future_in)
    merged = tf.keras.layers.concatenate([past_lstm, future_dense, future_residue])

    # Final output layer
    merged = tf.keras.layers.Dense(6* target_dim)(merged) 
    out = tf.keras.layers.Dense(target_dim)(merged)

    model = tf.keras.Model([past_in, future_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse"
    )
    return model

def load_dataset(path):
    """Loads your pickle and returns two tf.data.Dataset for train & val."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    x_tr  = data["train"][DATASET]["past_variables"]
    f_tr  = data["train"][DATASET]["future_variables"]
    y_tr  = data["train"][DATASET]["y"]

    x_val = data["test"][DATASET]["past_variables"]
    f_val = data["test"][DATASET]["future_variables"]
    y_val = data["test"][DATASET]["y"]

    ds_tr = tf.data.Dataset.from_tensor_slices(((x_tr,f_tr), y_tr))
    ds_val= tf.data.Dataset.from_tensor_slices(((x_val,f_val), y_val))

    if SHUFFLE:
        ds_tr  = ds_tr.shuffle(ds_tr.cardinality())

    return ds_tr.batch(BATCH_SIZE), ds_val.batch(BATCH_SIZE)

def evaluate_with_trials(data_path, min_file, max_file, trials=5):
    """For each path, run `trials` independent trainings and aggregate val_loss."""
    summary = {}

    for i in range(min_file, max_file+1):
        path = data_path + str(i) + ".pkl"
        print(f"\n=== Dataset: {os.path.basename(path)} ===")
        ds_tr, ds_val = load_dataset(path)

        # infer shapes once
        for (p_batch, f_batch), y_batch in ds_tr.take(1):
            past_shape   = (p_batch.shape[1], p_batch.shape[2])
            future_shape = (f_batch.shape[1], f_batch.shape[2])
            target_dim   = y_batch.shape[1]
        # end shape inference

        losses = []
        for t in range(trials):
            print(f" Run {t+1}/{trials}...", end="", flush=True)
            model = build_model(past_shape, future_shape, target_dim)

            history = model.fit(
                ds_tr,
                epochs=EPOCHS,
                validation_data=ds_val,
                callbacks=[es_callback, reduce_lr],
                verbose=0
            )
            best_val = min(history.history['val_loss'])
            losses.append(best_val)
            print(f" best_val_loss={best_val:.5f}")

        mean_loss = np.mean(losses)
        std_loss  = np.std(losses)
        summary[path] = {
            "losses": losses,
            "mean": mean_loss,
            "std": std_loss
        }
        print(f" → mean val_loss = {mean_loss:.5f} ± {std_loss:.5f}")

    # pick the dataset with lowest mean val_loss
    best_path = min(summary, key=lambda p: summary[p]["mean"])
    best_stats = summary[best_path]
    print(f"\n*** Best dataset is {os.path.basename(best_path)} "
          f"with mean val_loss = {best_stats['mean']:.5f} "
          f"(std={best_stats['std']:.5f})")
    return best_path, summary

if __name__ == "__main__":
    # list your datasets here:
    DATA_PATH = "../3_data_windows/f6/paquetes_s6_cov_full_p"
    min_i = 6
    max_i = 36
    best_ds, all_scores = evaluate_with_trials(DATA_PATH, min_i, max_i)
    print(f"Best dataset: {best_ds}")
    print(f"All scores: {all_scores}")