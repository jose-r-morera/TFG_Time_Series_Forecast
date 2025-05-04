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
EPOCHS        = 700

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.4, patience=3, min_lr=1e-7, verbose=0
)

def build_model(past_shape, future_shape, target_dim):
    """Returns a compiled Keras model (fresh weights)."""
    past_data_layer   = tf.keras.layers.Input(shape=past_shape,   name="past_data")
    future_data_layer = tf.keras.layers.Input(shape=future_shape, name="future_data")

    x1 = tf.keras.layers.Conv1D(63, 2, activation='relu', padding='causal')(past_data_layer)
    #x1 = tf.keras.layers.AveragePooling1D(pool_size=2)(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    # Future data: Flatten + Dense compression
    x2 = tf.keras.layers.Flatten()(future_data_layer)
    x2 = tf.keras.layers.Dense(4, activation='relu')(x2)

    # Combine and predict
    y = tf.keras.layers.concatenate([x1, x2])
    outputs = tf.keras.layers.Dense(target_dim, name='outputs')(y)


    model = tf.keras.Model([past_data_layer, future_data_layer], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse"
    )
    return model

def load_dataset(path):
    """Loads your pickle and returns two tf.data.Dataset for train & val."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    x_tr  = data["train"]["air_temperature"]["past_variables"]
    f_tr  = data["train"]["air_temperature"]["future_variables"]
    y_tr  = data["train"]["air_temperature"]["y"]

    x_val = data["test"]["air_temperature"]["past_variables"]
    f_val = data["test"]["air_temperature"]["future_variables"]
    y_val = data["test"]["air_temperature"]["y"]

    ds_tr = tf.data.Dataset.from_tensor_slices(((x_tr,f_tr), y_tr))
    ds_val= tf.data.Dataset.from_tensor_slices(((x_val,f_val), y_val))

    if SHUFFLE:
        ds_tr  = ds_tr.shuffle(15000)
        ds_val = ds_val.shuffle(15000)

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
    DATA_PATH = "../3_data_windows/processed_windows/paquetes_s6_cov_p"
    min_i = 5
    max_i = 29
    best_ds, all_scores = evaluate_with_trials(DATA_PATH, min_i, max_i)
    print(f"Best dataset: {best_ds}")
    print(f"All scores: {all_scores}")