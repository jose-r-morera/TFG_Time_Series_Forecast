from celery import Celery
from celery.signals import worker_process_init
import tensorflow as tf
import numpy as np
import logging
import json
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────
# Configure Celery
celery = Celery("tasks", broker="redis://redis:6379/0", backend="redis://redis:6379/0")
celery.conf.update(
    result_serializer='json',
    accept_content=['json'],
    task_serializer='json',
)

# ──────────────────────────────────────────────────────────────
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
def denormalize_temp_prediction(temp_array: np.ndarray):
    mean_temp = 20.09918054
    std_temp = 4.02342859
    for i in range(len(temp_array)):
        temp_array[i] = temp_array[i] * std_temp + mean_temp
    return temp_array
# ──────────────────────────────────────────────────────────────

# Global model variable
model = None
input_shape = None

@worker_process_init.connect
def init_worker(**kwargs):
    global model, input_shape
    logger.info("🔄 Initializing LSTM model...")

    try:
        # Build the same architecture before loading weights
        past_data_shape = [17, 7]  # (time_steps, num_features)
        future_data_shape = [3, 4]  # (time_steps, num_features)
        input_shape = (past_data_shape, future_data_shape)
        target_shape=3
        # Encoder part (LSTM for past data)
        past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
        past_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(65, return_sequences=False))(past_data_layer)

        # Decoder part (LSTM for future exogenous features)
        future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
        future_lstm = tf.keras.layers.LSTM(4, return_sequences=False)(future_data_layer)

        # Combine the outputs of encoder and decoder (you can concatenate or merge them)
        future_residue = tf.keras.layers.Flatten()(future_data_layer)
        merged = tf.keras.layers.concatenate([past_lstm, future_lstm, future_residue])

        # Final output layer
        outputs = tf.keras.layers.Dense(target_shape)(merged)

        # Create the model
        model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss="mse")
        model.load_weights("t_f3_2849.weights.h5")

        logger.info("✅ Model loaded successfully with shape %s", input_shape)
    except Exception as e:
        logger.exception("❌ Failed to load LSTM model.")
        model = None
        input_shape = None

# ──────────────────────────────────────────────────────────────
# Celery task
@celery.task(bind=True)
def predict_task(self, input_tensor, base_time_str):
    logger.info("📥 Received prediction task.")

    try:
        if model is None or input_shape is None:
            raise ValueError("Model is not loaded properly.")

        past_input, future_input = input_tensor
        # Convert list to numpy array and validate shape
        past_input = np.array(past_input)
        future_input = np.array(future_input)
        if past_input.shape != (1, *input_shape[0]):
            raise ValueError(f"Expected past input shape (1, {input_shape[0]}), but got {past_input.shape}")
        if future_input.shape != (1, *input_shape[1]):
            raise ValueError(f"Expected future input shape (1, {input_shape[1]}), but got {future_input.shape}")
    

        logger.info("🚀 Running model inference...")
        prediction = model.predict((past_input, future_input), verbose=0)[0]
        prediction = [round(float(p), 3) for p in prediction]
        prediction = denormalize_temp_prediction(prediction)
        
        # Determine base time from input (assumes model input ends at t0)
        base_time = datetime.fromisoformat(base_time_str.replace("Z", "")).replace(minute=0, second=0, microsecond=0)

        results = []
        for i, value in enumerate(prediction):
            start = base_time + timedelta(hours=i + 1)
            end = start + timedelta(minutes=59, seconds=59, milliseconds=999)
            results.append({
                "hour": f"{start.month}/{start.day} {start.hour}:00",
                "timestamp": start.isoformat() + "Z",
                "startTime": start.isoformat() + "Z",
                "endTime": end.isoformat() + "Z",
                "predicted": True,
                "values": [],
                "average": round(float(value), 3),
            })
        
        logger.info("✅ Inference complete.")
        return results

    except Exception as e:
        logger.exception("❌ Prediction failed.")
        return {"status": "failed", "error": str(e)}