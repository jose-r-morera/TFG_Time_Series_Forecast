from celery import Celery
from celery.signals import worker_process_init
import tensorflow as tf
import numpy as np
import logging
import json

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
# Global model variable
model = None
input_shape = None

@worker_process_init.connect
def init_worker(**kwargs):
    global model, input_shape
    logger.info("🔄 Initializing LSTM model...")

    try:
        # Build the same architecture before loading weights
        past_data_input = tf.keras.layers.Input(shape=(72, 7), name="past_data")
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, return_sequences=True))(past_data_input)
        x = tf.keras.layers.LSTM(16, return_sequences=False)(x)
        outputs = tf.keras.layers.Dense(3)(x)  # adjust this to match output dim

        model = tf.keras.Model(inputs=past_data_input, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss="mse")
        model.load_weights("lstm_future_checkpoint.weights.h5")
        input_shape = (72, 7)

        logger.info("✅ Model loaded successfully with shape %s", input_shape)
    except Exception as e:
        logger.exception("❌ Failed to load LSTM model.")
        model = None
        input_shape = None

# ──────────────────────────────────────────────────────────────
# Celery task
@celery.task(bind=True)
def predict_task(self, input_tensor):
    logger.info("📥 Received prediction task.")

    try:
        if model is None or input_shape is None:
            raise ValueError("Model is not loaded properly.")

        # Convert list to numpy array and validate shape
        input_array = np.array(input_tensor)
        if input_array.shape != (1, *input_shape):
            raise ValueError(f"Expected input shape (1, {input_shape}), but got {input_array.shape}")

        logger.info("🚀 Running model inference...")
        prediction = model.predict(input_array, verbose=0)[0]

        result = {
            "status": "completed",
            "predictions": [round(float(p), 3) for p in prediction]
        }
        logger.info("✅ Inference complete.")
        return result

    except Exception as e:
        logger.exception("❌ Prediction failed.")
        return {"status": "failed", "error": str(e)}