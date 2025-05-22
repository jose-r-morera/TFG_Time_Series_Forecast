from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from celery.result import AsyncResult
from celery_worker import predict_task
import uuid
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HourlyData(BaseModel):
    hour: str
    timestamp: str
    startTime: str
    endTime: str
    values: Optional[List[float]] = []
    average: Optional[float] = None

class SensorData(BaseModel):
    name: str
    description: Optional[str]
    unitSymbol: Optional[str]
    hourlyData: List[HourlyData]

class PredictionRequest(BaseModel):
    stationId: int
    stationName: str
    targetSensorId: int
    targetSensorName: str
    predictionHours: int
    allSensorsData: Dict[int, SensorData]

# Time feature computation
def compute_time_features(timestamp: str):
    dt = datetime.fromisoformat(timestamp.replace("Z", ""))
    hour = dt.hour
    minute = dt.minute
    hour_fraction = hour + minute / 60.0

    #day_of_week = dt.weekday()
    #week_fraction = day_of_week + hour_fraction / 24.0

    day_of_year = dt.timetuple().tm_yday
    year_fraction = (day_of_year - 1) + hour_fraction / 24.0
    is_leap = (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0)
    days_in_year = 366 if is_leap else 365

    return [
        np.sin(2 * np.pi * hour_fraction / 24),
        np.cos(2 * np.pi * hour_fraction / 24),
        #np.sin(2 * np.pi * week_fraction / 7),
        #np.cos(2 * np.pi * week_fraction / 7),
        np.sin(2 * np.pi * year_fraction / days_in_year),
        np.cos(2 * np.pi * year_fraction / days_in_year)
    ]


def save_input_tensor_channels(input_tensor: np.ndarray, job_id: str, output_dir: Path):
    """
    Generates 1D line plots for each channel in the input tensor.
    input_tensor shape: [1, T, C]
    """
    if input_tensor.ndim != 3 or input_tensor.shape[0] != 1:
        raise ValueError("Expected input_tensor with shape [1, T, C]")

    data = input_tensor[0]  # shape [T, C]
    timesteps, num_channels = data.shape
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel_idx in range(num_channels):
        plt.figure(figsize=(8, 3))
        plt.plot(range(timesteps), data[:, channel_idx], marker="o")
        plt.title(f"Channel {channel_idx + 1}")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()

        path = output_dir / f"{job_id}_channel_{channel_idx + 1}.png"
        plt.savefig(path, dpi=150)
        plt.close()

@app.post("/predict")
async def submit_prediction(request: PredictionRequest):
    past_timesteps = 47 # if we want to pass 48 point to the model

    # Define the sensors to extract (match lowercased names)
    feature_names = [
        "air temperature (avg.)",
        "relative humidity (avg.)",
        "atmosferic pressure (avg.)"
    ]

    # Build map of sensor name -> ID (case-insensitive)
    name_to_id = {
        (sensor.name or "").strip().lower(): sid
        for sid, sensor in request.allSensorsData.items()
    }

    print(feature_names)
    print(name_to_id)

    # Validate presence of required features
    missing = [name for name in feature_names if name not in name_to_id]
    if missing:
        print("me falta algo brotha")
        raise HTTPException(status_code=400, detail=f"Missing required sensors: {missing}")


    # Build input tensor
    feature_tensor = []
    for t in range(past_timesteps):
        timestep_features = []

        # Use target sensor’s timestamp for time features
        try:
            timestamp = request.allSensorsData[request.targetSensorId].hourlyData[-past_timesteps + t].timestamp
        except IndexError:
            raise HTTPException(status_code=400, detail=f"Not enough historical data for timestep {t}")

        timestep_features.extend(compute_time_features(timestamp))

        # Add sensor values
        for name in feature_names:
            sid = name_to_id[name]
            hd = request.allSensorsData[sid].hourlyData

            if len(hd) < past_timesteps:
                val = 0.0
            else:
                val = hd[-past_timesteps + t].average or 0.0

            timestep_features.append(val)

        feature_tensor.append(timestep_features)

    input_tensor = np.array(feature_tensor, dtype=np.float32).reshape(1, past_timesteps, -1)
    save_input_tensor_channels(input_tensor, "0000", Path("debug_inputs"))

    job_id = str(uuid.uuid4())
    task = predict_task.apply_async(args=[input_tensor.tolist()], task_id=job_id)

    return {"job_id": job_id, "status": "submitted"}

@app.get("/predict/{job_id}")
async def get_prediction_status(job_id: str):
    result = AsyncResult(job_id)

    if result.state == "PENDING":
        return {"job_id": job_id, "status": "in progress"}
    elif result.state == "SUCCESS":
        return {"job_id": job_id, "status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(result.result)}
    else:
        return {"job_id": job_id, "status": result.state}