from locust import HttpUser, task, between
import uuid
import random
import time
from datetime import datetime, timedelta


class PredictionUser(HttpUser):
    wait_time = between(2, 5)

    REQUIRED_SENSORS = [
        "air temperature (avg.)",
        "relative humidity (avg.)",
        "atmosferic pressure (avg.)"
    ]

    def generate_fake_hourly_data(self, hours=48):
        now = datetime.utcnow()
        return [
            {
                "hour": f"{(now - timedelta(hours=hours - i)).strftime('%m/%d %H:00')}",
                "timestamp": (now - timedelta(hours=hours - i)).isoformat() + "Z",
                "startTime": (now - timedelta(hours=hours - i)).isoformat() + "Z",
                "endTime": (now - timedelta(hours=hours - i - 1)).isoformat() + "Z",
                "average": round(random.uniform(10, 30), 2)
            }
            for i in range(hours)
        ]

    def generate_payload(self):
        sensor_ids = [str(uuid.uuid4()) for _ in range(len(self.REQUIRED_SENSORS))]
        hourly_data = self.generate_fake_hourly_data()

        sensor_data = {
            sid: {
                "name": name,
                "description": "Simulated for load test",
                "unitSymbol": "°C" if "temperature" in name else "%",
                "hourlyData": hourly_data
            }
            for sid, name in zip(sensor_ids, self.REQUIRED_SENSORS)
        }

        return {
            "stationId": 1001,
            "stationName": "Simulated Station",
            "targetSensorId": sensor_ids[0],
            "targetSensorName": self.REQUIRED_SENSORS[0],
            "predictionHours": 3,
            "allSensorsData": sensor_data
        }

    @task
    def predict_and_poll(self):
        payload = self.generate_payload()

        with self.client.post("/predict", json=payload, catch_response=True) as res:
            if res.status_code != 200 or "job_id" not in res.json():
                res.failure("❌ POST /predict failed")
                return

            job_id = res.json()["job_id"]
            polling_attempts = 0
            max_attempts = 6
            poll_interval = 3  # segundos

            final_status = None

            while polling_attempts < max_attempts:
                time.sleep(poll_interval)
                polling_attempts += 1

                with self.client.get(f"/predict/{job_id}", name="/predict/{job_id}", catch_response=True) as r:
                    if r.status_code != 200:
                        r.failure("❌ GET /predict/{job_id} failed")
                        return

                    status = r.json().get("status", "").lower()

                    if status in ("completed", "success"):
                        r.success()
                        return
                    elif status in ("failed", "failure"):
                        r.failure("❌ Prediction failed")
                        return
                    elif status in ("pending", "in progress"):
                        final_status = status
                        continue  # no se considera éxito aún
                    else:
                        r.failure(f"⚠️ Estado desconocido: {status}")
                        return

            # Si salimos del bucle sin éxito ni error explícito
            if final_status in ("pending", "in progress"):
                r.failure("⏱ Timeout polling job still in progress")