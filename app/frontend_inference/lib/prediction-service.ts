import type { HourlyDataPoint, Datastream, JobResponse } from "@/lib/types"
import { generateMockPredictions } from "@/lib/data-processing"

// Simulated API endpoint for predictions
const PREDICTION_API_URL = "http://localhost:8000/predict"

// Submit a prediction job
export async function submitPredictionJob(
    datastream: Datastream,
    stationId: number,
    stationName: string,
    sensorData: Record<string, any>,
): Promise<string> {
    // Prepare the request payload
    const payload = {
        stationId: stationId,
        stationName: stationName,
        targetSensorId: datastream.id,
        targetSensorName: datastream.name,
        allSensorsData: sensorData,
        predictionHours: 6, // Request 24 hours of predictions
    }

    console.log("Submitting prediction job for:", datastream.name)

    // In a real implementation, this would be an actual API call
    const response = await fetch(PREDICTION_API_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        throw new Error(`Prediction API error: ${response.status}`);
    }

    const data = await response.json();
    return data.job_id;

    // For demonstration, we'll simulate the API response
    // Generate a random job ID
    const jobId = `job_${Math.random().toString(36).substring(2, 15)}`

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 1000))

    return jobId
}

// Check the status of a prediction job
export async function checkPredictionJobStatus(
    jobId: string,
    datastream: Datastream,
    hourlyData: HourlyDataPoint[],
): Promise<JobResponse> {
    console.log(`Checking status for job ${jobId}`)

    // In a real implementation, this would be an actual API call
    const response = await fetch(`${PREDICTION_API_URL}/${jobId}`, {
        method: 'GET',
        headers: {
        'Content-Type': 'application/json',
        }
    });

    if (!response.ok) {
        throw new Error(`Prediction API error: ${response.status}`);
    }

    return await response.json();

    // For demonstration, we'll simulate the API response
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 1000))

    // Simulate different job states based on time
    // First call will be "in progress", second call will be "completed"
    const now = new Date().getTime()
    const jobState = localStorage.getItem(`job_state_${jobId}`)

    if (!jobState) {
        // First check - job is in progress
        localStorage.setItem(`job_state_${jobId}`, "PENDING")
        localStorage.setItem(`job_time_${jobId}`, now.toString())

        return {
            job_id: jobId,
            status: "in progress",
        }
    } else if (jobState === "PENDING") {
        const jobTime = Number.parseInt(localStorage.getItem(`job_time_${jobId}`) || "0")
        const elapsedTime = now - jobTime

        if (elapsedTime < 3000) {
            // Still in progress
            return {
                job_id: jobId,
                status: "in progress",
            }
        } else {
            // Job completed
            localStorage.setItem(`job_state_${jobId}`, "SUCCESS")

            // Generate mock prediction data
            const mockPredictions = generateMockPredictions(datastream.id, hourlyData)

            return {
                job_id: jobId,
                status: "completed",
                result: mockPredictions,
            }
        }
    } else if (jobState === "SUCCESS") {
        // Job already completed
        const mockPredictions = generateMockPredictions(datastream.id, hourlyData)

        return {
            job_id: jobId,
            status: "completed",
            result: mockPredictions,
        }
    } else {
        // Simulate a failure case occasionally
        const shouldFail = Math.random() < 0.1

        if (shouldFail) {
            localStorage.setItem(`job_state_${jobId}`, "FAILURE")

            return {
                job_id: jobId,
                status: "failed",
                error: "Prediction model failed to converge",
            }
        } else {
            localStorage.setItem(`job_state_${jobId}`, "SUCCESS")

            // Generate mock prediction data
            const mockPredictions = generateMockPredictions(datastream.id, hourlyData)

            return {
                job_id: jobId,
                status: "completed",
                result: mockPredictions,
            }
        }
    }
}
