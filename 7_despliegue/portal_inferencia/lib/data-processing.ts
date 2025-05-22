import type { HourlyDataPoint, Observation } from "./types"

// Calculate hourly averages for a set of observations
export function calculateHourlyAverages(
    observations: Observation[],
    startDate: Date,
    endDate: Date,
): HourlyDataPoint[] {
    // Create an array of hour slots for the 48-hour period
    const hourlyData: HourlyDataPoint[] = []

    // Make a copy of the start date to avoid modifying the original
    const currentDate = new Date(startDate)

    // Normalize to the start of the hour (XX:00:00.000)
    currentDate.setMinutes(0, 0, 0)

    // Generate slots for each hour in the 48-hour period
    while (currentDate <= endDate) {
        const hourStart = new Date(currentDate)

        // End time is XX:59:59.999 of the same hour
        const hourEnd = new Date(currentDate)
        hourEnd.setMinutes(59, 59, 999)

        // Move to next hour for the next iteration
        currentDate.setHours(currentDate.getHours() + 1)

        hourlyData.push({
            hour: formatHourLabel(hourStart),
            timestamp: hourStart.toISOString(),
            startTime: hourStart,
            endTime: hourEnd,
            values: [],
            average: null,
        })
    }

    // Assign each observation to its corresponding hour slot
    observations.forEach((obs) => {
        const obsTime = new Date(obs.resultTime)

        // Find the matching hour slot - include observations exactly at the end time
        const hourSlot = hourlyData.find((slot) => obsTime >= slot.startTime && obsTime <= slot.endTime)

        if (hourSlot) {
            // Convert result to number and add to values array
            const value = Number.parseFloat(obs.result)
            if (!isNaN(value)) {
                hourSlot.values.push(value)
            }
        }
    })

    // Calculate the average for each hour that has values
    hourlyData.forEach((hourSlot) => {
        if (hourSlot.values.length > 0) {
            const sum = hourSlot.values.reduce((acc, val) => acc + val, 0)
            hourSlot.average = sum / hourSlot.values.length
        }
        // If no values, average remains null
    })

    // Count how many hours have data
    const hoursWithData = hourlyData.filter((hour) => hour.average !== null).length
    console.log(`${hoursWithData} out of ${hourlyData.length} hours have data`)

    return hourlyData
}

// Format hour label for display
export function formatHourLabel(date: Date): string {
    return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:00-59`
}

// Generate mock prediction data for demonstration
export function generateMockPredictions(datastreamId: number, hourlyData: HourlyDataPoint[]): HourlyDataPoint[] {
    if (!hourlyData || hourlyData.length === 0) return []

    const lastRealDataPoint = hourlyData[hourlyData.length - 1]
    const lastValue = lastRealDataPoint.average || 0
    const lastTime = new Date(lastRealDataPoint.endTime)

    // Generate 24 hours of predictions
    const predictions: HourlyDataPoint[] = []
    let currentValue = lastValue
    let currentTime = new Date(lastTime)

    for (let i = 0; i < 24; i++) {
        // Move to next hour
        currentTime = new Date(currentTime)
        currentTime.setHours(currentTime.getHours() + 1)

        // Generate a somewhat realistic prediction with some randomness
        // and a slight trend based on recent data
        const randomFactor = 0.9 + Math.random() * 0.2 // 0.9 to 1.1

        // Add some cyclical pattern for realism
        const hourOfDay = currentTime.getHours()
        const timeOfDayFactor =
            hourOfDay >= 6 && hourOfDay <= 18
                ? 1 + Math.sin(((hourOfDay - 6) * Math.PI) / 12) * 0.1 // Daytime increase
                : 1 - Math.sin(((hourOfDay - 18) * Math.PI) / 12) * 0.05 // Nighttime decrease

        currentValue = currentValue * randomFactor * timeOfDayFactor

        // Create proper hour slots with XX:00 to XX:59 format
        const hourStart = new Date(currentTime)
        hourStart.setMinutes(0, 0, 0)
        const hourEnd = new Date(currentTime)
        hourEnd.setMinutes(59, 59, 999)

        predictions.push({
            hour: formatHourLabel(hourStart),
            timestamp: hourStart.toISOString(),
            startTime: hourStart,
            endTime: hourEnd,
            predicted: true,
            values: [],
            average: currentValue,
        })
    }

    return predictions
}
