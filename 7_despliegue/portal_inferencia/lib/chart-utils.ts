import type { Datastream, HourlyDataPoint } from "./types"

// Get unit symbol from datastream
export function getUnitSymbol(datastream: Datastream): string {
    if (!datastream.unitOfMeasurement) return ""

    try {
        if (typeof datastream.unitOfMeasurement === "object") {
            return datastream.unitOfMeasurement.symbol || ""
        }

        const unitData = JSON.parse(datastream.unitOfMeasurement)
        return unitData.symbol || ""
    } catch (e) {
        return ""
    }
}

// Generate chart colors
export function getChartColor(index: number): string {
    const colors = [
        "#2563eb", // blue-600
        "#16a34a", // green-600
        "#dc2626", // red-600
        "#9333ea", // purple-600
        "#ea580c", // orange-600
    ]
    return colors[index % colors.length]
}

// Prepare chart datasets
export function prepareChartDatasets(
    hourlyData: HourlyDataPoint[],
    predictionData: HourlyDataPoint[],
    unitSymbol: string,
    chartColor: string,
) {
    const hasPredictions = predictionData.length > 0

    // Basic dataset for historical data
    const datasets = [
        {
            label: `Hourly Average ${unitSymbol ? `(${unitSymbol})` : ""}`,
            data: hourlyData.map((hour) => hour.average),
            fill: false,
            borderColor: chartColor,
            backgroundColor: chartColor,
            tension: 0.1,
            pointRadius: 3,
            spanGaps: true, // This connects lines across null values
        },
    ]

    // Add prediction dataset if available
    if (hasPredictions) {
        // Find the last valid data point from historical data
        const lastValidHistoricalIndex = hourlyData
            .map((h) => h.average)
            .lastIndexOf(hourlyData.filter((h) => h.average !== null).pop()?.average)

        // Create a dataset for prediction that includes the last historical point
        const predictionDataset = {
            label: `Prediction ${unitSymbol ? `(${unitSymbol})` : ""}`,
            data: Array(hourlyData.length + predictionData.length)
                .fill(null)
                .map((_, i) => {
                    if (i === lastValidHistoricalIndex) {
                        // Include the last valid historical point to create connection
                        return hourlyData[lastValidHistoricalIndex].average
                    } else if (i > lastValidHistoricalIndex && i < hourlyData.length) {
                        // Nulls for the rest of the historical period
                        return null
                    } else if (i >= hourlyData.length) {
                        // Values for prediction period
                        return predictionData[i - hourlyData.length].average
                    }
                    return null
                }),
            fill: false,
            borderColor: "#f97316", // orange-500
            backgroundColor: "#f97316",
            borderWidth: 2,
            borderDash: [5, 5], // Dashed line for predictions
            pointRadius: 3,
            pointStyle: "triangle",
            tension: 0.1,
            spanGaps: true, // Connect across the transition point
        }

        datasets.push(predictionDataset)
    }

    return datasets
}

// Calculate statistics from hourly data
export function calculateStatistics(hourlyData: HourlyDataPoint[]) {
    // Calculate statistics using only valid values
    const validHourlyValues = hourlyData.filter((hour) => hour.average !== null).map((hour) => hour.average as number)

    return {
        min: validHourlyValues.length > 0 ? Math.min(...validHourlyValues) : null,
        max: validHourlyValues.length > 0 ? Math.max(...validHourlyValues) : null,
        avg:
            validHourlyValues.length > 0
                ? (validHourlyValues.reduce((sum, val) => sum + val, 0) / validHourlyValues.length).toFixed(2)
                : null,
    }
}
