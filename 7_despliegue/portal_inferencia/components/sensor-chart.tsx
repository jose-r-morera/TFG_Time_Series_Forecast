"use client"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Line } from "react-chartjs-2"
import { Loader2, Play } from "lucide-react"
import type { Datastream, HourlyDataPoint, DebugInfo, Observation, JobStatus } from "@/lib/types"
import { getUnitSymbol, getChartColor, prepareChartDatasets, calculateStatistics } from "@/lib/chart-utils"
import { PredictionStatus } from "@/components/prediction-status"
import "@/lib/chart-setup" // Import the chart setup to register components

interface SensorChartProps {
    datastream: Datastream
    observations: Observation[]
    hourlyData: HourlyDataPoint[]
    predictionData: HourlyDataPoint[]
    debugInfo: DebugInfo
    index: number
    onFetchPredictions: (datastream: Datastream) => void
    isPredicting: boolean
    predictionStatus?: JobStatus
    predictionError?: string
}

export function SensorChart({
    datastream,
    observations,
    hourlyData,
    predictionData,
    debugInfo,
    index,
    onFetchPredictions,
    isPredicting,
    predictionStatus,
    predictionError,
}: SensorChartProps) {
    const unitSymbol = getUnitSymbol(datastream)
    const chartColor = getChartColor(index)
    const hasPredictions = predictionData.length > 0
    const stats = calculateStatistics(hourlyData)

    // Prepare chart datasets
    const datasets = prepareChartDatasets(hourlyData, predictionData, unitSymbol, chartColor)

    return (
        <Card className="relative">
            <CardHeader className="pb-2">
                <div className="flex justify-between items-center">
                    <div>
                        <CardTitle className="text-lg">{datastream.name || `Sensor ${datastream.id}`}</CardTitle>
                        <CardDescription>{datastream.description || ""}</CardDescription>
                    </div>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-4 right-4"
                        onClick={() => onFetchPredictions(datastream)}
                        disabled={isPredicting}
                    >
                        {isPredicting ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
                        <span className="sr-only">
                            {isPredicting ? "Loading prediction" : hasPredictions ? "Update prediction" : "Predict future values"}
                        </span>
                    </Button>
                </div>

                {/* Show prediction status if we're predicting or have a status */}
                {(isPredicting || predictionStatus) && (
                    <div className="mt-2">
                        <PredictionStatus status={predictionStatus || "submitted"} error={predictionError} />
                    </div>
                )}
            </CardHeader>
            <CardContent>
                <div className="h-[250px]">
                    <Line
                        data={{
                            labels: hasPredictions
                                ? [...hourlyData.map((hour) => hour.hour), ...predictionData.map((hour) => hour.hour)]
                                : hourlyData.map((hour) => hour.hour),
                            datasets: datasets,
                        }}
                        options={{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    title: {
                                        display: !!unitSymbol,
                                        text: unitSymbol,
                                    },
                                    beginAtZero: false,
                                },
                                x: {
                                    ticks: {
                                        maxRotation: 45,
                                        minRotation: 45,
                                        autoSkip: true,
                                        maxTicksLimit: 12, // Show fewer x-axis labels for readability
                                    },
                                    title: {
                                        display: true,
                                        text: hasPredictions
                                            ? `48-hour history + 24-hour prediction`
                                            : `48-hour period (${new Date(hourlyData[0]?.startTime).toLocaleDateString()} to ${new Date(hourlyData[hourlyData.length - 1]?.endTime).toLocaleDateString()})`,
                                        font: {
                                            size: 10,
                                        },
                                    },
                                },
                            },
                            plugins: {
                                tooltip: {
                                    callbacks: {
                                        label: (context) => {
                                            const value = context.raw
                                            if (value === null) return "No data available"

                                            const datasetLabel = context.dataset.label || ""
                                            const isPrediction = datasetLabel.includes("Prediction")

                                            return `${isPrediction ? "Predicted" : "Average"}: ${Number(value).toFixed(2)}${unitSymbol ? ` ${unitSymbol}` : ""}`
                                        },
                                        title: (tooltipItems) => {
                                            const index = tooltipItems[0].dataIndex
                                            const datasetIndex = tooltipItems[0].datasetIndex

                                            // Handle prediction dataset
                                            if (datasetIndex === 1 && hasPredictions) {
                                                // Adjust index for prediction data
                                                const adjustedIndex = index - hourlyData.length
                                                if (adjustedIndex >= 0 && adjustedIndex < predictionData.length) {
                                                    const slot = predictionData[adjustedIndex]
                                                    return `Prediction: ${new Date(slot.startTime).toLocaleString()} - ${new Date(slot.endTime).toLocaleTimeString()}`
                                                }
                                            }

                                            // Handle historical data
                                            if (index < hourlyData.length) {
                                                const slot = hourlyData[index]
                                                if (slot) {
                                                    return `${new Date(slot.startTime).toLocaleString()} - ${new Date(slot.endTime).toLocaleTimeString()}`
                                                }
                                            }

                                            return tooltipItems[0].label
                                        },
                                    },
                                },
                                legend: {
                                    display: true,
                                },
                            },
                        }}
                    />
                </div>

                {/* Statistics summary */}
                <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
                    <div className="border rounded p-2 text-center">
                        <div className="font-medium">Min</div>
                        <div>{stats.min !== null ? `${stats.min.toFixed(2)} ${unitSymbol}` : "N/A"}</div>
                    </div>
                    <div className="border rounded p-2 text-center">
                        <div className="font-medium">Max</div>
                        <div>{stats.max !== null ? `${stats.max.toFixed(2)} ${unitSymbol}` : "N/A"}</div>
                    </div>
                    <div className="border rounded p-2 text-center">
                        <div className="font-medium">Avg</div>
                        <div>{stats.avg !== null ? `${stats.avg} ${unitSymbol}` : "N/A"}</div>
                    </div>
                </div>

                <div className="mt-2 text-xs text-muted-foreground text-center">
                    Based on {observations.length} data points from {new Date(hourlyData[0]?.startTime).toLocaleDateString()} to{" "}
                    {new Date(hourlyData[hourlyData.length - 1]?.endTime).toLocaleDateString()}
                    {hasPredictions && (
                        <span className="ml-1 text-orange-500">
                            • Prediction extends to{" "}
                            {new Date(predictionData[predictionData.length - 1]?.endTime).toLocaleDateString()}
                        </span>
                    )}
                </div>

                {/* Debug info */}
                {debugInfo.totalObservations && (
                    <div className="mt-4 text-xs text-muted-foreground border-t pt-2">
                        <details>
                            <summary className="cursor-pointer">Data details</summary>
                            <div className="mt-1 space-y-1 pl-2">
                                <p>Total observations: {debugInfo.totalObservations}</p>
                                <p>Filtered observations: {debugInfo.filteredObservations}</p>
                                <p>Pages loaded: {debugInfo.pagesLoaded}</p>
                                <p>Stop reason: {debugInfo.stopReason}</p>
                                {debugInfo.firstObservation && debugInfo.lastObservation && (
                                    <p>
                                        Range: {new Date(debugInfo.firstObservation).toLocaleString()} to{" "}
                                        {new Date(debugInfo.lastObservation).toLocaleString()}
                                    </p>
                                )}
                                {debugInfo.minDate && debugInfo.maxDate && (
                                    <p>
                                        All data: {new Date(debugInfo.minDate).toLocaleString()} to{" "}
                                        {new Date(debugInfo.maxDate).toLocaleString()}
                                    </p>
                                )}
                            </div>
                        </details>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
