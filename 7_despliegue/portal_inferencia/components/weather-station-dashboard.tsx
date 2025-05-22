"use client"

import { useState, useEffect } from "react"
import { Loader2 } from "lucide-react"
import { LoadingIndicator } from "@/components/loading-indicator"
import { StationInfo } from "@/components/station-info"
import { NoDataMessage } from "@/components/no-data-message"
import { SensorChart } from "@/components/sensor-chart"
import "@/lib/chart-setup" // Import the chart setup to register components
import {
  type Station,
  type Location,
  type Datastream,
  type Observation,
  type HourlyDataPoint,
  type DebugInfo,
  type PredictionJob,
  LOADING_STEPS,
  type LoadingState,
} from "@/lib/types"
import { fetchApi, isAverageDatastream } from "@/lib/api"
import { calculateHourlyAverages } from "@/lib/data-processing"
import { submitPredictionJob, checkPredictionJobStatus } from "@/lib/prediction-service"
import { StylishHeader } from "@/components/stylish-header"

export function WeatherStationDashboard() {
  // State for stations and data
  const [stations, setStations] = useState<Station[]>([])
  const [locations, setLocations] = useState<Record<number, Location>>({})
  const [selectedStation, setSelectedStation] = useState<Station | null>(null)
  const [stationDatastreams, setStationDatastreams] = useState<Datastream[]>([])
  const [observations, setObservations] = useState<Record<number, Observation[]>>({})
  const [hourlyAverages, setHourlyAverages] = useState<Record<number, HourlyDataPoint[]>>({})

  // Loading states
  const [loading, setLoading] = useState(true)
  const [loadingDatastreams, setLoadingDatastreams] = useState(false)
  const [loadingObservations, setLoadingObservations] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Enhanced loading state
  const [loadingStep, setLoadingStep] = useState(LOADING_STEPS.IDLE)
  const [loadingProgress, setLoadingProgress] = useState({ current: 0, total: 0, percent: 0 })
  const [loadingDetail, setLoadingDetail] = useState("")

  // Debug state
  const [debugInfo, setDebugInfo] = useState<Record<string, any>>({})

  // Prediction states
  const [predictingDatastreamId, setPredictingDatastreamId] = useState<number | null>(null)
  const [predictions, setPredictions] = useState<Record<number, HourlyDataPoint[]>>({})
  const [predictionJobs, setPredictionJobs] = useState<Record<number, PredictionJob>>({})

  // Fetch all pages of observations for a datastream
  async function fetchAllObservations(datastreamId: number) {
    // Get last 48 hours - ensure we use the full 48 hours
    const now = new Date()
    const twoDaysAgo = new Date(now.getTime() - 48 * 60 * 60 * 1000)

    // Format date in ISO 8601 format for filtering
    const timeFilterISO = twoDaysAgo.toISOString()

    // Initial URL with time filter to only fetch data from the last 48 hours
    // Using OData $filter to limit results to our time window
    let url = `/observations/?datastream=${datastreamId}&$orderby=resultTime desc&$top=1000&$filter=resultTime ge ${encodeURIComponent(timeFilterISO)}`

    // Array to collect all observations
    let allObservations: Observation[] = []
    let pageCount = 0
    let hasMorePages = true
    let reachedTimeLimit = false

    // Fetch pages until we have all data in our time range
    while (hasMorePages && !reachedTimeLimit && pageCount < 10) {
      pageCount++
      setLoadingDetail(`Fetching page ${pageCount} for datastream ${datastreamId}`)

      const response = await fetchApi(url)
      const observations = response.results || []

      if (observations.length === 0) {
        // No observations in this page, stop fetching
        hasMorePages = false
        continue
      }

      // Add this page's observations to our collection
      allObservations = [...allObservations, ...observations]

      // Check if we've reached our time limit by examining the oldest observation in this page
      if (observations.length > 0) {
        // Sort by time to find the oldest observation in this page
        const sortedObservations = [...observations].sort(
          (a, b) => new Date(a.resultTime).getTime() - new Date(b.resultTime).getTime(),
        )

        const oldestObsTime = new Date(sortedObservations[0].resultTime)

        // Log the time range of this page
        const newestObsTime = new Date(sortedObservations[sortedObservations.length - 1].resultTime)
        console.log(`Page ${pageCount} time range: ${oldestObsTime.toISOString()} to ${newestObsTime.toISOString()}`)

        // If the oldest observation is already older than our time window, we can stop
        if (oldestObsTime < twoDaysAgo) {
          console.log(`Reached time limit at page ${pageCount} - oldest observation: ${oldestObsTime.toISOString()}`)
          reachedTimeLimit = true
        }
      }

      // Check if there's a next page
      if (response.next && !reachedTimeLimit) {
        url = response.next
        // Update progress for pagination
        setLoadingStep(LOADING_STEPS.PAGINATION)
        setLoadingDetail(`Loading additional data (page ${pageCount + 1})`)
      } else {
        hasMorePages = false
      }
    }

    const stopReason = reachedTimeLimit
      ? "reached 48-hour time limit"
      : pageCount >= 10
        ? "reached maximum page count"
        : "no more pages available"

    console.log(
      `Datastream ${datastreamId}: Fetched ${pageCount} pages with ${allObservations.length} total observations (${stopReason})`,
    )

    // Filter observations to the 48-hour window (just to be safe)
    const filteredObservations = allObservations.filter((obs) => {
      const obsTime = new Date(obs.resultTime)
      return obsTime >= twoDaysAgo && obsTime <= now
    })

    // Sort observations by time (oldest first)
    filteredObservations.sort((a, b) => new Date(a.resultTime).getTime() - new Date(b.resultTime).getTime())

    // Debug info
    const allDates = allObservations.map((obs) => new Date(obs.resultTime))
    const minDate = allDates.length > 0 ? new Date(Math.min(...allDates.map((d) => d.getTime()))) : null
    const maxDate = allDates.length > 0 ? new Date(Math.max(...allDates.map((d) => d.getTime()))) : null

    const filteredDates = filteredObservations.map((obs) => new Date(obs.resultTime))
    const filteredMinDate =
      filteredDates.length > 0 ? new Date(Math.min(...filteredDates.map((d) => d.getTime()))) : null
    const filteredMaxDate =
      filteredDates.length > 0 ? new Date(Math.max(...filteredDates.map((d) => d.getTime()))) : null

    const debugData: DebugInfo = {
      totalObservations: allObservations.length,
      filteredObservations: filteredObservations.length,
      pagesLoaded: pageCount,
      stopReason: stopReason,
      firstObservation: filteredMinDate ? filteredMinDate.toISOString() : null,
      lastObservation: filteredMaxDate ? filteredMaxDate.toISOString() : null,
      minDate: minDate ? minDate.toISOString() : null,
      maxDate: maxDate ? maxDate.toISOString() : null,
    }

    return { observations: filteredObservations, debug: debugData }
  }

  // Load observations for datastreams
  async function loadObservations(datastreamsList: Datastream[]) {
    try {
      setLoadingObservations(true)
      setLoadingStep(LOADING_STEPS.OBSERVATIONS)
      setLoadingProgress({ current: 0, total: datastreamsList.length, percent: 0 })
      setLoadingDetail(`Starting to fetch data for ${datastreamsList.length} sensors`)

      // Get last 48 hours - ensure we use the full 48 hours
      const now = new Date()
      const twoDaysAgo = new Date(now.getTime() - 48 * 60 * 60 * 1000)

      // Format dates for debugging
      console.log(`Target time range: ${twoDaysAgo.toISOString()} to ${now.toISOString()}`)

      // Create a map to store observations by datastream ID
      const obsMap: Record<number, Observation[]> = {}
      const hourlyMap: Record<number, HourlyDataPoint[]> = {}
      const debugData: Record<number, DebugInfo> = {}

      // Process datastreams in batches to improve UI responsiveness
      const batchSize = 2 // Reduced batch size since we're doing more work per datastream
      const batches = Math.ceil(datastreamsList.length / batchSize)

      for (let batchIndex = 0; batchIndex < batches; batchIndex++) {
        const start = batchIndex * batchSize
        const end = Math.min(start + batchSize, datastreamsList.length)
        const batchDatastreams = datastreamsList.slice(start, end)

        // Update progress once per batch
        setLoadingProgress({
          current: start,
          total: datastreamsList.length,
          percent: Math.round((start / datastreamsList.length) * 100),
        })
        setLoadingDetail(`Fetching data for sensors ${start + 1}-${end} of ${datastreamsList.length}`)

        // Process this batch of datastreams in parallel
        const batchPromises = batchDatastreams.map(async (datastream) => {
          try {
            // Fetch all pages of observations for this datastream
            const { observations, debug } = await fetchAllObservations(datastream.id)

            return {
              id: datastream.id,
              observations: observations,
              debug: debug,
            }
          } catch (err) {
            console.error(`Error loading observations for datastream ${datastream.id}:`, err)
            return { id: datastream.id, observations: [], debug: { error: (err as Error).message } }
          }
        })

        // Wait for all datastreams in this batch to complete
        const batchResults = await Promise.all(batchPromises)

        // Add results to our maps
        batchResults.forEach(({ id, observations, debug }) => {
          if (observations.length > 0) {
            obsMap[id] = observations
          }
          debugData[id] = debug
        })
      }

      // Store debug info
      setDebugInfo({
        targetTimeRange: {
          start: twoDaysAgo.toISOString(),
          end: now.toISOString(),
        },
        datastreams: debugData,
      })

      // Final observation progress update
      setLoadingProgress({
        current: datastreamsList.length,
        total: datastreamsList.length,
        percent: 100,
      })
      setLoadingDetail(`Fetched data for all ${datastreamsList.length} sensors`)

      // Process the data
      setLoadingStep(LOADING_STEPS.PROCESSING)
      setLoadingProgress({ current: 0, total: Object.keys(obsMap).length, percent: 0 })
      setLoadingDetail("Processing sensor data and calculating hourly averages")

      // Calculate hourly averages for each datastream with observations
      const datastreamIds = Object.keys(obsMap).map((id) => Number.parseInt(id))

      // Process all hourly averages at once
      const hourlyAveragesResults = datastreamIds.map((datastreamId) => {
        const observations = obsMap[datastreamId]
        return {
          id: datastreamId,
          hourlyData: calculateHourlyAverages(observations, twoDaysAgo, now),
        }
      })

      // Update the hourly map with all results
      hourlyAveragesResults.forEach(({ id, hourlyData }) => {
        hourlyMap[id] = hourlyData
      })

      // Final processing update
      setLoadingProgress({
        current: datastreamIds.length,
        total: datastreamIds.length,
        percent: 100,
      })
      setLoadingDetail("Data processing complete")

      // Set the final data
      setObservations(obsMap)
      setHourlyAverages(hourlyMap)

      // Complete loading
      setLoadingStep(LOADING_STEPS.COMPLETE)
    } catch (err) {
      console.error("Failed to load observations:", err)
      setError("Failed to load observation data. Please try again later.")
    } finally {
      setLoadingObservations(false)
    }
  }

  // Function to fetch predictions for a datastream using the job-based API
  async function fetchPredictions(datastream: Datastream) {
    try {
      // Set the datastream as predicting
      setPredictingDatastreamId(datastream.id)

      // Initialize the prediction job status
      setPredictionJobs((prev) => ({
        ...prev,
        [datastream.id]: {
          jobId: "",
          datastreamId: datastream.id,
          status: "submitted",
          lastChecked: new Date(),
        },
      }))

      // Prepare data for the prediction API
      const allSensorData: Record<string, any> = {}

      // Get all datastreams with data
      const datastreamsWithData = stationDatastreams.filter(
        (ds) => observations[ds.id] && observations[ds.id].length > 0,
      )

      // Collect hourly averages for all sensors
      datastreamsWithData.forEach((ds) => {
        if (hourlyAverages[ds.id]) {
          allSensorData[ds.id] = {
            name: ds.name,
            description: ds.description,
            unitSymbol: ds.unitOfMeasurement?.symbol || "",
            hourlyData: hourlyAverages[ds.id],
          }
        }
      })

      // Submit the prediction job
      const jobId = await submitPredictionJob(
        datastream,
        selectedStation?.id || 0,
        selectedStation?.name || "",
        allSensorData,
      )

      console.log(`Prediction job submitted with ID: ${jobId}`)

      // Update the job status
      setPredictionJobs((prev) => ({
        ...prev,
        [datastream.id]: {
          ...prev[datastream.id],
          jobId: jobId,
          status: "submitted",
          lastChecked: new Date(),
        },
      }))

      // Start polling for job status
      pollJobStatus(jobId, datastream)
    } catch (err) {
      console.error("Failed to submit prediction job:", err)
      setError(`Failed to submit prediction job: ${(err as Error).message}`)

      // Update job status to failed
      setPredictionJobs((prev) => ({
        ...prev,
        [datastream.id]: {
          ...prev[datastream.id],
          status: "failed",
          error: (err as Error).message,
          lastChecked: new Date(),
        },
      }))

      setPredictingDatastreamId(null)
    }
  }

  // Poll for job status
  async function pollJobStatus(jobId: string, datastream: Datastream) {
    try {
      // Check the job status
      const jobResponse = await checkPredictionJobStatus(jobId, datastream, hourlyAverages[datastream.id] || [])

      console.log(`Job ${jobId} status: ${jobResponse.status}`)

      // Update the job status
      setPredictionJobs((prev) => ({
        ...prev,
        [datastream.id]: {
          ...prev[datastream.id],
          status: jobResponse.status,
          error: jobResponse.error,
          lastChecked: new Date(),
        },
      }))

      // Handle different job statuses
      if (jobResponse.status === "completed" && jobResponse.result) {
        // Job completed successfully, update predictions
        setPredictions((prev) => ({
          ...prev,
          [datastream.id]: jobResponse.result,
        }))

        // No longer predicting
        setPredictingDatastreamId(null)
      } else if (jobResponse.status === "failed") {
        // Job failed
        console.error(`Prediction job failed: ${jobResponse.error}`)
        setError(`Prediction failed: ${jobResponse.error}`)

        // No longer predicting
        setPredictingDatastreamId(null)
      } else if (jobResponse.status === "in progress") {
        // Job still in progress, continue polling
        setTimeout(() => pollJobStatus(jobId, datastream), 2000) // Poll every 2 seconds
      } else {
        // Unknown status
        console.warn(`Unknown job status: ${jobResponse.status}`)

        // No longer predicting
        setPredictingDatastreamId(null)
      }
    } catch (err) {
      console.error("Failed to check job status:", err)

      // Update job status to failed
      setPredictionJobs((prev) => ({
        ...prev,
        [datastream.id]: {
          ...prev[datastream.id],
          status: "failed",
          error: (err as Error).message,
          lastChecked: new Date(),
        },
      }))

      setPredictingDatastreamId(null)
    }
  }

  // Load stations and locations on component mount
  useEffect(() => {
    async function loadStationsAndLocations() {
      try {
        setLoading(true)
        setError(null)

        // Fetch all locations first
        const locationsResponse = await fetchApi("/locations/")
        const locationsList = locationsResponse.results || []

        // Create a map of location IDs to location objects
        const locationsMap: Record<number, Location> = {}
        locationsList.forEach((location: Location) => {
          locationsMap[location.id] = location
        })
        setLocations(locationsMap)
        console.log(`Loaded ${locationsList.length} locations`)

        // Fetch all things (stations)
        const thingsResponse = await fetchApi("/things/")
        const stationsList = thingsResponse.results || []

        setStations(stationsList)
        console.log(`Loaded ${stationsList.length} stations`)
      } catch (err) {
        console.error("Failed to load stations or locations:", err)
        setError("Failed to load stations. Please try again later.")
      } finally {
        setLoading(false)
      }
    }

    loadStationsAndLocations()
  }, [])

  // Load datastreams when a station is selected
  useEffect(() => {
    if (!selectedStation) return

    async function loadDatastreams() {
      try {
        // Reset states
        setLoadingDatastreams(true)
        setStationDatastreams([])
        setObservations({})
        setHourlyAverages({})
        setError(null)
        setDebugInfo({})
        setPredictions({})
        setPredictionJobs({})

        // Set initial loading state
        setLoadingStep(LOADING_STEPS.DATASTREAMS)
        setLoadingProgress({ current: 0, total: 1, percent: 0 })
        setLoadingDetail("Fetching available sensors for this station")

        // Get datastreams for this station
        const datastreamResponse = await fetchApi(`/datastreams/?thing=${selectedStation.id}`)
        const allDatastreamsList = datastreamResponse.results || []

        // Filter to only include average datastreams (not min/max)
        const averageDatastreams = allDatastreamsList.filter(isAverageDatastream)

        console.log(
          `Filtered ${allDatastreamsList.length} datastreams to ${averageDatastreams.length} average datastreams`,
        )

        if (averageDatastreams.length > 0) {
          setStationDatastreams(averageDatastreams)
          console.log(`Loaded ${averageDatastreams.length} average datastreams for station ${selectedStation.id}`)

          // Update loading state
          setLoadingProgress({ current: 1, total: 1, percent: 100 })
          setLoadingDetail(`Found ${averageDatastreams.length} sensors (average values only)`)

          // Load observations for each datastream
          await loadObservations(averageDatastreams)
        } else {
          console.log(`No average datastreams found for station ${selectedStation.id}`)
          setLoadingStep(LOADING_STEPS.COMPLETE)
          setLoadingDetail("No average sensors found for this station")
        }
      } catch (err) {
        console.error(`Failed to load datastreams for station ${selectedStation.id}:`, err)
        setError("Failed to load sensor data. Please try again later.")
      } finally {
        setLoadingDatastreams(false)
      }
    }

    loadDatastreams()
  }, [selectedStation])

  // Filter datastreams to only show those with observations
  const datastreamsWithData = stationDatastreams.filter(
    (datastream) => observations[datastream.id] && observations[datastream.id].length > 0,
  )

  // Determine if we're in a loading state
  const isLoading = loadingDatastreams || loadingObservations

  // Create loading state object for LoadingIndicator
  const loadingState: LoadingState = {
    step: loadingStep,
    progress: loadingProgress,
    detail: loadingDetail,
  }

  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col space-y-4">
        {/* Header with station selector */}
        <StylishHeader
          stations={stations}
          locations={locations}
          selectedStation={selectedStation}
          onSelectStation={setSelectedStation}
          disabled={loading || isLoading}
        />

        {/* Error message */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <p>{error}</p>
          </div>
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary mr-2" />
            <p>Loading stations...</p>
          </div>
        )}

        {/* Station info and charts */}
        {selectedStation && !loading && (
          <div className="mt-4">
            {/* Station information */}
            <StationInfo station={selectedStation} locations={locations} />

            {/* Enhanced loading indicator with progress */}
            {isLoading && <LoadingIndicator loadingState={loadingState} />}

            {/* No data message */}
            {!isLoading && datastreamsWithData.length === 0 && (
              <NoDataMessage datastreamCount={stationDatastreams.length} />
            )}

            {/* Datastream charts - only show datastreams with data */}
            {!isLoading && datastreamsWithData.length > 0 && (
              <div className="grid grid-cols-1 gap-6">
                {datastreamsWithData.map((datastream, index) => (
                  <SensorChart
                    key={datastream.id}
                    datastream={datastream}
                    observations={observations[datastream.id] || []}
                    hourlyData={hourlyAverages[datastream.id] || []}
                    predictionData={predictions[datastream.id] || []}
                    debugInfo={debugInfo.datastreams?.[datastream.id] || {}}
                    index={index}
                    onFetchPredictions={fetchPredictions}
                    isPredicting={predictingDatastreamId === datastream.id}
                    predictionStatus={predictionJobs[datastream.id]?.status}
                    predictionError={predictionJobs[datastream.id]?.error}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
