export interface Station {
    id: number
    name: string
    description?: string
    location_set: string[]
    [key: string]: any
}

export interface Location {
    id: number
    name: string
    location: string
    [key: string]: any
}

export interface Datastream {
    id: number
    name: string
    description?: string
    unitOfMeasurement: any
    [key: string]: any
}

export interface Observation {
    id: number
    result: string
    resultTime: string
    [key: string]: any
}

export interface HourlyDataPoint {
    hour: string
    timestamp: string
    startTime: Date
    endTime: Date
    values: number[]
    average: number | null
    predicted?: boolean
}

export interface DebugInfo {
    totalObservations?: number
    filteredObservations?: number
    pagesLoaded?: number
    stopReason?: string
    firstObservation?: string | null
    lastObservation?: string | null
    minDate?: string | null
    maxDate?: string | null
    error?: string
}

export interface LoadingState {
    step: string
    progress: {
        current: number
        total: number
        percent: number
    }
    detail: string
}

export const LOADING_STEPS = {
    IDLE: "idle",
    DATASTREAMS: "Loading datastreams",
    OBSERVATIONS: "Fetching sensor observations",
    PAGINATION: "Loading additional pages",
    PROCESSING: "Processing data",
    COMPLETE: "Complete",
}

// Job status types
export type JobStatus = "submitted" | "in progress" | "completed" | "failed" | string

export interface JobResponse {
    job_id: string
    status: JobStatus
    result?: any
    error?: string
}

export interface PredictionJob {
    jobId: string
    datastreamId: number
    status: JobStatus
    result?: HourlyDataPoint[]
    error?: string
    lastChecked: Date
}
