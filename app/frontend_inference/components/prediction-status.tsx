import { Loader2, AlertCircle, CheckCircle2 } from "lucide-react"
import type { JobStatus } from "@/lib/types"

interface PredictionStatusProps {
    status: JobStatus
    error?: string
}

export function PredictionStatus({ status, error }: PredictionStatusProps) {
    return (
        <div className="flex items-center justify-center space-x-2 text-sm">
            {status === "submitted" && (
                <>
                    <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                    <span className="text-blue-600">Submitting prediction request...</span>
                </>
            )}

            {status === "in progress" && (
                <>
                    <Loader2 className="h-4 w-4 animate-spin text-amber-500" />
                    <span className="text-amber-600">Processing prediction...</span>
                </>
            )}

            {status === "completed" && (
                <>
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span className="text-green-600">Prediction complete</span>
                </>
            )}

            {status === "failed" && (
                <>
                    <AlertCircle className="h-4 w-4 text-red-500" />
                    <span className="text-red-600">Prediction failed{error ? `: ${error}` : ""}</span>
                </>
            )}

            {!["submitted", "in progress", "completed", "failed"].includes(status) && (
                <>
                    <Loader2 className="h-4 w-4 animate-spin text-gray-500" />
                    <span className="text-gray-600">Status: {status}</span>
                </>
            )}
        </div>
    )
}
