import { Card, CardContent } from "@/components/ui/card"
import { AlertCircle } from "lucide-react"

interface NoDataMessageProps {
    datastreamCount: number
}

export function NoDataMessage({ datastreamCount }: NoDataMessageProps) {
    return (
        <Card>
            <CardContent className="flex flex-col items-center justify-center h-40 space-y-2">
                <div className="flex items-center text-amber-600">
                    <AlertCircle className="h-5 w-5 mr-2" />
                    <p>No sensor data available for this station</p>
                </div>

                {datastreamCount > 0 && (
                    <p className="text-sm text-muted-foreground">
                        This station has {datastreamCount} sensor{datastreamCount !== 1 ? "s" : ""}, but none have data in the last
                        48 hours.
                    </p>
                )}
            </CardContent>
        </Card>
    )
}
