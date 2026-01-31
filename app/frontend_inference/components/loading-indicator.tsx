import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { Loader2 } from "lucide-react"
import type { LoadingState } from "@/lib/types"

interface LoadingIndicatorProps {
    loadingState: LoadingState
}

export function LoadingIndicator({ loadingState }: LoadingIndicatorProps) {
    const { step, progress, detail } = loadingState

    return (
        <Card className="mb-6">
            <CardContent className="pt-6">
                <div className="flex flex-col items-center space-y-4">
                    <div className="flex items-center">
                        <Loader2 className="h-6 w-6 animate-spin text-primary mr-3" />
                        <h3 className="text-lg font-medium">{step}</h3>
                    </div>

                    <div className="w-full max-w-md space-y-2">
                        <Progress value={progress.percent} className="h-2" />

                        <div className="flex justify-between text-sm text-muted-foreground">
                            <span>
                                {progress.current} of {progress.total}
                                {progress.total > 0 && ` (${progress.percent}%)`}
                            </span>
                            <span>{detail}</span>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
