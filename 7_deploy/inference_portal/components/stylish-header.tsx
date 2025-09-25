import { Cloud, CloudSun } from "lucide-react"
import { StationSelector } from "./station-selector"
import type { Station, Location } from "@/lib/types"

interface StylishHeaderProps {
    stations: Station[]
    locations: Record<number, Location>
    selectedStation: Station | null
    onSelectStation: (station: Station | null) => void
    disabled: boolean
}

export function StylishHeader({ stations, locations, selectedStation, onSelectStation, disabled }: StylishHeaderProps) {
    return (
        <header className="sticky top-0 z-10 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex flex-col sm:flex-row h-auto sm:h-20 items-center px-4 py-4 sm:py-0">
                <div className="flex items-center mb-3 sm:mb-0 sm:mr-4">
                    <CloudSun className="h-8 w-8 mr-3 text-blue-500" />
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
                        Canary Islands Weather Stations
                    </h1>
                </div>
                <div className="flex flex-1 items-center justify-center sm:justify-end">
                    <div className="relative">
                        <Cloud className="absolute left-[-30px] top-1/2 transform -translate-y-1/2 h-5 w-5 text-blue-400 opacity-70" />
                        <StationSelector
                            stations={stations}
                            locations={locations}
                            selectedStation={selectedStation}
                            onSelectStation={onSelectStation}
                            disabled={disabled}
                        />
                    </div>
                </div>
            </div>
        </header>
    )
}
