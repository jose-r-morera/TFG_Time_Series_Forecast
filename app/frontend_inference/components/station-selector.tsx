"use client"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { Station, Location } from "@/lib/types"
import { extractIdFromUri } from "@/lib/api"

interface StationSelectorProps {
    stations: Station[]
    locations: Record<number, Location>
    selectedStation: Station | null
    onSelectStation: (station: Station | null) => void
    disabled: boolean
}

export function StationSelector({
    stations,
    locations,
    selectedStation,
    onSelectStation,
    disabled,
}: StationSelectorProps) {
    // Get location for a station
    function getStationLocation(station: Station) {
        if (!station || !station.location_set || !station.location_set.length) {
            return null
        }

        // Extract location IDs from the location_set URIs
        const locationIds = station.location_set.map((uri) => extractIdFromUri(uri))

        // Find the first valid location
        for (const locId of locationIds) {
            if (locId && locations[locId]) {
                return locations[locId]
            }
        }

        return null
    }

    return (
        <Select
            value={selectedStation?.id?.toString() || ""}
            onValueChange={(value) => {
                const station = stations.find((s) => s.id.toString() === value)
                onSelectStation(station || null)
            }}
            disabled={disabled}
        >
            <SelectTrigger className="w-[350px]">
                <SelectValue placeholder={disabled ? "Loading stations..." : "Select a weather station"} />
            </SelectTrigger>
            <SelectContent className="max-h-[400px]">
                {stations.length === 0 ? (
                    <SelectItem value="none" disabled>
                        {disabled ? "Loading stations..." : "No stations found"}
                    </SelectItem>
                ) : (
                    (() => {
                        // Group stations by location
                        const stationsByLocation: Record<string, Station[]> = {}

                        stations.forEach((station) => {
                            const stationLocation = getStationLocation(station)
                            const locationName = stationLocation ? stationLocation.name : "Unknown location"

                            if (!stationsByLocation[locationName]) {
                                stationsByLocation[locationName] = []
                            }

                            stationsByLocation[locationName].push(station)
                        })

                        // Render grouped stations
                        return Object.entries(stationsByLocation).map(([locationName, locationStations]) => (
                            <div key={locationName} className="px-2 py-1.5">
                                <div className="text-sm font-semibold text-muted-foreground mb-1">{locationName}</div>
                                {locationStations.map((station) => (
                                    <SelectItem key={station.id} value={station.id.toString()}>
                                        {station.name || `Station ${station.id}`}
                                    </SelectItem>
                                ))}
                            </div>
                        ))
                    })()
                )}
            </SelectContent>
        </Select>
    )
}
