import { MapPin } from "lucide-react"
import type { Station, Location } from "@/lib/types"
import { extractIdFromUri } from "@/lib/api"

interface StationInfoProps {
    station: Station
    locations: Record<number, Location>
}

export function StationInfo({ station, locations }: StationInfoProps) {
    // Get location information for a station
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

    // Format location coordinates from GeoJSON
    function formatLocationCoordinates(location: Location) {
        if (!location || !location.location) return "Unknown location"

        try {
            const geoJson = JSON.parse(location.location)
            if (geoJson.type === "Point" && geoJson.coordinates && geoJson.coordinates.length >= 2) {
                const [longitude, latitude] = geoJson.coordinates
                return `${location.name || "Location"} (${latitude.toFixed(6)}, ${longitude.toFixed(6)})`
            }
            return location.name || "Unknown location"
        } catch (e) {
            return location.name || "Unknown location"
        }
    }

    const selectedStationLocation = getStationLocation(station)
    const locationDisplay = selectedStationLocation
        ? formatLocationCoordinates(selectedStationLocation)
        : "Unknown location"

    return (
        <div className="mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
                {station.name || `Station ${station.id}`}
            </h2>
            <div className="flex items-center text-muted-foreground mt-1">
                <MapPin className="h-4 w-4 mr-1 text-blue-500" />
                <p>{locationDisplay}</p>
            </div>
            <p className="text-sm text-muted-foreground mt-2">{station.description || ""}</p>
        </div>
    )
}
