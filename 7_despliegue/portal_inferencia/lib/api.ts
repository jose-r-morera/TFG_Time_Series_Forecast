// API configuration
export const API_BASE = "https://sensores.grafcan.es/api/v1.0"
export const API_KEY = "RnmI6Wig.b2PNFs0HfcLw93jNi1ZsXudlcQRPV0MF"

// Helper function for API requests
export async function fetchApi(endpoint: string) {
    // If endpoint is a full URL, use it directly
    const url = endpoint.startsWith("http") ? endpoint : `${API_BASE}${endpoint}`

    console.log(`Fetching: ${url}`)
    try {
        const response = await fetch(url, {
            headers: {
                Authorization: `Api-Key ${API_KEY}`,
                "Content-Type": "application/json",
            },
            cache: "no-store",
        })

        if (!response.ok) {
            const errorText = await response.text()
            console.error(`API error (${response.status}):`, errorText)
            throw new Error(`API error: ${response.status}`)
        }

        const data = await response.json()

        // Log the count of results if available
        if (data.results && Array.isArray(data.results)) {
            console.log(`API returned ${data.results.length} results for ${url}`)
        }

        return data
    } catch (error) {
        console.error(`Error fetching ${url}:`, error)
        throw error
    }
}

// Extract ID from URI
export function extractIdFromUri(uri: string | null) {
    if (!uri || typeof uri !== "string") return null
    const parts = uri.split("/")
    return Number.parseInt(parts[parts.length - 2], 10)
}

// Check if a datastream is an average measurement (not min or max)
export function isAverageDatastream(datastream: any) {
    const name = datastream.name?.toLowerCase() || ""
    const description = datastream.description?.toLowerCase() || ""

    // Log the datastream name for debugging
    console.log(`Checking datastream: ${datastream.name}`)

    // More comprehensive check for min/max variations including with periods
    const minMaxPatterns = [
        "(min)",
        "(max)",
        "(min.)",
        "(max.)",
        "(minimum)",
        "(maximum)",
        "(mín)",
        "(máx)",
        "(mín.)",
        "(máx.)",
        "minimum",
        "maximum",
        "máxima",
        "mínima",
        "max.",
        "min.",
    ]

    // Check if any min/max pattern is found in the name or description
    for (const pattern of minMaxPatterns) {
        if (name.includes(pattern) || description.includes(pattern)) {
            console.log(`  Filtered out: contains "${pattern}"`)
            return false
        }
    }

    // Check if it's explicitly an average datastream
    const avgPatterns = [
        "(avg)",
        "(average)",
        "(mean)",
        "(avg.)",
        "(average.)",
        "(mean.)",
        "(media)",
        "(promedio)",
        "average",
        "mean",
        "media",
        "promedio",
    ]

    for (const pattern of avgPatterns) {
        if (name.includes(pattern) || description.includes(pattern)) {
            console.log(`  Included: contains "${pattern}"`)
            return true
        }
    }

    // If the name doesn't contain any specific indicators, check if it's a base measurement
    // without qualifiers (likely to be an average)
    const hasQualifier =
        name.includes("(") || name.includes("[") || description.includes("(") || description.includes("[")

    if (!hasQualifier) {
        console.log(`  Included: no qualifiers, assumed to be average`)
        return true
    }

    // For any other case, log and exclude
    console.log(`  Filtered out: has qualifiers but not identified as average`)
    return false
}
