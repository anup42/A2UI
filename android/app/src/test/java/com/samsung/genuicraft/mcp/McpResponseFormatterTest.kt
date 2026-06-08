package com.samsung.genuicraft.mcp

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class McpResponseFormatterTest {

    @Test
    fun normalizeForStage3_fixesCommonMojibakeTokens() {
        val input = "Temp: 33Ã‚Â°C Ã¢â‚¬â€ feels like 35Ã‚Â°C Â· steady"
        val output = McpResponseFormatter.normalizeForStage3(input)

        assertEquals("Temp: 33°C — feels like 35°C · steady", output)
    }

    @Test
    fun buildDataSection_keepsWeatherAsSingleCompactTable() {
        val data = JsonObject().apply {
            addProperty("location", "Bengaluru")
            addProperty("country", "India")
            addProperty("requested_forecast_days", 2)
            add("current", JsonObject().apply {
                addProperty("temperature_2m", "27")
                addProperty("relative_humidity_2m", "62")
                addProperty("apparent_temperature", "29")
                addProperty("weather_code", 1)
                addProperty("wind_speed_10m", "8")
                addProperty("wind_direction_10m", "120")
                addProperty("uv_index", "4")
            })
            add("current_units", JsonObject().apply {
                addProperty("temperature_2m", "C")
                addProperty("relative_humidity_2m", "%")
                addProperty("apparent_temperature", "C")
                addProperty("wind_speed_10m", "km/h")
                addProperty("uv_index", "")
            })
            add("daily", JsonObject().apply {
                add("time", JsonArray().apply {
                    add("2026-06-08")
                    add("2026-06-09")
                })
                add("temperature_2m_max", JsonArray().apply {
                    add("31")
                    add("30")
                })
                add("temperature_2m_min", JsonArray().apply {
                    add("22")
                    add("21")
                })
                add("apparent_temperature_max", JsonArray().apply {
                    add("33")
                    add("32")
                })
                add("precipitation_probability_max", JsonArray().apply {
                    add("10")
                    add("15")
                })
                add("precipitation_sum", JsonArray().apply {
                    add("0")
                    add("0.2")
                })
                add("rain_sum", JsonArray().apply {
                    add("0")
                    add("0.2")
                })
                add("wind_speed_10m_max", JsonArray().apply {
                    add("12")
                    add("14")
                })
                add("wind_gusts_10m_max", JsonArray().apply {
                    add("18")
                    add("20")
                })
                add("wind_direction_10m_dominant", JsonArray().apply {
                    add("120")
                    add("140")
                })
                add("uv_index_max", JsonArray().apply {
                    add("5")
                    add("4")
                })
                add("weather_code", JsonArray().apply {
                    add(1)
                    add(2)
                })
            })
            add("daily_units", JsonObject().apply {
                addProperty("temperature_2m_max", "C")
                addProperty("temperature_2m_min", "C")
                addProperty("apparent_temperature_max", "C")
                addProperty("precipitation_probability_max", "%")
                addProperty("precipitation_sum", "mm")
                addProperty("rain_sum", "mm")
                addProperty("wind_speed_10m_max", "km/h")
                addProperty("wind_gusts_10m_max", "km/h")
                addProperty("uv_index_max", "")
            })
        }

        val output = McpResponseFormatter.buildDataSection(
            McpClient.McpResult(
                domain = McpSettings.Domain.WEATHER,
                success = true,
                data = data,
                rawJson = null,
                error = null
            ),
            queryText = "show weather in Bengaluru"
        )

        assertTrue(output.contains("Weather forecast table (domain: weather, preferredPresentation: cards)."))
        assertTrue(output.contains("| Day | Date | Condition | Temp | High | Low | Feels Like | Rain Chance | Rain | Wind | Gusts | Humidity | UV | Best Window | Morning | Afternoon | Evening | Night | What to wear |"))
        assertTrue(output.contains("| Today | 2026-06-08 |"))
        assertTrue(!output.contains("**Today:**"))
        assertTrue(!output.contains("current conditions block"))
    }

    @Test
    fun buildDataSection_preservesHotelPhotosInCompactTable() {
        val image1 = "https://lh3.googleusercontent.com/gps-cs-s/photo1=s287-w287-h192-n-k-no-v1"
        val image2 = "https://lh3.googleusercontent.com/gps-cs-s/photo2=s287-w287-h192-n-k-no-v1"
        val hotel = JsonObject().apply {
            addProperty("name", "Lemon Tree Hotel")
            addProperty("hotel_class", "4-star hotel")
            addProperty("overall_rating", 4.2)
            addProperty("reviews", 1200)
            addProperty("description", "Modern hotel near Whitefield.")
            addProperty("link", "https://example.com/book")
            addProperty("check_in_time", "2:00 PM")
            addProperty("check_out_time", "11:00 AM")
            add("rate_per_night", JsonObject().apply { addProperty("lowest", "INR 4,661") })
            add("amenities", JsonArray().apply {
                add("Free Wi-Fi")
                add("Pool")
            })
            add("gps_coordinates", JsonObject().apply {
                addProperty("latitude", 12.9725)
                addProperty("longitude", 77.718)
            })
            add("images", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("thumbnail", image1)
                    addProperty("original_image", image1)
                })
                add(JsonObject().apply {
                    addProperty("thumbnail", image2)
                    addProperty("original_image", image2)
                })
            })
            addProperty(
                "serpapi_google_hotels_photos_link",
                "https://serpapi.com/search.json?engine=google_hotels_photos&property_token=abc"
            )
        }
        val data = JsonObject().apply {
            addProperty("location", "Mahadevpura, Bengaluru")
            addProperty("query", "hotels in Mahadevpura Bengaluru")
            addProperty("check_in_date", "2026-06-20")
            addProperty("check_out_date", "2026-06-21")
            addProperty("adults", 2)
            add("properties", JsonArray().apply { add(hotel) })
        }

        val output = McpResponseFormatter.buildDataSection(
            McpClient.McpResult(
                domain = McpSettings.Domain.HOTELS,
                success = true,
                data = data,
                rawJson = null,
                error = null
            ),
            queryText = "hotels in Mahadevpura Bengaluru"
        )

        assertTrue(output.contains("| Hotel | Class | Rating | Reviews | Price / Night |"))
        assertTrue(output.contains("Photo URLs"))
        assertTrue(output.contains(image1))
        assertTrue(output.contains(image2))
        assertTrue(output.contains("Book / Website"))
        assertTrue(output.contains("https://www.google.com/maps/search/?api=1&query=12.9725,77.718"))
        assertTrue(output.contains("Google Hotels via SerpApi"))
        assertTrue(output.contains("## Quick Actions"))
        assertTrue(output.contains("Search Google Hotels"))
        assertTrue(!output.contains("Media: Image="))
    }

    @Test
    fun buildDataSection_preservesFlightLogoPriceAndActionFields() {
        val flight = JsonObject().apply {
            addProperty("airline_logo", "https://www.gstatic.com/flights/airline_logos/70px/6E.png")
            addProperty("price", 6667)
            addProperty("total_duration", 470)
            addProperty("type", "Round trip")
            addProperty("booking_token", "booking-token-1")
            add("carbon_emissions", JsonObject().apply {
                addProperty("difference_percent", 35)
            })
            add("layovers", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("id", "MAA")
                    addProperty("duration", 255)
                })
            })
            add("flights", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("airline", "IndiGo")
                    addProperty("flight_number", "6E 356")
                    addProperty("duration", 65)
                    addProperty("airplane", "Airbus A321neo")
                    addProperty("airline_logo", "https://www.gstatic.com/flights/airline_logos/70px/6E.png")
                    add("departure_airport", JsonObject().apply {
                        addProperty("id", "BLR")
                        addProperty("time", "2026-06-15 07:15")
                    })
                    add("arrival_airport", JsonObject().apply {
                        addProperty("id", "MAA")
                        addProperty("time", "2026-06-15 08:20")
                    })
                })
                add(JsonObject().apply {
                    addProperty("airline", "IndiGo")
                    addProperty("flight_number", "6E 515")
                    addProperty("duration", 150)
                    addProperty("airplane", "Airbus A320neo")
                    add("departure_airport", JsonObject().apply {
                        addProperty("id", "MAA")
                        addProperty("time", "2026-06-15 12:35")
                    })
                    add("arrival_airport", JsonObject().apply {
                        addProperty("id", "LKO")
                        addProperty("time", "2026-06-15 15:05")
                    })
                })
            })
        }
        val data = JsonObject().apply {
            addProperty("origin", "BLR")
            addProperty("destination", "LKO")
            addProperty("outbound_date", "2026-06-15")
            addProperty("currency", "INR")
            addProperty("travel_class", "Economy")
            addProperty("adults", 1)
            add("flights", JsonArray().apply { add(flight) })
            add("other_flights", JsonArray())
        }

        val output = McpResponseFormatter.buildDataSection(
            McpClient.McpResult(
                domain = McpSettings.Domain.FLIGHTS,
                success = true,
                data = data,
                rawJson = null,
                error = null
            ),
            queryText = "show flights from blr to lko on 15th june"
        )

        assertTrue(output.contains("| Airline | Departure | Arrival | Duration | Stops | Fare | Status | Legs | Layover | Carbon | Booking | Airline Logo | Booking URL | Action Label |"))
        assertTrue(output.contains("IndiGo - 6E 356 / 6E 515"))
        assertTrue(output.contains("BLR 07:15"))
        assertTrue(output.contains("LKO 15:05"))
        assertTrue(output.contains("BLR 07:15 -> MAA 08:20 (6E 356, Airbus A321neo); MAA 12:35 -> LKO 15:05 (6E 515, Airbus A320neo)"))
        assertTrue(output.contains("MAA 4h 15m"))
        assertTrue(output.contains("+35% CO2 vs typical"))
        assertTrue(output.contains("Booking token available"))
        assertTrue(output.contains("\u20B96,667 /adult"))
        assertTrue(output.contains("https://www.gstatic.com/flights/airline_logos/70px/6E.png"))
        assertTrue(output.contains("https://www.google.com/travel/flights"))
        assertTrue(output.contains("View fare"))
        assertTrue(output.contains("## Quick Actions"))
        assertTrue(output.contains("Search Google Flights"))
    }

    @Test
    fun buildDataSection_preservesNewsAsCompactTable() {
        val article = JsonObject().apply {
            addProperty("title", "Bengaluru civic update")
            addProperty("source_name", "Example News")
            addProperty("source_icon", "https://example.com/source.png")
            addProperty("source_url", "https://example.com")
            addProperty("pubDate", "2026-06-08 10:30:00")
            addProperty("description", "A short update about Bengaluru city services.")
            addProperty("link", "https://example.com/news/story")
            addProperty("image_url", "https://example.com/news/story.jpg")
            add("category", JsonArray().apply {
                add("top")
                add("local")
            })
        }
        val data = JsonObject().apply {
            addProperty("topic", "latest news")
            addProperty("location", "Bengaluru")
            addProperty("country", "in")
            addProperty("language", "en")
            add("results", JsonArray().apply { add(article) })
            addProperty("totalResults", 1)
        }

        val output = McpResponseFormatter.buildDataSection(
            McpClient.McpResult(
                domain = McpSettings.Domain.NEWS,
                success = true,
                data = data,
                rawJson = null,
                error = null
            ),
            queryText = "latest news in Bengaluru"
        )

        assertTrue(output.contains("News results table (domain: news, preferredPresentation: cards)."))
        assertTrue(output.contains("| Article | Source | Published | Category | Summary | Image URL | Source Icon | Article URL | Source URL | Action Label |"))
        assertTrue(output.contains("Bengaluru civic update"))
        assertTrue(output.contains("https://example.com/news/story.jpg"))
        assertTrue(output.contains("https://example.com/source.png"))
        assertTrue(output.contains("Read Article"))
        assertTrue(!output.contains("Media: Image="))
        assertTrue(!output.contains("## 1. Bengaluru civic update"))
    }

}

