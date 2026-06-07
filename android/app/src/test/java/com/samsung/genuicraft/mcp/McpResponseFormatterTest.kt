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
        assertTrue(!output.contains("Media: Image="))
    }

}

