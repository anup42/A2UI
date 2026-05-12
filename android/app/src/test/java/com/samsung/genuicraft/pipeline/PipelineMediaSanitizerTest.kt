package com.samsung.genuicraft.pipeline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineMediaSanitizerTest {
    @Test
    fun ensureFlightListContent_injectsComparisonTable_whenMissing() {
        val response = """
            Flights from BLR to LKO on March 15 include options from IndiGo and Air India.
            IndiGo 06:15 AM to 08:50 AM, 2h 35m, Non-stop, INR 6,212.
            Air India 10:15 AM to 12:50 PM, 2h 35m, Non-stop, INR 6,266.
        """.trimIndent()

        val transformed = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = response,
            queryText = "best flights from BLR to LKO on march 15 with fares and stops"
        )

        assertTrue(transformed.contains("Airline | Departure | Arrival | Duration | Stops | Fare"))
        assertTrue(transformed.contains("IndiGo"))
        assertTrue(transformed.contains("Air India"))
        assertTrue(PipelineMediaSanitizer.responseContainsFlightList(transformed))
    }

    @Test
    fun ensureFlightListContent_preservesExistingFlightTable() {
        val response = """
            Flight Comparison
            Airline | Departure | Arrival | Duration | Stops | Fare
            IndiGo | 06:15 AM | 08:50 AM | 2h 35m | Non-stop | INR 6,212
            Air India | 10:15 AM | 12:50 PM | 2h 35m | Non-stop | INR 6,266
        """.trimIndent()

        val transformed = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = response,
            queryText = "best flights from BLR to LKO on march 15 with fares and stops"
        )

        assertEquals(response, transformed)
    }

    @Test
    fun hasInlineImageUrl_ignoresBootstrapIconImageAssignments() {
        val response = "Media: Image=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg"

        assertEquals(false, PipelineMediaSanitizer.hasInlineImageUrl(response))
        assertTrue(PipelineMediaSanitizer.hasInlineIconUrl("Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg"))
    }

    @Test
    fun hasInlineImageUrl_ignoresRandomPlaceholderHosts() {
        val response = "Media: Image=https://loremflickr.com/1200/800/travel,bengaluru"

        assertEquals(false, PipelineMediaSanitizer.hasInlineImageUrl(response))
    }

    @Test
    fun ensureTravelInlineMedia_addsIconOnlyFallbackWithoutHardcodedImages() {
        val response = """
            # 4-Day City Vacation

            Day 1: Central district and park
            - Morning: Walk around the main park.
        """.trimIndent()

        val transformed = PipelineMediaSanitizer.ensureTravelInlineMedia(
            responseText = response,
            queryText = "show 4 day itinerary for vacation in a city"
        )

        assertEquals(false, transformed.contains("Image="))
        assertTrue(transformed.contains("Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/"))
    }

    @Test
    fun ensureGenUiHasImageComponent_injectsMissingFlatSpecImage() {
        val json = """
            {
              "root": "rootStack",
              "state": {},
              "elements": {
                "rootStack": { "type": "Stack", "props": {}, "children": ["title"] },
                "title": { "type": "Text", "props": { "text": "Trip Plan" }, "children": [] }
              }
            }
        """.trimIndent()
        val response = """
            Day 1: City center
            Media: Image=https://commons.wikimedia.org/wiki/Special:FilePath/Vidhana_Soudha.jpg Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg
        """.trimIndent()

        val transformed = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
            jsonText = json,
            stage2Response = response,
            queryText = "show 4 day itinerary for vacation"
        )

        assertTrue(transformed.contains("\"type\":\"Image\""))
        assertTrue(transformed.contains("Vidhana_Soudha.jpg"))
        assertTrue(transformed.contains("\"auto_media_"))
    }

    @Test
    fun normalizeUrlTokensForDisplay_removesUnsafeUrls() {
        val response = "Open http://127.0.0.1/admin or https://www.google.com/search?q=bengaluru"

        val transformed = PipelineMediaSanitizer.normalizeUrlTokensForDisplay(response)

        assertEquals(false, transformed.contains("127.0.0.1"))
        assertTrue(transformed.contains("https://www.google.com/search?q=bengaluru"))
    }

    @Test
    fun enforceSafeGenUiContent_removesUnsafeMediaAndOpenUrlActions() {
        val json = """
            {
              "root": "rootStack",
              "state": {
                "rows": [
                  {
                    "name": "Unsafe row",
                    "image": "https://loremflickr.com/1200/800/travel",
                    "bookingUrl": "javascript:alert(1)"
                  }
                ]
              },
              "elements": {
                "rootStack": { "type": "Stack", "props": {}, "children": ["image", "table", "button"] },
                "image": { "type": "Image", "props": { "url": "https://picsum.photos/800/600" }, "children": [] },
                "table": {
                  "type": "Table",
                  "props": {
                    "columns": [{"key":"name","label":"Name"},{"key":"image","label":"Image"}],
                    "statePath": "/rows"
                  },
                  "children": []
                },
                "button": {
                  "type": "Button",
                  "props": { "label": "Open" },
                  "on": { "press": { "action": "openUrl", "params": { "url": "file:///sdcard/secret" } } },
                  "children": []
                }
              }
            }
        """.trimIndent()

        val result = PipelineMediaSanitizer.enforceSafeGenUiContent(json)

        assertTrue(result.changed)
        assertEquals(false, result.jsonText.contains("loremflickr"))
        assertEquals(false, result.jsonText.contains("picsum"))
        assertEquals(false, result.jsonText.contains("file:///sdcard"))
        assertEquals(false, result.jsonText.contains("javascript:alert"))
    }
}
