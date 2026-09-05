package com.samsung.genuicraft.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.mcp.McpClient
import com.samsung.genuicraft.mcp.McpResponseFormatter
import com.samsung.genuicraft.mcp.McpSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResponseFactCoveragePresentationHintTest {
    @Test
    fun missingFacts_ignoresWeatherFormatterPresentationHint() {
        assertPresentationHintIsNotVisibleFact(
            "Weather forecast table (domain: weather, preferredPresentation: cards)."
        )
    }

    @Test
    fun missingFacts_ignoresVacationFormatterPresentationHint() {
        assertPresentationHintIsNotVisibleFact(
            "Vacation itinerary table (domain: schedule, preferredPresentation: cards)."
        )
    }

    @Test
    fun missingFacts_ignoresNewsFormatterPresentationHint() {
        assertPresentationHintIsNotVisibleFact(
            "News results table (domain: news, preferredPresentation: cards)."
        )
    }

    @Test
    fun missingFacts_recognizesWholePresentationHintWithWhitespaceAndSupportedModes() {
        listOf("cards", "table", "auto").forEach { presentation ->
            assertPresentationHintIsNotVisibleFact(
                "  Weather forecast table (domain: weather, preferredPresentation: $presentation)  "
            )
        }
    }

    @Test
    fun missingFacts_acceptsFormatterWeatherRowsInDecodedExpressWithoutDisplayingMetadata() {
        val response = weatherResponse()
        val graph = decodeWeatherTable(response)
        val props = graph.getAsJsonObject("elements").getAsJsonObject("forecast")
            .getAsJsonObject("props")

        assertTrue(response.contains("Weather forecast table (domain: weather, preferredPresentation: cards)."))
        assertEquals("weather", props.get("domain").asString)
        assertEquals("cards", props.get("preferredPresentation").asString)
        assertEquals(19, props.getAsJsonArray("columns").size())
        assertEquals(2, props.getAsJsonArray("rows").size())
        assertEquals("Today", props.getAsJsonArray("rows")[0].asJsonArray[0].asString)
        assertEquals("37 C", props.getAsJsonArray("rows")[0].asJsonArray[4].asString)
        assertNull(ResponseFactCoverage.failureReason(response, graph))
    }

    @Test
    fun missingFacts_stillRejectsOmittedWeatherDay() {
        val response = weatherResponse()
        val graph = decodeWeatherTable(response)
        graph.getAsJsonObject("elements").getAsJsonObject("forecast")
            .getAsJsonObject("props").getAsJsonArray("rows").remove(1)

        val missing = ResponseFactCoverage.missingFacts(response, graph)

        assertTrue(missing.toString(), missing.any { it.value.contains("2026-09-04") })
    }

    @Test
    fun missingFacts_stillRejectsChangedWeatherTemperature() {
        val response = weatherResponse()
        val graph = decodeWeatherTable(response)
        graph.getAsJsonObject("elements").getAsJsonObject("forecast")
            .getAsJsonObject("props").getAsJsonArray("rows")[0].asJsonArray
            .set(4, JsonPrimitive("30 C"))

        val missing = ResponseFactCoverage.missingFacts(response, graph)

        assertTrue(missing.toString(), missing.any { it.label == "Today" && it.value.contains("37 C") })
    }

    @Test
    fun missingFacts_stillRejectsOmittedWeatherLocation() {
        val response = weatherResponse()
        val graph = decodeWeatherTable(response)
        graph.getAsJsonObject("elements").getAsJsonObject("heading")
            .getAsJsonObject("props").addProperty("text", "Weather forecast")

        val missing = ResponseFactCoverage.missingFacts(response, graph)

        assertTrue(missing.toString(), missing.any { it.value == "Weather in Bengaluru, India" })
    }

    @Test
    fun missingFacts_keepsOrdinaryColonFactsAndIncompleteMetadataLikeProse() {
        val ordinaryFacts = listOf(
            "Forecast preference: weather, preferredPresentation: cards).",
            "Weather forecast table (domain: weather).",
            "Weather forecast table (preferredPresentation: cards).",
            "Current weather (domain: weather, preferredPresentation: cards).",
            "Weather forecast table (domain: weather, preferredPresentation: tiles).",
            "Weather forecast table (domain: weather, preferredPresentation: cards, temperature: 37 C).",
            "Weather forecast table (domain: weather, preferredPresentation: cards). Rain expected."
        )

        ordinaryFacts.forEach { response ->
            val missing = ResponseFactCoverage.missingFacts(response, textGraph("Ready"))
            assertTrue("Must preserve ordinary source content: $response", missing.isNotEmpty())
        }
    }

    @Test
    fun missingFacts_ignoresOnlyHintLineAndPreservesAdjacentRealFacts() {
        val response = """
            City: Bengaluru
            Weather forecast table (domain: weather, preferredPresentation: cards).
            Temperature: 37 C
        """.trimIndent()

        val missing = ResponseFactCoverage.missingFacts(response, textGraph("Bengaluru"))

        assertEquals(listOf(ResponseFactCoverage.Fact("Temperature", "37 C")), missing)
    }

    @Test
    fun missingFacts_preservesReferenceIdentifierInsidePresentationHintCaption() {
        val response = "Booking AB123 table (domain: booking, preferredPresentation: cards)."

        assertEquals(
            listOf(ResponseFactCoverage.Fact("Booking", "AB123")),
            ResponseFactCoverage.missingFacts(response, textGraph("Booking confirmed"))
        )
        assertNull(ResponseFactCoverage.failureReason(response, textGraph("Booking AB123")))
    }

    private fun assertPresentationHintIsNotVisibleFact(hint: String) {
        val response = "Forecast: Light rain\r\n$hint\r\n"
        assertNull(ResponseFactCoverage.failureReason(response, textGraph("Light rain")))
    }

    private fun textGraph(text: String): JsonObject = GenUiIrCodec.decode(
        JsonPrimitive("<a2ui>\nroot=Text(${JsonPrimitive(text)},\"body\")\n</a2ui>")
    ).canonicalGraph

    private fun weatherResponse(): String {
        val data = JsonParser.parseString(
            """
            {
              "location": "Bengaluru",
              "country": "India",
              "requested_forecast_days": 2,
              "current": {"temperature_2m": "26", "weather_code": 3},
              "current_units": {"temperature_2m": "C"},
              "daily": {
                "time": ["2026-09-03", "2026-09-04"],
                "temperature_2m_max": ["37", "29"],
                "temperature_2m_min": ["21", "20"],
                "weather_code": [3, 61]
              },
              "daily_units": {"temperature_2m_max": "C", "temperature_2m_min": "C"}
            }
            """.trimIndent()
        ).asJsonObject
        return McpResponseFormatter.buildDataSection(
            McpClient.McpResult(
                domain = McpSettings.Domain.WEATHER,
                success = true,
                data = data,
                rawJson = null,
                error = null
            ),
            queryText = "show weather in Bengaluru"
        )
    }

    private fun decodeWeatherTable(response: String): JsonObject {
        // Exercise the real formatter's full 19-column table through the production Express decoder.
        val tableLines = response.lineSequence().filter { it.startsWith('|') }.toList()
        fun cells(line: String): JsonArray = JsonArray().apply {
            line.split('|').drop(1).dropLast(1).forEach { add(it.trim()) }
        }
        val columns = cells(tableLines.first())
        val rows = JsonArray().apply { tableLines.drop(2).forEach { add(cells(it)) } }
        val decoded = GenUiIrCodec.decode(
            JsonPrimitive(
                """
                <a2ui>
                root=Column([heading,forecast,sources,actions],gap="md")
                heading=Text("Weather in Bengaluru, India","h2")
                forecast=Table($columns,rows=$rows,domain="weather",preferredPresentation="cards")
                sources=Text("Sources","h3")
                actions=Text("Quick Actions","h3")
                </a2ui>
                """.trimIndent()
            )
        )
        assertEquals(GenUiIrFormat.A2UI_EXPRESS_V1, decoded.sourceFormat)
        return decoded.canonicalGraph
    }
}
