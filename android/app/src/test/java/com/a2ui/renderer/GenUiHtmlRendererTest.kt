package com.samsung.genuicraft

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class GenUiHtmlRendererTest {
    @Test
    fun rendersStandardGenUiObject() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_1"}},
                {"updateComponents":{"surfaceId":"surface_1","components":[
                  {"id":"root","component":"Column","children":["title","button_1"]},
                  {"id":"title","component":"Text","text":"Hello Renderer","variant":"h2"},
                  {"id":"button_1","component":"Button","variant":"primary","child":"button_1_text","action":{"functionCall":{"call":"openUrl","args":{"url":"https://example.com"}}}},
                  {"id":"button_1_text","component":"Text","text":"Open Example"}
                ]}}
              ]
            }
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(input, sourceDir = File("C:/runs/sample"))
        assertTrue(result.html.contains("Hello Renderer"))
        assertTrue(result.html.contains("https://example.com"))
    }

    @Test
    fun readsFirstLineFromJsonl() {
        val jsonl = """
            {"genui_json":[{"createSurface":{"surfaceId":"s1"}},{"updateComponents":{"surfaceId":"s1","components":[{"id":"root","component":"Column","children":["t"]},{"id":"t","component":"Text","text":"From JSONL"}]}}]}
            {"genui_json":[{"createSurface":{"surfaceId":"s2"}}]}
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(jsonl)
        assertTrue(result.warnings.any { it.contains("JSONL") })
        assertTrue(result.html.contains("From JSONL"))
    }

    @Test
    fun rewritesLocalAssetPathsWhenSourceDirIsKnown() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_1"}},
                {"updateComponents":{"surfaceId":"surface_1","components":[
                  {"id":"root","component":"Column","children":["img"]},
                  {"id":"img","component":"Image","url":"/assets/icon.svg"}
                ]}}
              ]
            }
        """.trimIndent()

        val sourceDir = createTempDirectory(prefix = "genuicraft_html_assets_").toFile()
        try {
            val assetFile = File(sourceDir, "assets/icon.svg")
            assetFile.parentFile?.mkdirs()
            assetFile.writeText("<svg xmlns=\"http://www.w3.org/2000/svg\" />")

            val result = GenUiHtmlRenderer.render(input, sourceDir = sourceDir)
            assertTrue(result.html.contains("/assets/icon.svg"))
            assertTrue(result.html.contains("file:/"))
        } finally {
            sourceDir.deleteRecursively()
        }
    }

    @Test
    fun rendersBookingFallbackTextAsStructuredContent() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_booking"}},
                {"updateComponents":{"surfaceId":"surface_booking","components":[
                  {"id":"root","component":"Column","children":["text_1"]},
                  {"id":"text_1","component":"Text","variant":"body","text":"Flight Booking Recommendation: London to Tokyo\n\nFlight Comparison (Live Sources)\nAirline | Fare | Stops\nANA (Direct) | GBP 1,035 | Direct\nEmirates (via DXB) | EUR 673 | 1 stop\n\nBooking Cards\nOption 1: ANA (Direct) | Direct service with strong fare in May window\nAction: [Button: Book ANA] https://flights.ana.co.jp/en-gb/flights-from-london-to-tokyo\n\nOption 2: Emirates (via DXB) | Lowest fare snapshot, but one-stop routing\nAction: [Button: Book Emirates] https://www.skyscanner.net/flights/airline-flights-to-city/hnd/emirates-ek/cheap-flights-with-emirates-ek-flights-to-tokyo-haneda\n\nAirline Logos\n- ANA: ../assets/r_000264_01_logo_ana_official.svg\n\nIcons:\n- Airline: ../assets/r_000264_01_1_car-front.svg\n- Time: ../assets/r_000264_01_2_alarm.svg\n- Calendar: ../assets/r_000264_01_3_receipt.svg"}
                ]}}
              ]
            }
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(input)
        assertTrue(result.html.contains("class=\"data-table\""))
        assertTrue(result.html.contains("class=\"booking-grid\""))
        assertTrue(result.html.contains("class=\"booking-logo\""))
        assertTrue(result.html.contains("Book ANA"))
        assertTrue(result.html.contains("/assets/r_000264_01_logo_ana_official.svg"))
        assertFalse(result.html.contains("Airline Logos"))
        assertFalse(result.html.contains("/assets/r_000264_01_1_car-front.svg"))
        assertFalse(result.html.contains("/assets/r_000264_01_2_alarm.svg"))
        assertFalse(result.html.contains("/assets/r_000264_01_3_receipt.svg"))
    }

    @Test
    fun rendersRecipeAssetAndIconLinesAsMediaCards() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_recipe"}},
                {"updateComponents":{"surfaceId":"surface_recipe","components":[
                  {"id":"root","component":"Column","children":["text_1"]},
                  {"id":"text_1","component":"Text","variant":"body","text":"Creamy Vegan Mushroom Risotto Overview\n\nVegan Mushroom Risotto Plating: ../assets/r_000701_01_1_mushroom-risotto.jpg\nArborio Rice Texture: ../assets/r_000701_01_2_arborio-rice.jpg\n\nFire: ../assets/r_000701_01_4_fire.svg\nClock: ../assets/r_000701_01_5_clock.svg\n\nIcons:\n- Airplane: ../assets/r_000701_01_7_airplane.svg\n- Briefcase: ../assets/r_000701_01_8_briefcase.svg\n\nSources\nExample: https://example.com"}
                ]}}
              ]
            }
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(input)
        assertTrue(result.html.contains("class=\"media-grid\""))
        assertTrue(result.html.contains("class=\"media-card\""))
        assertTrue(result.html.contains("class=\"media-inline-icon\""))
        assertTrue(result.html.contains("/assets/r_000701_01_1_mushroom-risotto.jpg"))
        assertTrue(result.html.contains("/assets/r_000701_01_8_briefcase.svg"))
        assertFalse(result.html.contains("<p class=\"media-label\">Airplane</p>"))
        assertFalse(result.html.contains("<p class=\"media-label\">Briefcase</p>"))
        assertFalse(result.html.contains("<h4 class=\"rich-heading\">Icons:</h4>"))

        val mediaCardCount = Regex("class=\\\"media-card\\\"").findAll(result.html).count()
        assertEquals(2, mediaCardCount)
    }

    @Test
    fun rendersWeightedRowsAsComponentTable() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_table"}},
                {"updateComponents":{"surfaceId":"surface_table","components":[
                  {"id":"root","component":"Column","children":["row_h","row_1"]},
                  {"id":"row_h","component":"Row","children":["h1","h2","h3"]},
                  {"id":"h1","component":"Text","text":"Time","variant":"h4","weight":1},
                  {"id":"h2","component":"Text","text":"Event","variant":"h4","weight":2},
                  {"id":"h3","component":"Text","text":"Notes","variant":"h4","weight":3},
                  {"id":"row_1","component":"Row","children":["c1","c2","c3"]},
                  {"id":"c1","component":"Text","text":"6:00 PM","weight":1},
                  {"id":"c2","component":"Text","text":"Dinner","weight":2},
                  {"id":"c3","component":"Text","text":"Main hall","weight":3}
                ]}}
              ]
            }
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(input)
        assertTrue(result.html.contains("class=\"data-table component-table\""))
        assertTrue(result.html.contains("<th>Time</th>"))
        assertTrue(result.html.contains("<td>Dinner</td>"))
    }

    @Test
    fun rendersAllBundledSamplesWithoutErrorPage() {
        val sampleFile = File("src/main/assets/golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4_genui.jsonl")
        assertTrue(sampleFile.exists())
        val lines = sampleFile.readLines().filter { it.isNotBlank() }
        assertTrue(lines.size >= 50)

        lines.forEachIndexed { index, line ->
            val result = GenUiNativeRenderer.render(line, sourceDir = sampleFile.parentFile)
            assertTrue("Sample ${index + 1} returned error: ${result.errorMessage}", result.errorMessage == null)
            assertTrue("Sample ${index + 1} missing native surface output", result.surfaces.isNotEmpty())
        }
    }
}

