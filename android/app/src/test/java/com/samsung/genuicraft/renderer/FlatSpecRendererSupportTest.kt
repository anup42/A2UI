package com.samsung.genuicraft.renderer

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatSpecRendererSupportTest {

    @Test
    fun resolveCoilMediaModel_mapsAssetPathsToAndroidAssetUris() {
        assertEquals(
            "file:///android_asset/icons/cloud.svg",
            resolveCoilMediaModel("/assets/icons/cloud.svg")
        )
        assertEquals(
            "file:///android_asset/icons/cloud.svg",
            resolveCoilMediaModel("assets/icons/cloud.svg")
        )
        assertEquals(
            "file:///android_asset/icons/cloud.svg",
            resolveCoilMediaModel("../assets/icons/cloud.svg")
        )
        assertEquals(
            "file:///android_asset/icons/cloud.svg",
            resolveCoilMediaModel("./assets/icons/cloud.svg")
        )
    }

    @Test
    fun resolveMediaUrlCandidate_supportsLegacyAndSourceStringFields() {
        val imageKeys = listOf("url", "src", "image", "source", "name")
        val legacyProps = mapOf<String, Any?>("url" to "https://example.com/a.jpg")
        val sourceProps = mapOf<String, Any?>("source" to "https://example.com/b.jpg")

        assertEquals("https://example.com/a.jpg", resolveMediaUrlCandidate(legacyProps, imageKeys))
        assertEquals("https://example.com/b.jpg", resolveMediaUrlCandidate(sourceProps, imageKeys))
    }

    @Test
    fun resolveMediaUrlCandidate_supportsSourceObjectAndNestedListForms() {
        val imageKeys = listOf("url", "src", "image", "source", "name")
        val sourceUriProps = mapOf<String, Any?>(
            "source" to mapOf("uri" to "https://example.com/c.jpg")
        )
        val listProps = mapOf<String, Any?>(
            "source" to listOf(
                mapOf("path" to ""),
                mapOf("value" to "https://example.com/d.jpg")
            )
        )

        assertEquals("https://example.com/c.jpg", resolveMediaUrlCandidate(sourceUriProps, imageKeys))
        assertEquals("https://example.com/d.jpg", resolveMediaUrlCandidate(listProps, imageKeys))
    }

    @Test
    fun resolveMediaUrlCandidate_supportsIconSourceField() {
        val iconKeys = listOf("name", "icon", "source", "url", "src")
        val props = mapOf<String, Any?>(
            "source" to mapOf("url" to "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/cloud.svg")
        )

        assertEquals(
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/cloud.svg",
            resolveMediaUrlCandidate(props, iconKeys)
        )
    }

    @Test
    fun resolveMediaUrlCandidate_ignoresUnresolvedBindingMaps() {
        val imageKeys = listOf("url", "src", "image", "source", "name")
        val unresolvedBindingProps = mapOf<String, Any?>(
            "source" to mapOf("\$item" to "logo")
        )

        assertEquals("", resolveMediaUrlCandidate(unresolvedBindingProps, imageKeys))
    }

    @Test
    fun asFlatSpacingDp_supportsSymbolicPaddingTokens() {
        assertEquals(16f, asFlatSpacingDp("md")!!.value, 0.01f)
        assertEquals(24f, asFlatSpacingDp("xl")!!.value, 0.01f)
        assertEquals(7f, asFlatSpacingDp(7)!!.value, 0.01f)
    }

    @Test
    fun rowChildFlex_readsPositiveFlexOnly() {
        assertEquals(
            1f,
            rowChildFlex(
                FlatElement(
                    type = "Stack",
                    props = mapOf("flex" to 1),
                    children = emptyList()
                )
            ),
            0.01f
        )
        assertEquals(
            0f,
            rowChildFlex(
                FlatElement(
                    type = "Stack",
                    props = mapOf("flex" to 0),
                    children = emptyList()
                )
            ),
            0.01f
        )
        assertEquals(0f, rowChildFlex(null), 0.01f)
    }

    @Test
    fun compactBulletItems_splitsTaskLikeSemicolonLists() {
        val items = compactBulletItems(
            "Key Tasks",
            "Register business name; Order packaging; Test checkout."
        )

        assertEquals(
            listOf("Register business name", "Order packaging", "Test checkout"),
            items
        )
        assertTrue(compactBulletItems("Goal", "Register business name; Order packaging").isEmpty())
        assertTrue(compactBulletItems("Key Tasks", "Register business name").isEmpty())
    }

    @Test
    fun extractDirectTableModel_keepsComparisonEntityMediaCompact() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "feature", "label" to "Feature"),
                mapOf("key" to "drip", "label" to "Drip Coffee Machine"),
                mapOf("key" to "french_press", "label" to "French Press")
            ),
            "rows" to listOf(
                mapOf(
                    "feature" to "Avg. Brew Time",
                    "drip" to "4-8 minutes",
                    "french_press" to "4-5 minutes"
                )
            ),
            "domain" to "comparison",
            "preferredPresentation" to "table",
            "entityMedia" to mapOf(
                "drip" to mapOf(
                    "image" to "../assets/drip.svg",
                    "alt" to "Drip coffee maker"
                ),
                "french_press" to "../assets/french_press.svg"
            )
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals("../assets/drip.svg", model!!.entityMedia["drip"]?.image)
        assertEquals("Drip coffee maker", model.entityMedia["drip"]?.alt)
        assertEquals("../assets/french_press.svg", model.entityMedia["frenchpress"]?.image)
    }

    @Test
    fun extractDirectTableModel_routesFlightDomainToFlightCards() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "carrier", "label" to "Carrier Combination"),
                mapOf("key" to "leg1", "label" to "Leg 1: JFK-LHR"),
                mapOf("key" to "leg2", "label" to "Leg 2: LHR-FCO"),
                mapOf("key" to "leg3", "label" to "Leg 3: FCO-JFK"),
                mapOf("key" to "feature", "label" to "Key Feature")
            ),
            "rows" to listOf(
                mapOf(
                    "carrier" to "United / Lufthansa",
                    "leg1" to "United (Direct)",
                    "leg2" to "Lufthansa via FRA",
                    "leg3" to "United (Direct)",
                    "feature" to "Most direct long-haul routing."
                )
            ),
            "domain" to "flight",
            "preferredPresentation" to "cards"
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = false
        )

        assertNotNull(model)
        assertEquals(FlatTableRenderMode.FLIGHT_CARDS, model!!.renderMode)
    }

    @Test
    fun extractDirectTableModel_infersPlaylistFromSongArtistPhaseColumns() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "phase", "label" to "Phase"),
                mapOf("key" to "artist", "label" to "Artist"),
                mapOf("key" to "songTitle", "label" to "Song Title")
            ),
            "rows" to listOf(
                mapOf(
                    "phase" to "1: Guest Arrival (0-60 min)",
                    "artist" to "Vitamin String Quartet",
                    "songTitle" to "thank u, next"
                )
            ),
            "domain" to "generic",
            "preferredPresentation" to "table"
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals("playlist", model!!.domain)
        assertEquals("cards", model.preferredPresentation)
        assertEquals(FlatTableShape.PLAYLIST, model.shape)
        assertEquals(FlatTableRenderMode.PLAYLIST_CARDS, model.renderMode)
    }

    @Test
    fun extractDirectTableModel_routesComparisonCardsWhenRequested() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "city", "label" to "City"),
                mapOf("key" to "avg_low", "label" to "Average Low"),
                mapOf("key" to "sunshine", "label" to "Sunshine")
            ),
            "rows" to listOf(
                mapOf("city" to "Rome", "avg_low" to "12Â°C", "sunshine" to "6h"),
                mapOf("city" to "Lisbon", "avg_low" to "15Â°C", "sunshine" to "7h")
            ),
            "domain" to "comparison",
            "preferredPresentation" to "cards"
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, model!!.renderMode)
        assertEquals(FlatTableShape.ENTITY_ROW, model.shape)
    }

    @Test
    fun extractDirectTableModel_routesUiStateTableToProcessCards() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "ui_state", "label" to "UI State"),
                mapOf("key" to "visuals", "label" to "Key Visuals"),
                mapOf("key" to "feedback", "label" to "On-Screen Text and Feedback")
            ),
            "rows" to listOf(
                mapOf(
                    "ui_state" to "Ready to Scan",
                    "visuals" to "Camera preview with QR scan frame.",
                    "feedback" to "Scan your ticket QR code."
                ),
                mapOf(
                    "ui_state" to "Check-in Success",
                    "visuals" to "Green confirmation overlay.",
                    "feedback" to "Check-in successful."
                ),
                mapOf(
                    "ui_state" to "Invalid Code",
                    "visuals" to "Error overlay on the scanner.",
                    "feedback" to "Invalid QR code."
                )
            ),
            "domain" to "status",
            "preferredPresentation" to "table"
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals(FlatTableRenderMode.PROCESS_CARDS, model!!.renderMode)
    }

    @Test
    fun looksLikeIncidentStatusTable_detectsComponentStatusNotesRows() {
        val headers = listOf("Component", "Current Status", "Notes")
        val rows = listOf(
            listOf("Database", "MAJOR_OUTAGE", "Widespread data access failures."),
            listOf("API", "DEGRADED_PERFORMANCE", "System responses may be slow."),
            listOf("Payment Gateway", "DEGRADED_PERFORMANCE", "Transactions may be delayed.")
        )

        assertTrue(looksLikeIncidentStatusTable(headers, rows, domain = "status"))
        assertFalse(
            looksLikeIncidentStatusTable(
                headers = listOf("Component", "Version", "Owner"),
                rows = listOf(listOf("API", "v1", "Platform")),
                domain = "status"
            )
        )
    }

    @Test
    fun isDetachedMediaDumpElement_detectsTrailingImagesCardOnly() {
        val elements = mapOf(
            "images_card" to FlatElement(
                type = "Card",
                props = emptyMap(),
                children = listOf("images_title", "image_one")
            ),
            "images_title" to FlatElement(
                type = "Text",
                props = mapOf("text" to "Images"),
                children = emptyList()
            ),
            "image_one" to FlatElement(
                type = "Image",
                props = mapOf("url" to "../assets/sample.jpg"),
                children = emptyList()
            ),
            "content_card" to FlatElement(
                type = "Card",
                props = emptyMap(),
                children = listOf("content_title", "content_body")
            ),
            "content_title" to FlatElement(
                type = "Text",
                props = mapOf("text" to "Check-in Success"),
                children = emptyList()
            ),
            "content_body" to FlatElement(
                type = "Text",
                props = mapOf("text" to "Show a green confirmation overlay."),
                children = emptyList()
            )
        )

        assertTrue(isDetachedMediaDumpElement("images_card", elements))
        assertFalse(isDetachedMediaDumpElement("content_card", elements))
    }

    @Test
    fun extractFlatTableModel_ignoresSectionHeaderRowsWithIcons() {
        val elements = mapOf(
            "sectionHeader" to FlatElement(
                type = "Stack",
                props = mapOf("direction" to "horizontal"),
                children = listOf("headerIcon", "headerText")
            ),
            "headerIcon" to FlatElement(
                type = "Icon",
                props = mapOf("name" to "../assets/list.svg"),
                children = emptyList()
            ),
            "headerText" to FlatElement(
                type = "Text",
                props = mapOf("text" to "Baking Steps"),
                children = emptyList()
            ),
            "list" to FlatElement(
                type = "Stack",
                props = mapOf("direction" to "vertical"),
                children = listOf("row"),
                repeat = RepeatConfig(statePath = "/steps", key = "id")
            ),
            "row" to FlatElement(
                type = "Stack",
                props = mapOf("direction" to "horizontal"),
                children = listOf("number", "body")
            ),
            "number" to FlatElement(
                type = "Text",
                props = mapOf("text" to mapOf("\$template" to "${'$'}{index_1}.")),
                children = emptyList()
            ),
            "body" to FlatElement(
                type = "Text",
                props = mapOf("text" to mapOf("\$item" to "description")),
                children = emptyList()
            )
        )

        val model = extractFlatTableModel(
            containerChildren = listOf("sectionHeader", "list"),
            containerProps = mapOf("direction" to "vertical"),
            elements = elements,
            state = mapOf("steps" to listOf(mapOf("id" to "1", "description" to "Preheat."))),
            compactScreen = true
        )

        assertNull(model)
    }

    @Test
    fun looksLikeClimateComparisonTable_detectsCityWeatherMetrics() {
        val headers = listOf(
            "City",
            "Verdict",
            "Average High",
            "Average Low",
            "Rainy Days",
            "Sunshine/Day"
        )
        val rows = listOf(
            listOf("Rome, Italy", "Mild", "22 C", "12 C", "8", "6 hours/day"),
            listOf("Lisbon, Portugal", "Recommended", "22 C", "15 C", "9", "7 hours/day")
        )

        assertTrue(looksLikeClimateComparisonTable(headers, rows, domain = "comparison"))
        assertFalse(
            looksLikeClimateComparisonTable(
                headers = listOf("Product", "Price", "Rating"),
                rows = listOf(listOf("A", "$10", "4.5")),
                domain = "comparison"
            )
        )
    }

    @Test
    fun extractChartPoints_parsesCurrencyBarChartRows() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "month", "label" to "Month"),
                mapOf("key" to "revenue", "label" to "Revenue")
            ),
            "rows" to listOf(
                mapOf("month" to "January", "revenue" to "$12,500"),
                mapOf("month" to "February", "revenue" to "$18,000")
            ),
            "xKey" to "month",
            "yKey" to "revenue"
        )

        val points = extractChartPoints(props, emptyMap())

        assertEquals(2, points.size)
        assertEquals("January", points[0].label)
        assertEquals(12500.0, points[0].value, 0.01)
        assertEquals("$18,000", points[1].displayValue)
    }

    @Test
    fun extractPercentageMatrixChartModel_detectsSurveyDistributionTable() {
        val columns = listOf(
            FlatDirectTableColumn("age_group", "Age Group"),
            FlatDirectTableColumn("tiktok", "TikTok"),
            FlatDirectTableColumn("instagram", "Instagram"),
            FlatDirectTableColumn("facebook", "Facebook"),
            FlatDirectTableColumn("linkedin", "LinkedIn")
        )
        val rows = listOf(
            listOf("18-24", "75.0%", "25.0%", "0.0%", "0.0%"),
            listOf("25-34", "0.0%", "66.7%", "0.0%", "33.3%"),
            listOf("35-44", "0.0%", "0.0%", "66.7%", "33.3%")
        )

        val model = extractPercentageMatrixChartModel(columns, rows, domain = "comparison")

        assertNotNull(model)
        assertEquals("Age Group", model!!.categoryLabel)
        assertEquals(3, model.rows.size)
        assertEquals("TikTok", model.rows.first().segments.first().label)
        assertEquals(75.0, model.rows.first().segments.first().value, 0.01)
    }

    @Test
    fun bookingRowActionLabel_readsSupportedActionLabelColumns() {
        val headers = listOf("Hotel", "Price", "Booking URL", "Action Label")
        val row = listOf(
            "Sakura Inn Kyoto",
            "$220",
            "https://www.booking.com/searchresults.html?ss=Sakura+Inn+Kyoto",
            "Check Availability"
        )

        assertEquals("Check Availability", bookingRowActionLabel(headers, row))
        assertEquals(
            "Reserve",
            bookingRowActionLabel(
                listOf("Hotel", "CTA Label"),
                listOf("Kyoto Grand Hotel", "Reserve")
            )
        )
        assertNull(
            bookingRowActionLabel(
                listOf("Hotel", "Button Label"),
                listOf("Kyoto Grand Hotel", "https://example.com/not-a-label")
            )
        )
    }

    @Test
    fun bookingRowImageUrl_readsSupportedImageColumns() {
        assertEquals(
            "../assets/sakura.jpg",
            bookingRowImageUrl(
                listOf("Hotel", "Image URL", "Booking URL"),
                listOf("Sakura Inn Kyoto", "../assets/sakura.jpg", "https://example.com")
            )
        )
        assertEquals(
            "../assets/gion.jpg",
            bookingRowImageUrl(
                listOf("Hotel", "Photo", "Price"),
                listOf("Gion Traditional Ryokan", "../assets/gion.jpg", "$245")
            )
        )
        assertNull(
            bookingRowImageUrl(
                listOf("Hotel", "Price"),
                listOf("Kyoto Grand Hotel", "$280")
            )
        )
    }

    @Test
    fun deriveImageFallbackUrl_returnsSeededFallbackForWikimediaWeatherLikeImages() {
        val sourceUrl =
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Bangalore_skyline.jpg/1280px-Bangalore_skyline.jpg"
        val props = mapOf<String, Any?>(
            "alt" to "Bengaluru weather skyline"
        )

        val fallback = deriveImageFallbackUrl(sourceUrl, props)

        assertEquals("https://picsum.photos/seed/genuicraft_weather_hero/1280/720", fallback)
    }

    @Test
    fun deriveImageFallbackUrl_prefersExplicitFallbackUrl() {
        val sourceUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/x/y/z.jpg/1200px-z.jpg"
        val props = mapOf<String, Any?>(
            "fallbackUrl" to "https://example.com/fallback.jpg"
        )

        val fallback = deriveImageFallbackUrl(sourceUrl, props)

        assertEquals("https://example.com/fallback.jpg", fallback)
    }

    @Test
    fun deriveImageFallbackUrl_returnsNullForNonWikimediaHosts() {
        val sourceUrl = "https://cdn.example.com/media/photo.jpg"

        val fallback = deriveImageFallbackUrl(sourceUrl, emptyMap())

        assertNull(fallback)
    }

    @Test
    fun parse_supportsTemplateAndRepeatInsideProps() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "items": [ { "id": "a", "title": "One" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["list"] },
                "list": {
                  "type": "List",
                  "props": {
                    "template": "row",
                    "repeat": { "path": "items", "key": "id" }
                  },
                  "children": []
                },
                "row": { "type": "Card", "props": {}, "children": ["title"] },
                "title": { "type": "Text", "text": "Top picks", "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)
        assertNotNull(parsed)
        val list = parsed!!.elements["list"]!!
        assertTrue(list.children.contains("row"))
        assertEquals("/items", list.repeat?.statePath)
        val title = parsed.elements["title"]!!
        assertEquals("Top picks", title.props["text"])
    }

    @Test
    fun parse_normalizesLegacyRowAndColumnTypesToStack() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "root",
              "state": {},
              "elements": {
                "root": { "type": "Column", "props": {}, "children": ["row1"] },
                "row1": { "type": "Row", "props": {}, "children": ["t1"] },
                "t1": { "type": "Text", "props": { "text": "ok" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)
        assertNotNull(parsed)
        val root = parsed!!.elements["root"]!!
        val row = parsed.elements["row1"]!!
        assertEquals("Stack", root.type)
        assertEquals("vertical", root.props["direction"])
        assertEquals("Stack", row.type)
        assertEquals("horizontal", row.props["direction"])
    }

    @Test
    fun setAtPath_updatesNestedMapPath() {
        val state = mutableMapOf<String, Any?>(
            "filters" to mapOf(
                "price" to mapOf(
                    "min" to 100
                )
            )
        )

        FlatSpecParser.setAtPath(state, "/filters/price/max", 500)

        assertEquals(100, FlatSpecParser.getAtPath(state, "/filters/price/min"))
        assertEquals(500, FlatSpecParser.getAtPath(state, "/filters/price/max"))
    }

    @Test
    fun setAtPath_updatesNestedListPath() {
        val state = mutableMapOf<String, Any?>()

        FlatSpecParser.setAtPath(state, "/items/0/name", "A")
        FlatSpecParser.setAtPath(state, "/items/1/name", "B")

        assertEquals("A", FlatSpecParser.getAtPath(state, "/items/0/name"))
        assertEquals("B", FlatSpecParser.getAtPath(state, "/items/1/name"))
    }

    @Test
    fun evaluateVisible_andStateExpressions_workForFlatSpec() {
        val state = mapOf<String, Any?>("tab" to "flights")
        val visibleExpr = mapOf(
            "\$state" to "/tab",
            "eq" to "flights"
        )
        val hiddenExpr = mapOf(
            "\$state" to "/tab",
            "eq" to "hotels"
        )

        assertTrue(FlatExprResolver.evaluateVisible(visibleExpr, state, null))
        assertTrue(!FlatExprResolver.evaluateVisible(hiddenExpr, state, null))
    }

    @Test
    fun resolve_itemAndTemplateExpressions() {
        val state = mapOf<String, Any?>(
            "user" to mapOf("name" to "Anup")
        )
        val item = mapOf<String, Any?>("url" to "https://example.com", "title" to "Prepare Dough")

        val itemExpr = mapOf("\$item" to "url")
        val templateExpr = mapOf("\$template" to "Hello ${'$'}{/user/name}")
        val mustacheItemTemplateExpr = mapOf("\$template" to "Step 1: {{${'$'}item.title}}")
        val legacyItemTemplateExpr = mapOf("\$template" to "Step 1: {${'$'}item.title}")
        val dollarItemTemplateExpr = mapOf("\$template" to "Step 1: ${'$'}{${'$'}item.title}")
        val compactItemTemplateExpr = mapOf("\$template" to "Price: ${'$'}{price}")

        assertEquals("https://example.com", FlatExprResolver.resolve(itemExpr, state, item))
        assertEquals("Hello Anup", FlatExprResolver.resolve(templateExpr, state, item))
        assertEquals("Step 1: Prepare Dough", FlatExprResolver.resolve(mustacheItemTemplateExpr, state, item))
        assertEquals("Step 1: Prepare Dough", FlatExprResolver.resolve(legacyItemTemplateExpr, state, item))
        assertEquals("Step 1: Prepare Dough", FlatExprResolver.resolve(dollarItemTemplateExpr, state, item))
        assertEquals(
            "Price: GBP 199",
            FlatExprResolver.resolve(compactItemTemplateExpr, state, item + ("price" to "GBP 199"))
        )
    }

    @Test
    fun resolve_supportsNestedItemIndexAndBindItem() {
        val state = mapOf<String, Any?>(
            "items" to listOf(
                mapOf("meta" to mapOf("title" to "Alpha"), "name" to "First")
            )
        )
        val scope = RepeatScope(
            item = mapOf("meta" to mapOf("title" to "Alpha"), "name" to "First"),
            index = 0,
            basePath = "/items/0"
        )

        val nestedItemExpr = mapOf("\$item" to "meta/title")
        val indexExpr = mapOf("\$index" to true)
        val bindItemExpr = mapOf("\$bindItem" to "name")

        assertEquals("Alpha", FlatExprResolver.resolve(nestedItemExpr, state, scope, emptyMap()))
        assertEquals(0, FlatExprResolver.resolve(indexExpr, state, scope, emptyMap()))
        assertEquals("First", FlatExprResolver.resolve(bindItemExpr, state, scope, emptyMap()))
    }

    @Test
    fun resolve_interpolatesLegacyInlineItemStrings() {
        val state = mapOf<String, Any?>(
            "user" to mapOf("name" to "Asha")
        )
        val scope = RepeatScope(
            item = mapOf("title" to "Prep Oven", "description" to "Preheat to 375 F"),
            index = 0
        )

        assertEquals("Prep Oven", FlatExprResolver.resolve("{${'$'}item.title}", state, scope, emptyMap()))
        assertEquals("Prep Oven", FlatExprResolver.resolve("{{${'$'}item.title}}", state, scope, emptyMap()))
        assertEquals("Step: Prep Oven", FlatExprResolver.resolve("Step: ${'$'}{${'$'}item.title}", state, scope, emptyMap()))
        assertEquals("Title: Prep Oven", FlatExprResolver.resolve("Title: ${'$'}{title}", state, scope, emptyMap()))
        assertEquals("Hello Asha", FlatExprResolver.resolve("Hello ${'$'}{/user/name}", state, scope, emptyMap()))
        assertEquals("Step 1.", FlatExprResolver.resolve("Step ${'$'}{index_1}.", state, scope, emptyMap()))
        assertEquals("Step 1: Prep Oven", FlatExprResolver.resolve("Step {${'$'}index + 1}: {${'$'}item.title}", state, scope, emptyMap()))
    }

    @Test
    fun resolve_recursesInsideMapsAndLists() {
        val state = mapOf<String, Any?>("user" to mapOf("name" to "Ravi"))
        val scope = RepeatScope(index = 2)
        val value = mapOf(
            "title" to mapOf("\$state" to "/user/name"),
            "meta" to listOf(mapOf("\$index" to true), "ok")
        )

        val resolved = FlatExprResolver.resolve(value, state, scope, emptyMap()) as Map<*, *>
        assertEquals("Ravi", resolved["title"])
        val meta = resolved["meta"] as List<*>
        assertEquals(2, meta[0])
        assertEquals("ok", meta[1])
    }

    @Test
    fun resolve_supportsComputedFunctions() {
        val state = mapOf<String, Any?>("user" to mapOf("name" to "Asha"))
        val computed = mapOf<String, FlatComputedFunction>(
            "greet" to { args -> "Hi ${args["name"]}" }
        )
        val value = mapOf(
            "\$computed" to "greet",
            "args" to mapOf("name" to mapOf("\$state" to "/user/name"))
        )

        val resolved = FlatExprResolver.resolve(value, state, null, computed)
        assertEquals("Hi Asha", resolved)
    }

    @Test
    fun actionRuntime_executesActionArraysInOrder() {
        val state = mutableMapOf<String, Any?>()
        FlatSpecParser.setAtPath(state, "/items", emptyList<Any>())

        val actions = listOf(
            mapOf(
                "action" to "setState",
                "params" to mapOf("statePath" to "/step", "value" to 1)
            ),
            mapOf(
                "action" to "pushState",
                "params" to mapOf("statePath" to "/items", "value" to mapOf("name" to "A"))
            ),
            mapOf(
                "action" to "removeState",
                "params" to mapOf("statePath" to "/items", "index" to 0)
            )
        )

        FlatActionRuntime.execute(
            actionCandidate = actions,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        assertEquals(1, FlatSpecParser.getAtPath(state, "/step"))
        val items = FlatSpecParser.getAtPath(state, "/items") as List<*>
        assertTrue(items.isEmpty())
    }

    @Test
    fun actionRuntime_resolvesBindItemStatePathForSetState() {
        val state = mutableMapOf<String, Any?>(
            "items" to listOf(mapOf("name" to "Old"))
        )
        val scope = RepeatScope(
            item = mapOf("name" to "Old"),
            index = 0,
            basePath = "/items/0"
        )
        val action = mapOf(
            "action" to "setState",
            "params" to mapOf(
                "statePath" to mapOf("\$item" to "name"),
                "value" to "New"
            )
        )

        FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = scope,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        assertEquals("New", FlatSpecParser.getAtPath(state, "/items/0/name"))
    }

    @Test
    fun actionRuntime_pushStateResolvesStateValues() {
        val state = mutableMapOf<String, Any?>(
            "template" to mapOf("name" to "FromState"),
            "items" to emptyList<Any>()
        )
        val action = mapOf(
            "action" to "pushState",
            "params" to mapOf(
                "statePath" to "/items",
                "value" to mapOf("\$state" to "/template")
            )
        )

        FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        val items = FlatSpecParser.getAtPath(state, "/items") as List<*>
        assertEquals(1, items.size)
        assertEquals("FromState", (items[0] as Map<*, *>)["name"])
    }

    @Test
    fun actionRuntime_validateFormWritesResult() {
        val state = mutableMapOf<String, Any?>()
        val action = mapOf(
            "action" to "validateForm",
            "params" to mapOf("statePath" to "/formValidation")
        )

        FlatActionRuntime.execute(
            actionCandidate = action,
            stateStore = state,
            repeatScope = null,
            computedFunctions = emptyMap(),
            onOpenUrl = {}
        )

        val result = FlatSpecParser.getAtPath(state, "/formValidation") as Map<*, *>
        assertEquals(true, result["valid"])
        assertTrue((result["errors"] as Map<*, *>).isEmpty())
    }

    @Test
    fun watchRuntime_triggersOnlyOnStateChange() {
        val elements = mapOf(
            "root" to FlatElement(
                type = "Stack",
                props = mapOf("direction" to "vertical"),
                children = emptyList(),
                watch = mapOf(
                    "/flag" to mapOf(
                        "action" to "setState",
                        "params" to mapOf("statePath" to "/touched", "value" to true)
                    )
                )
            )
        )
        val runtime = FlatWatchRuntime(elements)

        val initial = runtime.collectTriggered(mapOf("flag" to false))
        val unchanged = runtime.collectTriggered(mapOf("flag" to false))
        val changed = runtime.collectTriggered(mapOf("flag" to true))
        val stableAgain = runtime.collectTriggered(mapOf("flag" to true))

        assertTrue(initial.isEmpty())
        assertTrue(unchanged.isEmpty())
        assertEquals(1, changed.size)
        assertFalse(stableAgain.isNotEmpty())
    }

    @Test
    fun extractFlatTableModel_keepsWeatherInTableMode() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "forecast": [ { "day": "Mon", "temp": "29C", "humidity": "65%" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "rows"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3"] },
                "h1": { "type": "Text", "props": { "text": "Day" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Temperature" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Humidity" }, "children": [] },
                "rows": { "type": "Stack", "props": { "direction": "vertical" }, "repeat": { "statePath": "/forecast" }, "children": ["row"] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "day" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "temp" } }, "children": [] },
                "c3": { "type": "Text", "props": { "text": { "${'$'}item": "humidity" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)!!
        val table = parsed.elements["table"]!!
        val model = extractFlatTableModel(
            containerChildren = table.children,
            containerProps = table.props,
            elements = parsed.elements,
            state = parsed.state,
            compactScreen = true
        )

        assertNotNull(model)
        assertTrue(model!!.isWeather)
        assertEquals(FlatTableRenderMode.WEATHER_CARDS, model.renderMode)
    }

    @Test
    fun extractFlatTableModel_overridesGenericDomainHintForWeatherSignals() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "forecast": [ { "day": "Sun, Apr 12", "condition": "Partly sunny", "high": "34Â°C", "low": "22Â°C" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "domain": "generic", "preferredPresentation": "table" },
                  "children": ["header", "rows"]
                },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3", "h4"] },
                "h1": { "type": "Text", "props": { "text": "Date" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Conditions" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "High (Â°C/Â°F)" }, "children": [] },
                "h4": { "type": "Text", "props": { "text": "Low (Â°C/Â°F)" }, "children": [] },
                "rows": { "type": "Stack", "props": { "direction": "vertical" }, "repeat": { "statePath": "/forecast" }, "children": ["row"] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3", "c4"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "day" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "condition" } }, "children": [] },
                "c3": { "type": "Text", "props": { "text": { "${'$'}item": "high" } }, "children": [] },
                "c4": { "type": "Text", "props": { "text": { "${'$'}item": "low" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)!!
        val table = parsed.elements["table"]!!
        val model = extractFlatTableModel(
            containerChildren = table.children,
            containerProps = table.props,
            elements = parsed.elements,
            state = parsed.state,
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals("weather", model!!.domain)
        assertEquals("cards", model.preferredPresentation)
        assertEquals(FlatTableRenderMode.WEATHER_CARDS, model.renderMode)
    }

    @Test
    fun extractFlatTableModel_usesResponsiveCardsForWideGenericTableOnCompactScreen() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "rows": [ { "id": "a", "code": "A", "alpha": "Red", "beta": "Blue", "gamma": "Green" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "rows"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3", "h4"] },
                "h1": { "type": "Text", "props": { "text": "Code" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Alpha" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Beta" }, "children": [] },
                "h4": { "type": "Text", "props": { "text": "Gamma" }, "children": [] },
                "rows": { "type": "Stack", "props": { "direction": "vertical" }, "repeat": { "statePath": "/rows", "key": "id" }, "children": ["row"] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3", "c4"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "code" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "alpha" } }, "children": [] },
                "c3": { "type": "Text", "props": { "text": { "${'$'}item": "beta" } }, "children": [] },
                "c4": { "type": "Text", "props": { "text": { "${'$'}item": "gamma" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)!!
        val table = parsed.elements["table"]!!
        val model = extractFlatTableModel(
            containerChildren = table.children,
            containerProps = table.props,
            elements = parsed.elements,
            state = parsed.state,
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, model!!.renderMode)
        assertEquals(4, model.columns)
    }

    @Test
    fun extractFlatTableModel_keepsTwoColumnFactsAsSimpleTable() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "row1", "row2"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2"] },
                "h1": { "type": "Text", "props": { "text": "Metric" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Value" }, "children": [] },
                "row1": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["r11", "r12"] },
                "r11": { "type": "Text", "props": { "text": "Humidity" }, "children": [] },
                "r12": { "type": "Text", "props": { "text": "60%" }, "children": [] },
                "row2": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["r21", "r22"] },
                "r21": { "type": "Text", "props": { "text": "Wind" }, "children": [] },
                "r22": { "type": "Text", "props": { "text": "8 km/h" }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)!!
        val table = parsed.elements["table"]!!
        val model = extractFlatTableModel(
            containerChildren = table.children,
            containerProps = table.props,
            elements = parsed.elements,
            state = parsed.state,
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals(FlatTableRenderMode.TABLE, model!!.renderMode)
        assertEquals(2, model.columns)
        assertEquals(2, model.rows)
    }

    @Test
    fun extractFlatTableModel_usesFlightCardsWhenFlightHeadersDetected() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "rows": [ { "id": "a", "airline": "IndiGo", "dep": "06:00", "arr": "08:15", "fare": "INR 5,499" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["header", "rows"] },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3", "h4"] },
                "h1": { "type": "Text", "props": { "text": "Airline" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Departure" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Arrival" }, "children": [] },
                "h4": { "type": "Text", "props": { "text": "Fare" }, "children": [] },
                "rows": { "type": "Stack", "props": { "direction": "vertical" }, "repeat": { "statePath": "/rows", "key": "id" }, "children": ["row"] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3", "c4"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "airline" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "dep" } }, "children": [] },
                "c3": { "type": "Text", "props": { "text": { "${'$'}item": "arr" } }, "children": [] },
                "c4": { "type": "Text", "props": { "text": { "${'$'}item": "fare" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)!!
        val table = parsed.elements["table"]!!
        val model = extractFlatTableModel(
            containerChildren = table.children,
            containerProps = table.props,
            elements = parsed.elements,
            state = parsed.state,
            compactScreen = true
        )

        assertNotNull(model)
        assertTrue(model!!.isFlight)
        assertEquals(FlatTableRenderMode.FLIGHT_CARDS, model.renderMode)
    }

    @Test
    fun extractFlatTableModel_keepsCardsFirstForWeatherEvenWithTablePreferenceHint() {
        val payload = JsonParser.parseString(
            """
            {
              "root": "main",
              "state": { "forecast": [ { "day": "Mon", "temp": "29C", "humidity": "65%" } ] },
              "elements": {
                "main": { "type": "Stack", "props": { "direction": "vertical" }, "children": ["table"] },
                "table": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "domain": "weather", "preferredPresentation": "table" },
                  "children": ["header", "rows"]
                },
                "header": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["h1", "h2", "h3"] },
                "h1": { "type": "Text", "props": { "text": "Day" }, "children": [] },
                "h2": { "type": "Text", "props": { "text": "Temperature" }, "children": [] },
                "h3": { "type": "Text", "props": { "text": "Humidity" }, "children": [] },
                "rows": { "type": "Stack", "props": { "direction": "vertical" }, "repeat": { "statePath": "/forecast" }, "children": ["row"] },
                "row": { "type": "Stack", "props": { "direction": "horizontal" }, "children": ["c1", "c2", "c3"] },
                "c1": { "type": "Text", "props": { "text": { "${'$'}item": "day" } }, "children": [] },
                "c2": { "type": "Text", "props": { "text": { "${'$'}item": "temp" } }, "children": [] },
                "c3": { "type": "Text", "props": { "text": { "${'$'}item": "humidity" } }, "children": [] }
              }
            }
            """.trimIndent()
        )

        val parsed = FlatSpecParser.parse(payload)!!
        val table = parsed.elements["table"]!!
        val model = extractFlatTableModel(
            containerChildren = table.children,
            containerProps = table.props,
            elements = parsed.elements,
            state = parsed.state,
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals("table", model!!.preferredPresentation)
        assertEquals(FlatTableRenderMode.WEATHER_CARDS, model.renderMode)
    }

    @Test
    fun extractDirectTableModel_infersWeatherFromCompactColumns() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "day", "label" to "Day"),
                mapOf("key" to "condition", "label" to "Conditions"),
                mapOf("key" to "high", "label" to "High (Â°C/Â°F)"),
                mapOf("key" to "low", "label" to "Low (Â°C/Â°F)")
            ),
            "statePath" to "/forecast",
            "domain" to "generic",
            "preferredPresentation" to "table"
        )
        val state = mapOf<String, Any?>(
            "forecast" to listOf(
                mapOf("day" to "Sun", "condition" to "Clear", "high" to "34Â°C/94Â°F", "low" to "23Â°C/73Â°F")
            )
        )

        val model = extractDirectTableModel(
            props = props,
            state = state,
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals("weather", model!!.domain)
        assertEquals(FlatTableRenderMode.WEATHER_CARDS, model.renderMode)
        assertEquals(1, model.rows.size)
        assertEquals("Sun", model.rows.first().first())
    }

    @Test
    fun extractDirectTableModel_usesHorizontalScrollForWideGenericCompactTable() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "code", "label" to "Code"),
                mapOf("key" to "alpha", "label" to "Alpha"),
                mapOf("key" to "beta", "label" to "Beta"),
                mapOf("key" to "gamma", "label" to "Gamma")
            ),
            "rows" to listOf(
                mapOf("code" to "A", "alpha" to "Red", "beta" to "Blue", "gamma" to "Green")
            ),
            "domain" to "generic",
            "preferredPresentation" to "table"
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals("generic", model!!.domain)
        assertEquals(FlatTableRenderMode.TABLE_HORIZONTAL_SCROLL, model.renderMode)
        assertEquals(4, model.columns.size)
    }

    @Test
    fun looksLikeMarketHoldingsTable_detectsPortfolioHoldings() {
        val headers = listOf("Ticker", "Current Value", "Day's Change ($)", "Day's Change (%)")
        val rows = listOf(
            listOf("AAPL", "$1,752.80", "+$21.80", "+1.26%"),
            listOf("GOOG", "$11,252.50", "+$27.50", "+0.24%"),
            listOf("TSLA", "$2,348.40", "-$69.00", "-2.85%")
        )

        assertTrue(looksLikeMarketHoldingsTable(headers, rows, domain = "generic"))
        assertTrue(looksLikeMarketHoldingsTable(headers, rows, domain = "schedule"))
        assertFalse(
            looksLikeMarketHoldingsTable(
                headers = listOf("Feature", "Option A", "Option B", "Option C"),
                rows = listOf(listOf("Battery", "Long", "Medium", "Short")),
                domain = "comparison"
            )
        )
    }

    @Test
    fun extractDirectTableModel_mapsGenericColumnKeysByPositionWhenRowKeysDiffer() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "column_1", "label" to "Day"),
                mapOf("key" to "column_2", "label" to "Conditions"),
                mapOf("key" to "column_3", "label" to "High"),
                mapOf("key" to "column_4", "label" to "Low")
            ),
            "rows" to listOf(
                mapOf("day" to "Sun", "conditions" to "Clear", "high_c" to "34", "low_c" to "22")
            ),
            "domain" to "generic",
            "preferredPresentation" to "table"
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals(listOf("Sun", "Clear", "34", "22"), model!!.rows.first())
    }

    @Test
    fun extractDirectTableModel_supportsRowsCellsShape() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "airline", "label" to "Airline"),
                mapOf("key" to "price", "label" to "Price")
            ),
            "rows" to listOf(
                mapOf("cells" to listOf("IndiGo", "INR 5,488"))
            )
        )

        val model = extractDirectTableModel(
            props = props,
            state = emptyMap(),
            compactScreen = true
        )

        assertNotNull(model)
        assertEquals(listOf("IndiGo", "INR 5,488"), model!!.rows.first())
    }

    @Test
    fun extractDirectTableModel_resolvesDynamicExpressionsInsideRows() {
        val props = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "city", "label" to "City"),
                mapOf("key" to "temp", "label" to "Temp")
            ),
            "rows" to listOf(
                mapOf(
                    "city" to mapOf("${'$'}state" to "/weather/city"),
                    "temp" to mapOf("${'$'}template" to "${'$'}{/weather/temp_c}Â°C")
                )
            )
        )
        val state = mapOf<String, Any?>(
            "weather" to mapOf(
                "city" to "Bengaluru",
                "temp_c" to 31
            )
        )

        val model = extractDirectTableModel(
            props = props,
            state = state,
            compactScreen = false
        )

        assertNotNull(model)
        assertEquals(listOf("Bengaluru", "31Â°C"), model!!.rows.first())
    }
    @Test
    fun formulaHelpers_parseMortgageFractionAndScripts() {
        val converted = plainBracketFractionToLatex("M = P [ i(1 + i)^n ] / [ (1 + i)^n - 1 ]")
        val fraction = parseFormulaFraction(converted)
        val segments = parseFormulaSegments("""M = P \frac{i(1+i)^n}{(1+i)^n - 1}""")

        assertNotNull(fraction)
        assertEquals("M = P", fraction!!.prefix)
        assertEquals("i(1 + i)^n", fraction.numerator)
        assertEquals("(1 + i)^n - 1", fraction.denominator)
        assertEquals(2, segments.size)
        assertTrue(segments[0] is FormulaVisualSegment.TextSegment)
        assertTrue(segments[1] is FormulaVisualSegment.FractionSegment)
        val mortgageFraction = segments[1] as FormulaVisualSegment.FractionSegment
        assertEquals("i(1+i)^n", mortgageFraction.numerator)
        assertEquals("(1+i)^n - 1", mortgageFraction.denominator)
        assertEquals("M = P ×", formulaDisplayTextSegment((segments[0] as FormulaVisualSegment.TextSegment).value, true))
        assertEquals("A =", formulaDisplayTextSegment("A =", true))
        assertTrue(looksLikeFormulaText("M = P [ i(1 + i)^n ] / [ (1 + i)^n - 1 ]"))
        assertEquals("Payment = principal", normalizeFormulaText("""Payment = \text{principal}"""))
    }

    @Test
    fun formulaHelpers_parseMultipleFractionsAndPlainScripts() {
        val areaSegments = parseFormulaSegments("""A = \frac{1}{2}bh + \frac{\pi r^2}{2}""")
        val plainSegments = parseFormulaSegments("E = mc^2")
        val roiSegments = parseFormulaSegments("""ROI = \frac{\text{Gain} - \text{Cost}}{\text{Cost}} \times 100\%""")

        assertEquals(4, areaSegments.size)
        assertTrue(areaSegments[0] is FormulaVisualSegment.TextSegment)
        assertTrue(areaSegments[1] is FormulaVisualSegment.FractionSegment)
        assertTrue(areaSegments[2] is FormulaVisualSegment.TextSegment)
        assertTrue(areaSegments[3] is FormulaVisualSegment.FractionSegment)
        assertEquals("E = mc2", formulaAnnotatedString((plainSegments.single() as FormulaVisualSegment.TextSegment).value).text)
        assertEquals(3, roiSegments.size)
        val roiFraction = roiSegments[1] as FormulaVisualSegment.FractionSegment
        assertEquals("Gain - Cost", normalizeFormulaText(roiFraction.numerator))
        assertEquals("Cost", normalizeFormulaText(roiFraction.denominator))
        assertTrue((roiSegments[2] as FormulaVisualSegment.TextSegment).value.contains("100"))
        assertTrue(formulaAnnotatedString("""x = \sqrt{b^2 - 4ac}""").text.contains("b2 - 4ac"))
    }

    @Test
    fun extractDirectTableModel_routesFormulaTables() {
        val variableProps = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "variable", "label" to "Variable"),
                mapOf("key" to "description", "label" to "Description"),
                mapOf("key" to "value", "label" to "Value")
            ),
            "rows" to listOf(
                mapOf("variable" to "P", "description" to "Principal loan amount", "value" to "${'$'}420,000")
            ),
            "domain" to "formula",
            "preferredPresentation" to "table"
        )
        val breakdownProps = mapOf<String, Any?>(
            "columns" to listOf(
                mapOf("key" to "component", "label" to "Component"),
                mapOf("key" to "amount", "label" to "Amount")
            ),
            "rows" to listOf(
                mapOf("component" to "Monthly Payment", "amount" to "${'$'}2,451.25")
            ),
            "domain" to "calculation",
            "preferredPresentation" to "table"
        )

        val variables = extractDirectTableModel(variableProps, emptyMap(), compactScreen = true)
        val breakdown = extractDirectTableModel(breakdownProps, emptyMap(), compactScreen = true)

        assertNotNull(variables)
        assertTrue(isFormulaVariablesTable(variables!!))
        assertNotNull(breakdown)
        assertEquals("formula", breakdown!!.domain)
        assertTrue(isCalculationBreakdownTable(breakdown))
    }
}

