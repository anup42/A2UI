package com.samsung.genuicraft.sdk

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiStructuredInputTest {
    private val fixture: String
        get() = requireNotNull(javaClass.classLoader?.getResource("bixby_view_mysore.json"))
            .readText(Charsets.UTF_8)

    @Test fun `captured Mysore View imports all 40 nodes and exact place data`() {
        val source = JsonParser.parseString(fixture).asJsonArray
        assertTrue(GenUiStructuredInput.recognizes(fixture))
        val result = GenUiStructuredInput.convertOrNull(GenUiRequest(fixture))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        result as GenUiConversionResult.Success
        assertEquals(GenUiStructuredInput.PROVIDER, result.provider)
        assertEquals(0, result.attempts)
        assertEquals(GenUiRepairKind.NONE, result.repairKind)
        assertTrue(result.warnings.any { it.contains("without model inference") })
        assertTrue(result.document.express.startsWith("<a2ui>"))

        val components = components(result)
        val byId = components.associateBy { it.get("id").asString }
        val ordered = byId.getValue("root").getAsJsonArray("children").map { byId.getValue(it.asString) }
        assertEquals(40, ordered.size)
        assertEquals(source.map { when (it.asJsonObject.get("\$type").asString) {
            "Markdown" -> "Text"
            "CompoundCard" -> "PlaceCard"
            else -> "Divider"
        } }, ordered.map { it.get("component").asString })

        val expectedMarkdown = source.filter { it.asJsonObject.get("\$type").asString == "Markdown" }
            .map { it.asJsonObject.get("value").asString }
        val actualMarkdown = ordered.filter { it.get("component").asString == "Text" }
            .map { it.get("text").asString }
        assertEquals(26, actualMarkdown.size)
        assertEquals(expectedMarkdown, actualMarkdown)
        assertEquals(8, ordered.count { it.get("component").asString == "Divider" })

        val sourceCards = source.filter { it.asJsonObject.get("\$type").asString == "CompoundCard" }
            .map { it.asJsonObject }
        val placeCards = ordered.filter { it.get("component").asString == "PlaceCard" }
        assertEquals(6, placeCards.size)
        var photoCount = 0
        var imageClickCount = 0
        var actionCount = 0
        sourceCards.zip(placeCards).forEach { (sourceCard, place) ->
            val blocks = sourceCard.getAsJsonArray("contents").map { it.asJsonObject }
            val title = blocks.one("TitleArea").getAsJsonObject("slot1").get("text").asString
                .removeSurrounding("**")
            assertEquals(title, place.get("title").asString)

            val facts = blocks.one("SingleLine").getAsJsonArray("contents").map { it.asJsonObject }
                .filter { it.get("\$type").asString == "Text" }
                .mapNotNull { it.get("value")?.takeUnless { value -> value.isJsonNull }?.asString }
            val rating = facts.firstOrNull { it.matches(Regex("^[0-5](?:\\.[0-9]+)?$")) }
            val reviews = facts.firstOrNull { it.matches(Regex("^\\([0-9][0-9,]*\\)$")) }
            assertEquals(rating, place.get("rating")?.asString)
            assertEquals(reviews, place.get("reviewCount")?.asString)

            val sourceAddress = blocks.one("ActionLinks").getAsJsonArray("items")[0].asJsonObject
            assertEquals(sourceAddress.get("label").asString, place.get("address").asString)
            assertEquals(sourceAddress.get("url").asString, place.get("addressUrl").asString)

            val sourceImages = blocks.one("ImageList").getAsJsonArray("images")
            val images = place.getAsJsonArray("images")
            assertEquals(sourceImages.size(), images.size())
            photoCount += images.size()
            sourceImages.forEachIndexed { index, image ->
                assertEquals(image.asJsonObject.get("url").asString, images[index].asJsonObject.get("url").asString)
                assertEquals(image.asJsonObject.get("title").asString, images[index].asJsonObject.get("title").asString)
                val clickUrl = image.asJsonObject.get("onClickUrl")?.takeUnless { it.isJsonNull }?.asString
                assertEquals(clickUrl, images[index].asJsonObject.get("onClickUrl")?.asString)
                if (clickUrl != null) imageClickCount++
            }

            val sourceActions = blocks.one("LinkButtons").getAsJsonArray("items")
            val actions = place.getAsJsonArray("actions")
            assertEquals(sourceActions.size(), actions.size())
            actionCount += actions.size()
            sourceActions.forEachIndexed { index, action ->
                assertEquals(action.asJsonObject.get("label").asString, actions[index].asJsonObject.get("label").asString)
                assertEquals(action.asJsonObject.get("url").asString, actions[index].asJsonObject.get("url").asString)
            }
        }
        assertEquals(18, photoCount)
        assertEquals(9, imageClickCount)
        assertEquals(11, actionCount)
        assertFalse(placeCards[4].has("rating")) // Source supplied no Mysore Palace rating.
        assertEquals("(2590)", placeCards[2].get("reviewCount").asString) // Star glyph is decoration.
    }

    @Test fun `typed outer View wrappers import the same nodes`() {
        val nodes = JsonParser.parseString(fixture).asJsonArray
        val typed = JsonObject().apply { addProperty("\$type", "View"); add("contents", nodes) }
        val outer = JsonObject().apply { add("view", typed) }
        val capturedEnvelope = JsonObject().apply { addProperty("type", "View"); addProperty("value", fixture) }
        val capturedStream = JsonArray().apply { add(capturedEnvelope) }
        listOf(typed, outer, capturedEnvelope, capturedStream).forEach { wrapper ->
            assertTrue(GenUiStructuredInput.recognizes(wrapper.toString()))
            val result = GenUiStructuredInput.convertOrNull(GenUiRequest(wrapper.toString()))
            assertTrue(result.toString(), result is GenUiConversionResult.Success)
            assertEquals(6, components(result as GenUiConversionResult.Success)
                .count { it.get("component").asString == "PlaceCard" })
        }
    }

    @Test fun `mixed Text and View envelopes retain source order`() {
        val stream = JsonArray().apply {
            add(JsonObject().apply { addProperty("type", "Text"); addProperty("value", "Before card") })
            add(JsonObject().apply { addProperty("type", "View"); addProperty("value", fixture) })
            add(JsonObject().apply { addProperty("type", "Text"); addProperty("value", "After card") })
        }
        val result = GenUiStructuredInput.convertOrNull(GenUiRequest(stream.toString()))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        val components = components(result as GenUiConversionResult.Success)
        val byId = components.associateBy { it.get("id").asString }
        val ordered = byId.getValue("root").getAsJsonArray("children").map { byId.getValue(it.asString) }
        assertEquals(42, ordered.size)
        assertEquals("Before card", ordered.first().get("text").asString)
        assertEquals("After card", ordered.last().get("text").asString)
    }

    @Test fun `ordinary Markdown stays on both model conversion paths`() = runBlocking {
        val markdown = "# Two-day trip\n\nVisit Mysore Palace."
        assertFalse(GenUiStructuredInput.recognizes(markdown))
        assertNull(GenUiStructuredInput.convertOrNull(GenUiRequest(markdown)))
        val provider = CountingProvider()
        val standard = GenUiConverter.withPrompt(provider, "contract", useSourceBindings = false)
            .convert(GenUiRequest(markdown))
        assertTrue(standard.toString(), standard is GenUiConversionResult.Success)
        assertEquals(1, provider.calls)

        val trained = CountingProvider()
        val contract = object : TrainedPrompt {
            override fun prompt(response: String): GenUiPrompt = GenUiPrompt("contract", response)
        }
        val result = GenUiTrainedConverter(trained, contract).convert(GenUiRequest(markdown))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        assertEquals(1, trained.calls)
    }

    @Test fun `both model converters bypass inference for structured View`() = runBlocking {
        val provider = CountingProvider()
        val standard = GenUiConverter.withPrompt(provider, "contract")
            .convert(GenUiRequest(fixture))
        assertTrue(standard.toString(), standard is GenUiConversionResult.Success)
        assertEquals(0, provider.calls)

        val trained = CountingProvider()
        val contract = object : TrainedPrompt {
            override fun prompt(response: String): GenUiPrompt = error("Structured input must not build a prompt.")
        }
        val result = GenUiTrainedConverter(trained, contract).convert(GenUiRequest(fixture))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        assertEquals(0, trained.calls)
    }

    @Test fun `unsupported content and unsafe URLs fail without model fallback`() {
        val cases = listOf<(JsonArray) -> Unit>(
            { it[0].asJsonObject.addProperty("unexpected", "must not disappear") },
            { it[0].asJsonObject.addProperty("\$type", "UnsupportedBlock") },
            { firstCard(it).getAsJsonArray("contents").add(JsonObject().apply {
                addProperty("\$type", "FactBadge"); addProperty("value", "Open now")
            }) },
            { firstCard(it).getAsJsonArray("contents").one("ImageList")
                .getAsJsonArray("images")[0].asJsonObject.addProperty("url", "javascript:alert(1)") },
            { firstCard(it).getAsJsonArray("contents").one("ImageList")
                .getAsJsonArray("images")[0].asJsonObject.addProperty("url", "http://www.google.com/photo.jpg") },
            { firstCard(it).getAsJsonArray("contents").one("ImageList")
                .getAsJsonArray("images")[0].asJsonObject.addProperty("onClickUrl", "javascript:alert(1)") },
            { firstCard(it).getAsJsonArray("contents").one("ImageList")
                .getAsJsonArray("images")[0].asJsonObject.addProperty("onClickUrl", "tel:08073092450") },
            { firstCard(it).getAsJsonArray("contents").one("LinkButtons")
                .getAsJsonArray("items")[0].asJsonObject.addProperty("url", "file:///tmp/a") },
            { firstCard(it).getAsJsonArray("contents").one("LinkButtons")
                .getAsJsonArray("items")[0].asJsonObject.getAsJsonObject("onClick")
                .addProperty("intent", "intent://arbitrary") },
            { firstCard(it).getAsJsonArray("contents").one("ActionLinks")
                .getAsJsonArray("items")[0].asJsonObject.addProperty("url", "javascript:alert(1)") },
        )
        cases.forEachIndexed { index, change ->
            val view = JsonParser.parseString(fixture).asJsonArray
            change(view)
            val result = GenUiStructuredInput.convertOrNull(GenUiRequest(view.toString()))
            assertTrue("case $index: $result", result is GenUiConversionResult.Failure)
            result as GenUiConversionResult.Failure
            assertEquals(GenUiStructuredInput.PROVIDER, result.provider)
            assertEquals(0, result.attempts)
            assertNull(result.rawOutput)
        }
    }

    @Test fun `unrelated typed JSON is not misrecognized`() {
        val ordinary = """{"${'$'}schema":"https://example.com/schema","items":[{"title":"Mysore"}]}"""
        assertFalse(GenUiStructuredInput.recognizes(ordinary))
        assertNull(GenUiStructuredInput.convertOrNull(GenUiRequest(ordinary)))
        assertFalse(GenUiStructuredInput.recognizes("{\"\$type\":\"Other\",\"bad\": "))
        val unrelatedArray = "[{\"\$type\":\"Other\",\"value\":\"Not a Bixby View\"}]"
        assertFalse(GenUiStructuredInput.recognizes(unrelatedArray))
        assertNull(GenUiStructuredInput.convertOrNull(GenUiRequest(unrelatedArray)))
        val mixedView = "[{\"\$type\":\"Markdown\",\"value\":\"Hello\"},{\"\$type\":\"Other\",\"value\":\"Do not drop\"}]"
        assertTrue(GenUiStructuredInput.recognizes(mixedView))
        assertTrue(GenUiStructuredInput.convertOrNull(GenUiRequest(mixedView)) is GenUiConversionResult.Failure)
        val unsupportedWrappedView = JsonObject().apply {
            addProperty("type", "View")
            addProperty("value", unrelatedArray)
        }.toString()
        assertTrue(GenUiStructuredInput.recognizes(unsupportedWrappedView))
        assertTrue(GenUiStructuredInput.convertOrNull(GenUiRequest(unsupportedWrappedView)) is GenUiConversionResult.Failure)
    }

    @Test fun `malformed Bixby View is rejected before model inference`() = runBlocking {
        val malformed = fixture.dropLast(1)
        assertTrue(GenUiStructuredInput.recognizes(malformed))
        val provider = CountingProvider()
        val result = GenUiConverter.withPrompt(provider, "contract").convert(GenUiRequest(malformed))
        assertTrue(result.toString(), result is GenUiConversionResult.Failure)
        assertEquals(GenUiStructuredInput.PROVIDER, (result as GenUiConversionResult.Failure).provider)
        assertEquals(0, provider.calls)
    }

    private fun components(result: GenUiConversionResult.Success): List<JsonObject> =
        JsonParser.parseString(result.document.a2uiJson).asJsonArray[1].asJsonObject
            .getAsJsonObject("updateComponents").getAsJsonArray("components").map { it.asJsonObject }

    private fun JsonArray.one(type: String): JsonObject =
        map { it.asJsonObject }.single { it.get("\$type").asString == type }

    private fun List<JsonObject>.one(type: String): JsonObject =
        single { it.get("\$type").asString == type }

    private fun firstCard(view: JsonArray): JsonObject =
        view.map { it.asJsonObject }.first { it.get("\$type").asString == "CompoundCard" }

    private class CountingProvider : GenUiProvider {
        override val id = "counting-provider"
        var calls = 0
        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
            calls++
            return GenUiModelOutput("<a2ui>\nroot=Column([a])\na=Text(\"# Two-day trip\\n\\nVisit Mysore Palace.\")\n</a2ui>", "stub")
        }
    }
}
