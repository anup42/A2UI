package com.samsung.genuicraft.pipeline

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiIrCodecTest {
    @Test
    fun sharedReferenceInventoryMatchesPythonFixture() {
        val resource = requireNotNull(javaClass.classLoader?.getResourceAsStream("flat_spec_reference_inventory_v1.json"))
        val fixture = resource.reader(Charsets.UTF_8).use { JsonParser.parseReader(it).asJsonObject }
        assertEquals(FlatSpecReferenceSemantics.VERSION, fixture.get("version").asString)
        fixture.getAsJsonArray("cases").forEach { rawCase ->
            val case = rawCase.asJsonObject
            val actual = FlatSpecReferenceSemantics.references(case.getAsJsonObject("element"))
                .map { listOf(it.targetId, it.sourcePath, it.kind) }
            val expected = case.getAsJsonArray("references").map { raw ->
                raw.asJsonArray.map { it.asString }
            }
            assertEquals(case.get("name").asString, expected, actual)
        }
    }

    @Test
    fun allNativeFormatsRoundTripOneRendererGraph() {
        val original = richSpec()
        val expected = FlatSpecIdRewriter.rewrite(original, shorten = true)

        val compact = CompactIrCodec.decode(CompactIrCodec.encode(original))
        val encodedExpress = A2uiExpressCodec.encode(original)
        val express = A2uiExpressCodec.decode(encodedExpress)
        val wire = A2uiWireCodec.decode(A2uiWireCodec.encode(original))

        assertEquals(expected, compact)
        assertEquals(expected, express)
        assertEquals(expected, wire)
        assertTrue(FlatSpecIngestor.ingest(CompactIrCodec.encode(original)) is FlatSpecIngestResult.CanonicalFlatSpec)
        assertTrue(
            FlatSpecIngestor.ingest(JsonPrimitive(A2uiExpressCodec.encode(original)))
                is FlatSpecIngestResult.CanonicalFlatSpec,
        )
    }

    @Test
    fun expressEventCompilesToEmitEventAndAliasesToStack() {
        val decoded = A2uiExpressCodec.decode(
            """
            <a2ui>
            root=Column([title,button,link],gap="md")
            title=Text("Continue?","h2")
            button=Button("Continue",onPress=Event("continue",{selectedId:${'$'}/selectedId},true,"/eventResult"))
            link=Button("Support",onPress=openUrl("https://www.samsung.com/support/"))
            </a2ui>
            """.trimIndent(),
        )
        val root = decoded.getAsJsonObject("elements").getAsJsonObject("root")
        val action = decoded.getAsJsonObject("elements").getAsJsonObject("button")
            .getAsJsonObject("on").getAsJsonObject("press")
        val linkAction = decoded.getAsJsonObject("elements").getAsJsonObject("link")
            .getAsJsonObject("on").getAsJsonObject("press")
        assertEquals("Stack", root.get("type").asString)
        assertEquals("vertical", root.getAsJsonObject("props").get("direction").asString)
        assertEquals("emitEvent", action.get("action").asString)
        assertEquals("continue", action.getAsJsonObject("params").get("name").asString)
        assertEquals("openUrl", linkAction.get("action").asString)
        assertEquals("https://www.samsung.com/support/", linkAction.getAsJsonObject("params").get("url").asString)
        assertTrue(FlatSpecIngestor.ingest(decoded) is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun expressPromptExampleIsValidSelectedFormatPayload() {
        val example = """
            <a2ui>
            root=Column([title,card,details,action,link],gap="md")
            title=Text("Example title","h2")
            status=Text("Example status","body")
            card=Card([status],"Summary")
            details=Table(["Detail","Value"],_,[["Example key","Example value"]],"Details","status","table")
            action=Button("Continue","primary",onPress=Event("continue",{},true,"/continueResult"))
            link=Button("Open support","primary",onPress=openUrl("https://www.samsung.com/support/"))
            </a2ui>
        """.trimIndent()

        val decoded = A2uiExpressCodec.decode(example)
        val ingested = FlatSpecIngestor.ingest(JsonPrimitive(example), FlatSpecIngestMode.STRICT)

        assertEquals("root", decoded.get("root").asString)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
        assertEquals(
            GenUiIrFormat.A2UI_EXPRESS_V1,
            (ingested as FlatSpecIngestResult.CanonicalFlatSpec).sourceFormat,
        )
    }

    @Test
    fun expressDecoderRepairsBoundedMissingCloserAndTrailingIconUrlAssignment() {
        val generated = """
            <a2ui>
            root=Column([Text("Order A1042"),Text("Carrier: SwiftShip"),Text("ETA: Today by 4:30 PM"),Button("Track Package","primary",onPress=Event("track_package",{},true,"/track/A1042"))]
            icon=Icon("star")="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/star.svg"
            </a2ui>
        """.trimIndent()

        val decoded = A2uiExpressCodec.decode(generated)
        val text = decoded.toString()

        assertTrue(text.contains("SwiftShip"))
        assertTrue(text.contains("Today by 4:30 PM"))
        assertTrue(text.contains("emitEvent"))
        assertTrue(FlatSpecIngestor.ingest(JsonPrimitive(generated), FlatSpecIngestMode.STRICT) is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun expressDecoderNormalizesHumanLabelUsedAsStackGap() {
        val generated = """
            <a2ui>
            root=Column([status,action],gap="tracking link")
            status=Text("Out for delivery")
            action=Button("Track package",onPress=Event("track_package"))
            </a2ui>
        """.trimIndent()

        val decoded = A2uiExpressCodec.decode(generated)
        val root = decoded.getAsJsonObject("elements").getAsJsonObject("root")

        assertEquals("md", root.getAsJsonObject("props").get("gap").asString)
        assertTrue(FlatSpecIngestor.ingest(JsonPrimitive(generated), FlatSpecIngestMode.STRICT) is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun compactPromptExampleIsValidSelectedFormatPayload() {
        val example = JsonParser.parseString(
            """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","p":{"gap":"md"},"c":["title","card","details","action"]},"title":{"t":"Text","p":{"text":"Example title","variant":"h2"}},"status":{"t":"Text","p":{"text":"Example status","variant":"body"}},"card":{"t":"Card","p":{"title":"Summary"},"c":["status"]},"details":{"t":"Table","p":{"columns":["Detail","Value"],"rows":[["Example key","Example value"]],"title":"Details","domain":"status","preferredPresentation":"table"}},"action":{"t":"Button","p":{"label":"Continue","variant":"primary"},"o":{"press":{"action":"emitEvent","params":{"name":"continue"}}}}}}
            """.trimIndent()
        )

        val decoded = CompactIrCodec.decode(example)
        val ingested = FlatSpecIngestor.ingest(example, FlatSpecIngestMode.STRICT)

        assertEquals("root", decoded.get("root").asString)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
        assertEquals(
            GenUiIrFormat.COMPACT_IR_V2,
            (ingested as FlatSpecIngestResult.CanonicalFlatSpec).sourceFormat,
        )
    }

    @Test
    fun wireMessageStreamHonorsRootAndDataUpdates() {
        val stream = JsonParser.parseString(
            """
            [
              {"version":"v1.0","createSurface":{"surfaceId":"s","catalogId":"${GenUiA2uiCatalog.CATALOG_ID}","rootId":"screen","components":[
                {"id":"screen","component":"Column","children":["label"],"accessibility":{"label":"Screen"}},
                {"id":"label","component":"Text","text":"Ready"}
              ],"dataModel":{"status":"pending"}}},
              {"version":"v1.0","updateDataModel":{"surfaceId":"s","path":"/status","value":"ready"}}
            ]
            """.trimIndent(),
        )
        val decoded = A2uiWireCodec.decode(stream)
        assertEquals("screen", decoded.get("root").asString)
        assertEquals("ready", decoded.getAsJsonObject("state").get("status").asString)
        val screen = decoded.getAsJsonObject("elements").getAsJsonObject("screen")
        assertEquals("Stack", screen.get("type").asString)
        assertTrue(screen.getAsJsonObject("props").has("accessibility"))
    }

    private fun richSpec(): JsonObject = JsonParser.parseString(
        """
        {
          "root":"screen","state":{"selectedId":"a"},"elements":{
            "screen":{"type":"Stack","props":{"direction":"vertical","gap":"md"},"children":["title","tabs","modal"]},
            "title":{"type":"Text","props":{"text":"Options","variant":"h2"},"children":[]},
            "tabs":{"type":"Tabs","props":{"tabs":[{"title":"First","child":"first"},{"title":"Second","content":"second"}]},"children":[]},
            "first":{"type":"Card","props":{"accessibility":{"label":"First"}},"children":["firstText"]},
            "firstText":{"type":"Text","props":{"text":"One"},"children":[]},
            "second":{"type":"List","props":{},"children":["row"],"repeat":{"statePath":"/items","template":"row"}},
            "row":{"type":"Text","props":{"text":{"${'$'}item":"label"}},"children":[]},
            "modal":{"type":"Modal","props":{"trigger":"open","content":"dialog"},"children":[]},
            "open":{"type":"Button","props":{"label":"Open"},"children":[],"on":{"press":{"action":"emitEvent","params":{"name":"open"}}}},
            "dialog":{"type":"Text","props":{"text":"Dialog"},"children":[]}
          }
        }
        """.trimIndent(),
    ).asJsonObject
}
