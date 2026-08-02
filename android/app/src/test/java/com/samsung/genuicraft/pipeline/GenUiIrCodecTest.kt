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
            root=Column([title,button],gap="md")
            title=Text("Continue?","h2")
            button=Button("Continue",onPress=Event("continue",{selectedId:${'$'}/selectedId},true,"/eventResult"))
            </a2ui>
            """.trimIndent(),
        )
        val root = decoded.getAsJsonObject("elements").getAsJsonObject("root")
        val action = decoded.getAsJsonObject("elements").getAsJsonObject("button")
            .getAsJsonObject("on").getAsJsonObject("press")
        assertEquals("Stack", root.get("type").asString)
        assertEquals("vertical", root.getAsJsonObject("props").get("direction").asString)
        assertEquals("emitEvent", action.get("action").asString)
        assertEquals("continue", action.getAsJsonObject("params").get("name").asString)
        assertTrue(FlatSpecIngestor.ingest(decoded) is FlatSpecIngestResult.CanonicalFlatSpec)
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
