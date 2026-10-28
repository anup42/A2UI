package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class GenUiPresentationTest {
    private val raw = """
        <a2ui>
        root=Column([options,notes])
        options=List(items=["**Acme Buds Pro** — ₹3,999 with 40 hours battery [1].","**Beta Buds Plus** — ₹2,999 with 30 hours battery [2]."])
        notes=List(items=["Prices change; check before buying [3].","Keep your receipt."])
        </a2ui>
    """.trimIndent()

    @Test fun strictCompilerAndRawRecoveryRemainAuditableWhilePresentationIsIdempotent() {
        val strict = GenUiCompiler.compile(raw)
        assertFalse(strict.express.contains("Table("))
        val recovered = GenUiCompiler.compileWithRepair(raw, allowSourceTextFallback = false, allowGeneratedDslRepair = true)
        assertEquals(GenUiRepairKind.NONE, recovered.repairKind)
        assertEquals(strict, recovered.document)
        val presented = GenUiPresentation.prepare(strict)
        assertEquals(1, presented.convertedLists)
        assertTrue(presented.document.express.contains("Table("))
        assertTrue(presented.document.express.contains("List("))
        assertTrue(presented.document.express.contains("[1]"))
        assertTrue(presented.document.express.contains("[2]"))
        assertTrue(presented.document.express.contains("[3]"))
        assertEquals(presented.document, GenUiPresentation.prepare(presented.document).document)
    }

    @Test fun streamingPublishesCardsWithStableIdsBeforeFinalClose() {
        val stream = GenUiExpressStreamCompiler()
        val prefix = raw.substringBefore("notes=List")
        val preview = requireNotNull(stream.accept(prefix))
        val graph = A2uiWireCodec.decode(JsonParser.parseString(preview.document.a2uiJson))
        assertTrue(graph.getAsJsonObject("elements").has("options"))
        assertEquals("Table", graph.getAsJsonObject("elements").getAsJsonObject("options").get("type").asString)
        assertFalse(preview.document.express.contains("Keep your receipt"))
        val final = requireNotNull(stream.accept(raw))
        assertEquals(GenUiPresentation.prepare(GenUiCompiler.compile(raw)).document,
            GenUiPresentation.prepare(GenUiCompiler.compile(final.document.express)).document)
    }

    @Test fun trainedConverterAdaptsPresentationWithoutRelabelingItAsDslRepair() = runBlocking {
        val contract = TrainedPromptContract.parse(File("src/main/assets/${GenUiTrainedConverter.PROMPT_ASSET}").readText())
        val provider = object : GenUiProvider {
            override val id = "test"
            override suspend fun generate(prompt: GenUiPrompt) = GenUiModelOutput(raw, "test")
        }
        val result = GenUiTrainedConverter(provider, contract, false, true, false)
            .convert(GenUiRequest("Captured model output test")) as GenUiConversionResult.Success
        assertEquals(GenUiRepairKind.NONE, result.repairKind)
        assertTrue(result.document.express.contains("Table("))
        assertTrue(result.warnings.any { it.startsWith("SDK presentation:") })
        assertEquals(GenUiTrainedConverter.PROFILE, result.document.profile)
    }

    @Test fun streamingDoesNotPromoteInteractiveListsAfterDisarmingTheirActions() {
        val interactive = raw.replace(
            "hours battery [2].\"])",
            "hours battery [2].\"],onPress=setState(\"/selected\",true))",
        )
        assertNotEquals(raw, interactive)
        val preview = requireNotNull(GenUiExpressStreamCompiler().accept(interactive))
        val graph = A2uiWireCodec.decode(JsonParser.parseString(preview.document.a2uiJson))
        val list = graph.getAsJsonObject("elements").getAsJsonObject("options")
        assertEquals("List", list.get("type").asString)
        assertFalse(list.has("on"))
        assertEquals(0, GenUiPresentation.prepare(GenUiCompiler.compile(interactive)).convertedLists)
    }
}
