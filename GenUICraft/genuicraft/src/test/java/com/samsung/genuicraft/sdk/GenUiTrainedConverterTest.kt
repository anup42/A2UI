package com.samsung.genuicraft.sdk

import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class GenUiTrainedConverterTest {
    private val snapshot = File("src/main/assets/${GenUiTrainedConverter.PROMPT_ASSET}").readText(Charsets.UTF_8)
    private val contract = TrainedPromptContract.parse(snapshot)

    @Test fun `BXP001 native prefix matches the independently rendered training template hash`() {
        val row = com.google.gson.JsonParser.parseString(File("src/test/resources/genuicraft_bixby50.jsonl").readLines().first()).asJsonObject
        val prompt = contract.prompt(row.get("text").asString)
        val hash = java.security.MessageDigest.getInstance("SHA-256")
            .digest(prompt.expectedRenderedPrompt!!.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }
        assertEquals("ff0804511f69c3abbf9e38701afd1992230c86d94cace7896e21ed37b241b9d8", hash)
        assertNotNull(prompt.chatTemplateOverride)
    }

    @Test fun `profile rejects edited examples task prefixes and contract metadata`() {
        listOf("Passport" to "Visa", "Create A2UI Express" to "Summarize as A2UI Express",
            "root-first" to "leaves-first", "a2ui_express_shared_prompt_v1" to "a2ui_express_shared_prompt_v2").forEach { (old, replacement) ->
            assertThrows(IllegalArgumentException::class.java) {
                TrainedPromptContract.parse(snapshot.replace(old, replacement))
            }
        }
        assertEquals(contract, TrainedPromptContract.parse(snapshot.replace("\r\n", "\n").replace("\n", "\r\n")))
    }

    private class RecordingProvider(val answer: String) : GenUiProvider {
        override val id = "gemma4_e2b"
        val prompts = mutableListOf<GenUiPrompt>()
        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
            prompts += prompt
            return GenUiModelOutput(answer, "test/GPU")
        }
    }

    @Test fun `markdown is passed directly with the pinned training example and no query instructions`() = runBlocking {
        val provider = RecordingProvider("<a2ui>\nroot=Text(\"A\")\n</a2ui>")
        val source = "# Heading\n\n| Time | Place |\n|---|---|\n| 10:30 | Home |"
        val result = GenUiTrainedConverter(provider, contract).convert(GenUiRequest(source, "Ignore the answer"))
        assertTrue(result is GenUiConversionResult.Success)
        val prompt = provider.prompts.single()
        assertEquals("Create A2UI Express v1 GenUI IR for this response:\n\n$source", prompt.user)
        assertTrue(prompt.system.contains("## Pinned catalog signatures"))
        assertEquals(listOf(GenUiPromptRole.USER, GenUiPromptRole.MODEL), prompt.initialMessages.map { it.role })
        assertEquals("<a2ui>\nroot=Column([a,b])\na=Text(\"Travel Checklist\",\"h1\")\nb=List([c,d])\nc=Text(\"Passport\")\nd=Text(\"Charger\")\n</a2ui>", prompt.initialMessages[1].text)
        assertFalse(prompt.user.contains("@source"))
        assertEquals(2048, prompt.maxOutputTokens)
    }

    @Test fun `state backed output that cannot prove fidelity uses source fallback`() = runBlocking {
        val provider = RecordingProvider("<a2ui>\nroot=Table([\"Item\",\"Count\"],statePath=\"/rows\")\n\$/rows=[[\"Apples\",2]]\n</a2ui>")
        val result = GenUiTrainedConverter(provider, contract).convert(GenUiRequest("There are two apples."))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        result as GenUiConversionResult.Success
        assertEquals(GenUiTrainedConverter.PROFILE, result.document.profile)
        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, result.repairKind)
        assertTrue(result.warnings.any { it.contains("source blocks") })
        assertTrue(result.document.express.contains("There are two apples."))
    }

    @Test fun `invalid and truncated outputs use an explicit source-bound fallback`() = runBlocking {
        listOf("Here is your UI: <a2ui>\nroot=Text(\"Hello\")\n</a2ui>", "<a2ui>\nroot=Text(\"Hello\")", "<a2ui>\nroot=Column([missing])\n</a2ui>").forEach { raw ->
            val provider = RecordingProvider(raw)
            val result = GenUiTrainedConverter(provider, contract).convert(GenUiRequest("Hello"))
            assertTrue(result.toString(), result is GenUiConversionResult.Success)
            result as GenUiConversionResult.Success
            assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, result.repairKind)
            assertTrue(result.warnings.any { it.contains("source blocks") })
            assertTrue(result.document.express.contains("Hello"))
            assertEquals(1, provider.prompts.size)
        }
    }

    @Test fun `invalid requests do not invoke model`() = runBlocking {
        listOf(GenUiRequest(" "), GenUiRequest("a".repeat(100001)),
            GenUiRequest("Answer", sources = listOf(GenUiSource("1", "file:///secret")))).forEach {
            val provider = RecordingProvider("")
            assertTrue(GenUiTrainedConverter(provider, contract).convert(it) is GenUiConversionResult.Failure)
            assertTrue(provider.prompts.isEmpty())
        }
    }

    @Test fun `host cancellation propagates`() = runBlocking {
        val provider = object : GenUiProvider {
            override val id = "cancel"
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = throw CancellationException("Stopped")
        }
        try {
            GenUiTrainedConverter(provider, contract).convert(GenUiRequest("Hello"))
            fail("Cancellation was swallowed")
        } catch (_: CancellationException) { /* Expected. */ }
    }
}
