package com.samsung.genuicraft.sdk

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class GenUiConverterTest {
    private class Stub(private val answers: MutableList<String>) : GenUiProvider {
        override val id = "test"
        var calls = 0
        var lastPrompt: GenUiPrompt? = null
        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
            calls++
            lastPrompt = prompt
            return GenUiModelOutput(answers.removeAt(0), "stub")
        }
    }
    private fun program(text: String) = "<a2ui>\nroot=Column([answer])\nanswer=Text(${com.google.gson.Gson().toJson(text)})\n</a2ui>"

    @Test fun `valid plain response produces both formats without a repair`() = runBlocking {
        val stub = Stub(mutableListOf(program("Ready at 10:30. [2]")))
        val result = GenUiConverter.withPrompt(stub, "contract").convert(GenUiRequest("Ready at 10:30. [2]"))
        assertTrue(result is GenUiConversionResult.Success)
        result as GenUiConversionResult.Success
        assertTrue(result.document.a2uiJson.contains("updateComponents"))
        assertEquals(1, result.attempts)
    }

    @Test fun `omitted source facts trigger bounded model repair`() = runBlocking {
        val stub = Stub(mutableListOf(program("Ready."), program("Ready at 10:30. [2]")))
        val result = GenUiConverter.withPrompt(stub, "contract").convert(GenUiRequest("Ready at 10:30. [2]"))
        assertTrue(result is GenUiConversionResult.Success)
        assertEquals(2, (result as GenUiConversionResult.Success).attempts)
    }

    @Test fun `invalid output is reported without text fallback`() = runBlocking {
        val stub = Stub(mutableListOf("not a program", "still not a program"))
        val result = GenUiConverter.withPrompt(stub, "contract").convert(GenUiRequest("Complete answer."))
        assertTrue(result is GenUiConversionResult.Failure)
        assertEquals(2, stub.calls)
    }

    @Test fun `original query instructions are excluded from lossless formatting input`() = runBlocking {
        val source = "Ready. If you want, I can provide more details."
        val stub = Stub(mutableListOf(program(source)))
        val result = GenUiConverter.withPrompt(stub, "contract").convert(GenUiRequest(source, query="Avoid all follow-up offers"))
        assertTrue(result is GenUiConversionResult.Success)
        assertFalse(stub.lastPrompt!!.user.contains("Avoid all follow-up offers"))
        assertTrue(stub.lastPrompt!!.user.contains(source))
    }

    @Test fun `bound model layout receives exact original numeric values`() = runBlocking {
        val stub = Stub(mutableListOf("<a2ui>\nroot=Column([a])\na=Table(columns=\"@source.a\",rows=\"@source.b\",domain=\"weather\",preferredPresentation=\"cards\")\n</a2ui>"))
        val text = "| Day | Rain |\n|---|---|\n| Fri, Sep 11 | 55% |"
        val result = GenUiConverter.withPrompt(stub, "binding contract", useSourceBindings = true).convert(GenUiRequest(text))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        val document = (result as GenUiConversionResult.Success).document
        assertTrue(document.express.contains("Fri, Sep 11"))
        assertTrue(document.express.contains("55%"))
        assertFalse(document.express.contains("@source."))
        assertTrue(result.warnings.any { it.contains("source bindings") })
        assertTrue(stub.lastPrompt!!.user.contains("\"blocks\""))
    }

    @Test fun `missing source binding receives bounded repair without a fallback layout`() = runBlocking {
        val stub = Stub(mutableListOf(
            "<a2ui>\nroot=Column([a])\na=Text(\"@source.a\")\n</a2ui>",
            "<a2ui>\nroot=Column([a,b])\na=Text(\"@source.a\")\nb=Text(\"@source.b\")\n</a2ui>",
        ))
        val result = GenUiConverter.withPrompt(stub, "binding contract", useSourceBindings = true)
            .convert(GenUiRequest("First paragraph.\n\nSecond paragraph."))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        assertEquals(2, (result as GenUiConversionResult.Success).attempts)
        assertTrue(stub.lastPrompt!!.user.contains("@source.b"))
        assertTrue(result.document.express.contains("Second paragraph."))
    }

    @Test fun `cancellation propagates to host lifecycle`() = runBlocking {
        val provider = object : GenUiProvider {
            override val id = "cancelled"
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = throw CancellationException("Host stopped")
        }
        try {
            GenUiConverter.withPrompt(provider, "contract").convert(GenUiRequest("text"))
            fail("Cancellation was swallowed")
        } catch (_: CancellationException) { /* expected */ }
    }

    @Test fun `source attribution is added without asking the model to preserve metadata`() = runBlocking {
        val stub = Stub(mutableListOf(program("Ready. [7]")))
        val result = GenUiConverter.withPrompt(stub, "contract").convert(GenUiRequest("Ready. [7]", sources=listOf(GenUiSource("7", "https://www.who.int/", "WHO report"))))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        val sourceRecord = com.google.gson.JsonParser.parseString(
            stub.lastPrompt!!.user.substringAfterLast('\n'),
        ).asJsonObject.getAsJsonArray("sources")[0].asJsonObject
        assertEquals("7", sourceRecord.get("id").asString)
        assertEquals("https://www.who.int/", sourceRecord.get("url").asString)
        assertEquals("WHO report", sourceRecord.get("title").asString)
        val document = (result as GenUiConversionResult.Success).document
        assertTrue(document.express.contains("[7] WHO report"))
        assertTrue(document.express.contains("https://www.who.int/"))
    }

    @Test fun `public http source remains an exact host controlled link`() = runBlocking {
        val sourceUrl = "http://www.who.int/report?edition=2026."
        val stub = Stub(mutableListOf(program("Ready. [7]")))

        val result = GenUiConverter.withPrompt(stub, "contract").convert(
            GenUiRequest("Ready. [7]", sources = listOf(GenUiSource("7", sourceUrl, "WHO report")))
        )

        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        assertTrue((result as GenUiConversionResult.Success).document.express.contains(sourceUrl))
    }

    @Test fun `oversized query source metadata and unsupported source URLs fail before model execution`() = runBlocking {
        listOf(
            GenUiRequest("Ready", query="q".repeat(8001)),
            GenUiRequest("Ready", sources=listOf(GenUiSource("7", "tel:123"))),
            GenUiRequest("Ready", sources=listOf(GenUiSource("7", "https://example.com/report"))),
            GenUiRequest("Ready", sources=List(101) { GenUiSource("$it", "https://www.who.int/") }),
        ).forEach { request ->
            val stub = Stub(mutableListOf())
            assertTrue(GenUiConverter.withPrompt(stub, "contract").convert(request) is GenUiConversionResult.Failure)
            assertEquals(0, stub.calls)
        }
    }
}
