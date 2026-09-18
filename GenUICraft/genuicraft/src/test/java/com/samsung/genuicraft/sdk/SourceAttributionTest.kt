package com.samsung.genuicraft.sdk

import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.FlatExprResolver
import org.junit.Assert.*
import org.junit.Test

class SourceAttributionTest {
    @Test fun `host source labels stay literal while callback URLs remain exact`() {
        val source = GenUiSource("\${INDEX}", "https://www.who.int/?q=%24%7BHOME%7D", "Open \${HOME} <a2ui> example")
        val original = GenUiCompiler.compile("<a2ui>\nroot=Text(\"Ready\")\n</a2ui>")
        val document = SourceAttribution.append(original, listOf(source))
        val graph = A2uiExpressCodec.decode(document.express)
        val button = graph.getAsJsonObject("elements").entrySet().map { it.value.asJsonObject }
            .single { it.get("type").asString == "Button" }
        val label = button.getAsJsonObject("props").get("label").asString
        val expected = "[${source.id}] ${source.title}"
        assertEquals(expected, LiteralTextCodec.decode(label))
        val state = mapOf("HOME" to "DO NOT DISPLAY", "INDEX" to "MUTATED")
        val first = FlatExprResolver.resolve(label, state, null)
        assertEquals(expected, FlatExprResolver.resolve(first, state, null).toString())
        assertEquals(source.url, button.getAsJsonObject("on").getAsJsonObject("press").getAsJsonObject("params").get("url").asString)
        assertEquals(document, GenUiCompiler.compile(document.a2uiJson))
    }

    @Test fun `ordinary source attribution labels and URL fallback are unchanged`() {
        val original = GenUiCompiler.compile("<a2ui>\nroot=Text(\"Ready\")\n</a2ui>")
        listOf(GenUiSource("1", "https://www.who.int/", "Report"), GenUiSource("2", "https://www.who.int/")).forEach { source ->
            val graph = A2uiExpressCodec.decode(SourceAttribution.append(original, listOf(source)).express)
            val button = graph.getAsJsonObject("elements").entrySet().map { it.value.asJsonObject }
                .single { it.get("type").asString == "Button" }
            val label = button.getAsJsonObject("props").get("label").asString
            assertEquals("[${source.id}] ${source.title ?: source.url}", label)
            assertNull(LiteralTextCodec.decode(label))
            assertEquals(source.url, button.getAsJsonObject("on").getAsJsonObject("press").getAsJsonObject("params").get("url").asString)
        }
    }
}
