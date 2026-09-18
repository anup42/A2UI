package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.renderer.GenUiNativeRenderer
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.FlatActionRuntime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiCompilerTest {
    private val express = """
        <a2ui>
        root=Column([title,details,formula,code,action],gap="md")
        title=Text("Order A1042","h2")
        details=Table(["Detail","Value"],rows=[["Status","Shipped"],["ETA","Friday"]],title="Order",domain="status",preferredPresentation="table")
        formula=Formula("E=mc^2","Formula","42",true)
        code=CodeBlock("Use ``` fences","text","Example")
        action=Button("Track","primary",onPress=openUrl("https://www.samsung.com/support/"))
        </a2ui>
    """.trimIndent()

    @Test
    fun compileExpressProducesBothStrictRepresentationsAndKeepsCatalogElements() {
        val document = GenUiCompiler.compile(express)

        assertTrue(document.express.startsWith("<a2ui>"))
        assertTrue(document.express.endsWith("</a2ui>"))
        assertTrue(JsonParser.parseString(document.a2uiJson).isJsonArray)
        assertEquals("genuicraft_express_v1", document.profile)
        assertEquals("v0.9", document.schemaVersion)

        val rendered = GenUiNativeRenderer.render(document.a2uiJson, sourceDir = null)
        assertNull(rendered.errorMessage)
        val spec = rendered.surfaces.single().canonicalSpec
        assertNotNull(spec)
        val types = spec!!.elements.values.map { it.type }.toSet()
        assertTrue(types.containsAll(setOf("Stack", "Text", "Table", "Formula", "CodeBlock", "Button")))
    }

    @Test
    fun compileWireIsDeterministicAcrossRepeatedCompilation() {
        val first = GenUiCompiler.compile(express)
        val second = GenUiCompiler.compile(first.a2uiJson)
        val third = GenUiCompiler.compile(first.a2uiJson)

        assertEquals(second, third)
        assertEquals(first.express, second.express)
        assertEquals(first.a2uiJson, second.a2uiJson)
    }

    @Test
    fun wireReplayKeepsOpenUrlExecutableForTheHost() {
        val replay = GenUiCompiler.compile(GenUiCompiler.compile(express).a2uiJson)
        val spec = GenUiNativeRenderer.render(replay.a2uiJson, sourceDir = null)
            .surfaces
            .single()
            .canonicalSpec!!
        val button = spec.elements.values.single { it.type == "Button" }
        var openedUrl: String? = null

        val result =
            FlatActionRuntime.executeDetailed(
                actionCandidate = button.on?.get("press"),
                stateStore = spec.state.toMutableMap(),
                repeatScope = null,
                computedFunctions = emptyMap(),
                onOpenUrl = { openedUrl = it },
            )

        assertEquals(1, result.attempted)
        assertTrue(result.actions.single().success)
        assertEquals("https://www.samsung.com/support/", openedUrl)
    }

    @Test
    fun strictBoundaryRejectsMarkdownProseLegacyAndTrailingDocuments() {
        val invalidInputs = listOf(
            "```a2ui\n$express\n```",
            "Here is the UI:\n$express",
            """{"root":"root","state":{},"elements":{"root":{"type":"Text","props":{"text":"legacy"},"children":[]}}}""",
            "$express\nadditional prose",
            "$express\n$express",
        )

        invalidInputs.forEach { invalid ->
            val error = assertThrows(IllegalArgumentException::class.java) {
                GenUiCompiler.compile(invalid)
            }
            assertTrue(error.message.orEmpty().isNotBlank())
        }
    }

    @Test
    fun blankInputReportsAnActionableError() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compile("   ")
        }
        assertTrue(error.message.orEmpty().contains("empty", ignoreCase = true))
    }
}
