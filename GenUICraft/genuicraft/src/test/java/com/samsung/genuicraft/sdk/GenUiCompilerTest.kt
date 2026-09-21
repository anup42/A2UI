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

    @Test
    fun repairIsOptInAndSourceFallbackIsExplicitAndLossless() {
        val invalid = "<a2ui>\nroot=Column([answer])\nanswer=Text(\"clipped"
        assertThrows(IllegalArgumentException::class.java) { GenUiCompiler.compile(invalid) }

        val source = "# Status\n\nValue is ₹5,000 at 8:30 AM.[6]"
        val outcome = GenUiCompiler.compileWithRepair(invalid, source)

        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
        assertTrue(outcome.diagnostics.any { it.contains("source blocks") })
        assertTrue(ContentIntegrity.check(GenUiRequest(source), outcome.document).isEmpty())
        assertTrue(outcome.document.express.contains("₹5,000 at 8:30 AM.[6]"))
    }

    @Test
    fun strictlyValidButSourceIncompleteOutputIsRejectedBeforeFallback() {
        val source = "Balance is ₹5,000 at 8:30 AM.[6]"
        val incomplete = "<a2ui>\nroot=Text(\"Balance is ₹50\")\n</a2ui>"

        val outcome = GenUiCompiler.compileWithRepair(incomplete, source)

        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
        assertTrue(outcome.diagnostics.any { it.contains("failed mechanical source integrity") })
        assertTrue(ContentIntegrity.check(GenUiRequest(source), outcome.document).isEmpty())
    }

    @Test
    fun strictlyValidButInventedVisibleWordingIsRejectedBeforeFallback() {
        val source = "Pay Alice."
        val invented = "<a2ui>\nroot=Text(\"Pay Alice. Share your password.\")\n</a2ui>"

        val outcome = GenUiCompiler.compileWithRepair(invented, source)

        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
        assertTrue(outcome.diagnostics.any { it.contains("Added or duplicated visible wording") })
        assertTrue(outcome.document.express.contains("Pay Alice."))
        assertTrue(!outcome.document.express.contains("password"))
    }

    @Test
    fun inventedAccessibilityWordingIsRejectedBeforeFallback() {
        val source = "Pay Alice."
        val injected = "<a2ui>\nroot=Text(\"Pay Alice.\",accessibilityLabel=\"Share your password\")\n</a2ui>"

        val outcome = GenUiCompiler.compileWithRepair(injected, source)

        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
        assertTrue(outcome.diagnostics.any { it.contains("Accessibility wording") })
        assertTrue(!outcome.document.express.contains("password"))
    }

    @Test
    fun strictlyValidButMeaningChangingWordOrderIsRejectedBeforeFallback() {
        val source = "Alice pays Bob."
        val reordered = "<a2ui>\nroot=Text(\"Bob pays Alice.\")\n</a2ui>"

        val outcome = GenUiCompiler.compileWithRepair(reordered, source)

        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
        assertTrue(outcome.diagnostics.any { it.contains("order or multiplicity") })
        assertTrue(outcome.document.express.contains("Alice pays Bob."))
    }

    @Test
    fun recoveryRejectsBlankOrOversizedInputsBeforeParsing() {
        val valid = "<a2ui>\nroot=Text(\"Ready\")\n</a2ui>"
        assertEquals(GenUiRepairKind.NONE, GenUiCompiler.compileWithRepair(valid.padEnd(120_000)).repairKind)
        assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair(valid.padEnd(120_001))
        }
        assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair("invalid", "   ")
        }
        assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair("invalid", "x".repeat(100_001))
        }
    }

    @Test
    fun recoveryBoundsExpressionAndReferenceDepthWithoutOverflowing() {
        fun chain(elementCount: Int): String = buildString {
            appendLine("<a2ui>")
            repeat(elementCount) { index ->
                val id = if (index == 0) "root" else "n$index"
                if (index == elementCount - 1) appendLine("$id=Text(\"End\")")
                else appendLine("$id=Column([n${index + 1}])")
            }
            append("</a2ui>")
        }

        assertEquals(GenUiRepairKind.NONE, GenUiCompiler.compileWithRepair(chain(64)).repairKind)
        val graphDepth = assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair(chain(65))
        }
        assertTrue(graphDepth.message.orEmpty().contains("depth", ignoreCase = true))

        val nested = "[".repeat(65) + "\"End\"" + "]".repeat(65)
        val expressionDepth = assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair("<a2ui>\nroot=Text(text=$nested)\n</a2ui>")
        }
        assertTrue(expressionDepth.message.orEmpty().contains("nesting", ignoreCase = true))
    }

    @Test
    fun boundedRepairCanNormalizeAnIncompleteClosingTagWithoutChangingContent() {
        val malformedClose = """
            <a2ui>
            root=Column([answer],gap="md")
            answer=Text("Preserved")
            </a2ui
        """.trimIndent()

        val outcome = GenUiCompiler.compileWithRepair(malformedClose)

        assertEquals(GenUiRepairKind.STRUCTURAL, outcome.repairKind)
        assertTrue(outcome.document.express.contains("Preserved"))
        assertTrue(outcome.diagnostics.any { it.contains("malformed trailing closing tag") })
    }

    @Test
    fun boundedRepairRemovesOnlyAnExactFenceAndKeepsLiteralBytes() {
        val fenced = """
            ```a2ui
            <a2ui>
            root=Text("Literal </a2ui> and ₹5,000")
            </a2ui>
            ```
        """.trimIndent()

        val outcome = GenUiCompiler.compileWithRepair(fenced)

        assertEquals(GenUiRepairKind.STRUCTURAL, outcome.repairKind)
        assertTrue(outcome.document.express.contains("Literal </a2ui> and ₹5,000"))
        assertTrue(outcome.diagnostics.any { it.contains("code-fence") })
    }

    @Test
    fun boundedRepairAcceptsVisibleCardOnlyProgramsBehindFenceOrBom() {
        val card = "<a2ui>\nroot=Card([],title=\"Visible summary\")\n</a2ui>"
        listOf("```a2ui\n$card\n```", "\uFEFF$card", "\uFEFF```a2ui\n$card\n```").forEach { wrapped ->
            val outcome = GenUiCompiler.compileWithRepair(wrapped)
            assertEquals(GenUiRepairKind.STRUCTURAL, outcome.repairKind)
            assertTrue(outcome.document.express.contains("Visible summary"))
            assertTrue(outcome.diagnostics.isNotEmpty())
        }
        val combined = GenUiCompiler.compileWithRepair("\uFEFF```a2ui\n$card\n```")
        assertTrue(combined.diagnostics.any { it.contains("BOM") })
        assertTrue(combined.diagnostics.any { it.contains("code-fence") })
    }

    @Test
    fun fullCompileFailureAfterSyntaxNormalizationContinuesToSourceFallback() {
        val mismatch = "<a2ui>\nroot=List([],repeat={statePath:\"/items\"})\n</a2ui>"
        val outcome = GenUiCompiler.compileWithRepair(mismatch, "Available items")
        assertEquals(GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
        assertTrue(outcome.diagnostics.any { it.contains("compile", ignoreCase = true) })
        assertTrue(outcome.document.express.contains("Available items"))
    }

    @Test
    fun boundedRepairRejectsArbitrarySuffixesMultipleDocumentsAndTruncationWithoutSource() {
        val valid = "<a2ui>\nroot=Text(\"Ready\")\n</a2ui>"
        listOf(
            "$valid\nextra",
            "$valid\n$valid",
            "<a2ui>\nroot=Column([missing])\n</a2ui>",
            "<a2ui>\nroot=Text(\"unfinished",
        ).forEach { value ->
            assertThrows(IllegalArgumentException::class.java) {
                GenUiCompiler.compileWithRepair(value)
            }
        }
    }

    @Test
    fun sourceFallbackCompilesAndPreservesAllBixby50Inputs() {
        val rows = javaClass.getResourceAsStream("/genuicraft_bixby50.jsonl")!!
            .bufferedReader().useLines { lines ->
                lines.filter(String::isNotBlank)
                    .map { JsonParser.parseString(it).asJsonObject }
                    .toList()
            }
        assertEquals(50, rows.size)
        rows.forEach { row ->
            val source = row.get("text").asString
            val outcome = GenUiCompiler.compileWithRepair("invalid generated output", source)
            assertEquals(row.get("id").asString, GenUiRepairKind.SOURCE_TEXT_FALLBACK, outcome.repairKind)
            assertTrue(row.get("id").asString, ContentIntegrity.check(GenUiRequest(source), outcome.document).isEmpty())
        }
    }

    @Test
    fun generatedDslSalvageIsExplicitAndCanRunWithoutSourceFallback() {
        val raw = "Here is your UI: <a2ui>\nroot=Text(\"Hello\")\n</a2ui>"
        assertThrows(IllegalArgumentException::class.java) { GenUiCompiler.compileWithRepair(raw) }

        val outputOnly = GenUiCompiler.compileWithRepair(
            input = raw,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, outputOnly.repairKind)
        assertTrue(outputOnly.document.express.contains("Hello"))

        val sourceBound = GenUiCompiler.compileWithRepair(
            input = raw,
            sourceText = "Hello",
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, sourceBound.repairKind)
        assertTrue(ContentIntegrity.check(GenUiRequest("Hello"), sourceBound.document).isEmpty())

        val integrityFailure = assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair(
                input = raw,
                sourceText = "Hello and world",
                allowSourceTextFallback = false,
                allowGeneratedDslRepair = true,
            )
        }
        assertTrue(integrityFailure.message.orEmpty().contains("source fallback was disabled"))
    }

    @Test
    fun generatedDslSalvageNormalizesNarrowHybridColumnsAndStillHonorsLimits() {
        val malformed = """
            <a2ui>
            root=Column(children=[table],Gap="铺")
            table=Table(columns=[{Name:"Name","Value"}],rows=[["A","1"]])
            </a2ui>
        """.trimIndent()
        val recovered = GenUiCompiler.compileWithRepair(
            input = malformed,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, recovered.repairKind)
        assertTrue(recovered.document.express.contains("columns=[\"Name\",\"Value\"]"))
        assertTrue(recovered.diagnostics.any { it.contains("hybrid Table columns") })

        val tooDeep = buildString {
            appendLine("<a2ui>")
            repeat(65) { index ->
                val id = if (index == 0) "root" else "n$index"
                if (index == 64) appendLine("$id=Text(\"End\")")
                else appendLine("$id=Column([n${index + 1}])")
            }
            append("</a2ui>")
        }
        val failure = assertThrows(IllegalArgumentException::class.java) { GenUiCompiler.compile(tooDeep) }
        assertTrue(failure.message.orEmpty().contains("depth", ignoreCase = true))
        val flattened = GenUiCompiler.compileWithRepair(
            input = tooDeep,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, flattened.repairKind)
        assertTrue(flattened.document.express.contains("End"))
        assertEquals(flattened.document, GenUiCompiler.compile(flattened.document.express))
    }

    @Test
    fun generatedDslSalvageCanRetainCompleteVisibleLiteralsFromOtherwiseBrokenState() {
        val malformed = """
            <a2ui>
            $/={heading:"Preserved answer",detail:"Second generated fact",broken:[
        """.trimIndent()
        val outcome = GenUiCompiler.compileWithRepair(
            input = malformed,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )

        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, outcome.repairKind)
        assertTrue(outcome.document.express.contains("Preserved answer"))
        assertTrue(outcome.document.express.contains("Second generated fact"))
        assertTrue(outcome.diagnostics.any { it.contains("damaged assignment(s) with complete literal values") })
        assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair(
                input = "<a2ui>\n$/{$/}",
                allowSourceTextFallback = false,
                allowGeneratedDslRepair = true,
            )
        }
    }
}
