package com.samsung.genuicraft.sdk

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiDuplicateTableRecovery
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

/** Proposed tests for exact ordered slot projection; no source fallback or state rewriting. */
class GenUiDuplicateTableRecoveryTest {
    private val columns = "[{key:\"item\",label:\"Item\"},{key:\"item\",label:\"Evidence\"}]"
    private fun program(rows: String, cols: String = columns, tail: String = ""): String =
        "<a2ui>\nroot=Column([heading,table])\nheading=Text(\"Evidence\",\"h2\")\n" +
            "table=Table(columns=" + cols + ",rows=" + rows + tail + ")\n</a2ui>"

    private fun recover(raw: String): GenUiCompileOutcome = GenUiCompiler.compileWithRepair(
        input = raw, allowSourceTextFallback = false, allowGeneratedDslRepair = true,
    )

    private fun graph(outcome: GenUiCompileOutcome): JsonObject =
        GenUiIrCodec.decode(JsonParser.parseString(outcome.document.a2uiJson)).canonicalGraph

    private fun leaves(value: JsonElement): List<String> = when {
        value.isJsonPrimitive && value.asJsonPrimitive.isString -> listOf(value.asString)
        value.isJsonArray -> value.asJsonArray.flatMap(::leaves)
        value.isJsonObject -> value.asJsonObject.entrySet().flatMap { leaves(it.value) }
        else -> emptyList()
    }

    private fun reachableTables(graph: JsonObject): List<JsonObject> {
        val elements = graph.getAsJsonObject("elements")
        val visited = linkedSetOf<String>()
        val tables = mutableListOf<JsonObject>()
        fun visit(id: String) {
            if (!visited.add(id)) return
            val element = elements.getAsJsonObject(id) ?: return
            if (element.get("type")?.asString == "Table") tables += element
            element.getAsJsonArray("children")?.forEach { visit(it.asString) }
        }
        visit(graph.get("root").asString)
        return tables
    }

    private fun projectedTable(raw: String, id: String = "table"): JsonObject {
        val prepared = A2uiDuplicateTableRecovery.prepare(raw)
        assertTrue(prepared.changes.isNotEmpty())
        val line = prepared.input.lineSequence().first { it.startsWith(id + "=") }
        return A2uiExpressCodec.decode("<a2ui>\nroot=" + line.substringAfter('=') + "\n</a2ui>")
            .getAsJsonObject("elements").getAsJsonObject("root").getAsJsonObject("props")
    }

    @Test fun exact040FixturePreservesAllMissingEvidenceLiteralsAndLeavesStateUntouched() {
        val bytes = requireNotNull(javaClass.getResourceAsStream("/recovery/BXP-040.fp16.raw.express")).use { it.readBytes() }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        assertEquals("bc64a42aa629310c4e0c80bc84a1a3f03c77640fdfd0da8ddd7b1b416d0d7d37", digest)
        val raw = bytes.toString(Charsets.UTF_8)
        val strict = assertThrows(IllegalArgumentException::class.java) { A2uiExpressCodec.decode(raw) }
        assertTrue(strict.message.orEmpty().contains("Duplicate map key 'item'"))
        val prepared = A2uiDuplicateTableRecovery.prepare(raw)
        assertEquals(listOf("Projected complete ordered duplicate-key Table 'c' to its emitted column labels and exact positional cell literals; state was not edited."),
            prepared.changes)
        val rootLine = raw.lineSequence().first { it.startsWith("root=") }
        val cardLine = raw.lineSequence().first { it.startsWith("a=Card") }
        assertEquals("root=Column([a,e,k,q,w,ac],gap=\"md\",padding=16)", rootLine)
        assertEquals("a=Card([b,c])", cardLine)
        val rootContext = A2uiExpressCodec.decodeStatements(listOf(rootLine), requireRoot = false)
            .getAsJsonObject("elements").getAsJsonObject("root")
        assertEquals("md", rootContext.getAsJsonObject("props").get("gap").asString)
        assertEquals(listOf("a", "e", "k", "q", "w", "ac"), rootContext.getAsJsonArray("children").map { it.asString })
        val cardContext = A2uiExpressCodec.decodeStatements(listOf(cardLine), requireRoot = false)
            .getAsJsonObject("elements").getAsJsonObject("a")
        assertEquals(listOf("b", "c"), cardContext.getAsJsonArray("children").map { it.asString })
        assertEquals(raw.lineSequence().first { it.startsWith("$/=") },
            prepared.input.lineSequence().first { it.startsWith("$/=") })
        assertEquals(raw.lineSequence().first { it.startsWith("root=") },
            prepared.input.lineSequence().first { it.startsWith("root=") })
        val props = projectedTable(raw, "c")
        assertEquals(listOf("Item", "Evidence"), props.getAsJsonArray("columns").map { it.asString })
        val rows = props.getAsJsonArray("rows")
        assertEquals(10, rows.size())
        assertEquals(listOf("job sheet or written service report", "if inspected or repaired"),
            rows.last().asJsonArray.map { it.asString })
        rows.take(9).forEach { row -> assertEquals(row.asJsonArray[0], row.asJsonArray[1]) }
        val outcome = recover(raw)
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, outcome.repairKind)
        val recoveredGraph = graph(outcome)
        val recoveredEvidence = reachableTables(recoveredGraph).filter { element ->
            element.getAsJsonObject("props").getAsJsonArray("columns") == props.getAsJsonArray("columns")
        }
        assertEquals("Exactly one reachable direct-row Item/Evidence table", 1, recoveredEvidence.size)
        val recoveredRows = recoveredEvidence.single().getAsJsonObject("props").getAsJsonArray("rows")
        assertEquals("All twenty literal cell occurrences preserve their exact ordered associations", rows, recoveredRows)
        val evidenceValues = leaves(recoveredRows)
        listOf("order confirmation", "invoice/bill", "product listing screenshots", "photos or videos",
            "defect", "unboxing", "emails, chat logs, complaint ticket numbers, pickup receipts, and return/refund status updates",
            "job sheet or written service report").forEach { value -> assertTrue(value, value in evidenceValues) }
        val values = leaves(recoveredGraph)
        assertFalse("Additional recovered text" in values)
        assertFalse(Regex("\\[[0-9]+]").containsMatchIn(values.joinToString(" ")))
        assertEquals(outcome.document, GenUiCompiler.compile(outcome.document.express))
    }

    @Test fun explicitColumnSlotsPreserveUnequalLiteralsWithoutInventingFieldKeys() {
        val raw = program("[{item:\"Report\",item:\"Only if inspected\"}]",
            tail = ",domain=\"generic\",preferredPresentation=\"cards\"")
        val props = projectedTable(raw)
        assertEquals(listOf("Item", "Evidence"), props.getAsJsonArray("columns").map { it.asString })
        assertEquals(listOf("Report", "Only if inspected"), props.getAsJsonArray("rows").single().asJsonArray.map { it.asString })
        assertEquals("generic", props.get("domain").asString)
        assertEquals("cards", props.get("preferredPresentation").asString)
    }

    @Test fun repeatedEqualCellsAndRepeatedRowsRemainRepeated() {
        val raw = program("[{item:\"Same\",item:\"Same\"},{item:\"Same\",item:\"Same\"}]")
        val rows = projectedTable(raw).getAsJsonArray("rows")
        assertEquals(2, rows.size())
        assertEquals(rows[0], rows[1])
        assertEquals(listOf("Same", "Same"), rows[0].asJsonArray.map { it.asString })
    }

    @Test fun columnUnitsLabelsAndKnownRootPlacementArePreserved() {
        val raw = program("[{item:\"13\",item:\"27\"}]",
            cols = "[{key:\"item\",label:\"Value (USD)\"},{key:\"item\",label:\"Value (INR)\"}]")
        val prepared = A2uiDuplicateTableRecovery.prepare(raw)
        val decoded = A2uiExpressCodec.decode(prepared.input)
        assertEquals(listOf("heading", "table"), decoded.getAsJsonObject("elements")
            .getAsJsonObject("root").getAsJsonArray("children").map { it.asString })
        assertTrue(decoded.getAsJsonObject("elements").has("table"))
        assertEquals(listOf("Value (USD)", "Value (INR)"),
            projectedTable(raw).getAsJsonArray("columns").map { it.asString })
        assertEquals(prepared, A2uiDuplicateTableRecovery.prepare(raw))
    }

    @Test fun unequalOccurrenceCountsOrDifferentKeyOrderAreNotGuessed() {
        val ambiguous = listOf(
            program("[{item:\"Only one\"}]"),
            program("[{item:\"First\",item:\"Second\",item:\"Third\"}]"),
            program("[{item:\"First\",other:\"Second\"}]"),
            program("[{item:\"First\",item:\"Second\"},{item:\"Third\"}]"),
            program("[{a:\"One\",a:\"Two\",b:\"Three\"}]",
                cols = "[{key:\"a\",label:\"A1\"},{key:\"b\",label:\"B\"},{key:\"a\",label:\"A2\"}]"),
        )
        ambiguous.forEach { raw -> assertEquals(raw, A2uiDuplicateTableRecovery.prepare(raw).input) }
    }

    @Test fun duplicateOrMissingLabelsAndIndirectColumnMetadataAreNotReinterpreted() {
        val cases = listOf(
            program("[{item:\"A\",item:\"B\"}]", cols = "[{key:\"item\",label:\"Same\"},{key:\"item\",label:\"Same\"}]"),
            program("[{item:\"A\",item:\"B\"}]", cols = "[{key:\"item\"},{key:\"item\",label:\"Evidence\"}]"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",primaryColumn=\"item\""),
            program("[{item:\"A\",item:\"B\"}]", tail = ",highlightColumns=[\"item\"]"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",statePath=\"/data\""),
        )
        cases.forEach { raw -> assertTrue(A2uiDuplicateTableRecovery.prepare(raw).changes.isEmpty()) }
    }

    @Test fun boundNestedActionGuardRepeatAndUnknownCellsOrMetadataAreRejected() {
        val cases = listOf(
            program("[{item:\"\$/secret\",item:\"B\"}]"),
            program("[{item:\"\$item.name\",item:\"B\"}]"),
            program("[{item:\"@source.a\",item:\"B\"}]"),
            program("[{item:{text:\"Nested\"},item:\"B\"}]"),
            program("[{item:[\"Nested\"],item:\"B\"}]"),
            program("[{item:openUrl(\"https://example.invalid/\"),item:\"B\"}]"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",onPress=openUrl(\"https://example.invalid/\")"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",visible=false"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",repeat={statePath:\"/data\",template:\"table\"}"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",unknown=\"metadata\""),
        )
        cases.forEach { raw -> assertEquals(raw, A2uiDuplicateTableRecovery.prepare(raw).input) }
    }

    @Test fun guardedOpaqueModalTabAndOrphanPlacementDoesNotPromoteATable() {
        val call = "table=Table(columns=" + columns + ",rows=[{item:\"A\",item:\"B\"}])"
        listOf(
            "root=Column([guard])\nguard=Card([table],visible=false)",
            "root=Column([guard])\nguard=Column([table],repeat={statePath:\"/rows\",template:\"table\"})",
            "root=Column([guard])\nguard=Card([table],visible:false)",
            "root=Modal(trigger=button,content=table)\nbutton=Button(\"Open\")",
            "root=Tabs(tabs=[{title:\"First\",child:\"table\"}])",
            "root=Text(\"Other content\")",
        ).forEach { parents ->
            val raw = "<a2ui>\n" + parents + "\n" + call + "\n</a2ui>"
            assertEquals(raw, A2uiDuplicateTableRecovery.prepare(raw).input)
        }
    }

    @Test fun conflictingIncompleteOpaqueAndNonCallIdentitiesCannotEstablishStaticParents() {
        val raw = program("[{item:\"A\",item:\"B\"}]")
        val parentRaw = raw.replace("root=Column([heading,table])",
            "root=Column([heading,parent])\nparent=Card([table])")
        val cases = listOf(
            raw.replace("</a2ui>", "root=Card([table],visible=false) unknown=\"opaque tail\"\n</a2ui>"),
            raw.replace("</a2ui>", "root=Card([table],visible=false\n</a2ui>"),
            raw.replace("</a2ui>", "root={visible:false}\n</a2ui>"),
            raw.replace("</a2ui>", "root=\"Opaque root\"\n</a2ui>"),
            raw.replace("</a2ui>", "broken=)\nroot=Card([table],visible=false)\n</a2ui>"),
            raw.replace("</a2ui>", "/* prefix */root=Card([table],visible=false)\n</a2ui>"),
            raw.replace("</a2ui>", "table=Table(columns=" + columns +
                ",rows=[{item:\"A\",item:\"B\"}],visible=false) unknown=\"opaque tail\"\n</a2ui>"),
            raw.replace("</a2ui>", "table={visible:false}\n</a2ui>"),
            parentRaw.replace("</a2ui>", "parent=Card([table],visible=false\n</a2ui>"),
            parentRaw.replace("</a2ui>", "parent={visible:false}\n</a2ui>"),
            parentRaw.replace("</a2ui>", "parent=Card([table],visible=false) opaque=\"tail\"\n</a2ui>"),
        )
        cases.forEach { input -> assertEquals(input, A2uiDuplicateTableRecovery.prepare(input).input) }
    }

    @Test fun semicolonIdentityConflictsAreRejectedWhileQuotedAndCommentedIdentitiesAreIgnored() {
        val raw = program("[{item:\"A\",item:\"B\"}]")
        val parentRaw = raw.replace("root=Column([heading,table])",
            "root=Column([heading,parent]);parent=Card([table])")
        val cases = listOf(
            raw.replace("root=Column([heading,table])", "root=Column([heading,table]);root=Card([table],visible=false)"),
            raw.replace("</a2ui>", ";table=Table(columns=" + columns +
                ",rows=[{item:\"C\",item:\"D\"}],visible=false)\n</a2ui>"),
            parentRaw.replace("</a2ui>", ";parent=Card([table],visible=false)\n</a2ui>"),
        )
        cases.forEach { input -> assertEquals(input, A2uiDuplicateTableRecovery.prepare(input).input) }
        val oneLine = raw.replace("\n", ";")
        val prepared = A2uiDuplicateTableRecovery.prepare(oneLine)
        assertEquals(1, prepared.changes.size)
        val table = A2uiExpressCodec.decode(prepared.input).getAsJsonObject("elements").getAsJsonObject("table")
        assertEquals(listOf("A", "B"), table.getAsJsonObject("props").getAsJsonArray("rows").single()
            .asJsonArray.map { it.asString })
        val text = "prefix;root=Card([table],visible=false);parent={visible:false}"
        val quoted = program("[{item:" + JsonPrimitive(text) + ",item:\"B\"}]")
            .replace("heading=Text", "// root=Card([table],visible=false)\n/* table={visible:false} */\nheading=Text")
        assertEquals(text, projectedTable(quoted).getAsJsonArray("rows").single().asJsonArray[0].asString)
    }

    @Test fun everyReservedDollarExpressionKeyInParentPropsBlocksProjectionRecursively() {
        val raw = program("[{item:\"A\",item:\"B\"}]")
        listOf("\$state", "\$bindState", "\$item", "\$bindItem", "\$index", "\$cond", "\$template",
            "\$computed", "\$futureExpression").forEach { key ->
            val expression = "{" + JsonPrimitive(key) + ":\"ordinaryFieldName\"}"
            listOf(expression, "{nested:[" + expression + "]}").forEach { value ->
                val parent = raw.replace("root=Column([heading,table])",
                    "root=Column([heading,parent])\nparent=Card([table],title=" + value + ")")
                assertEquals(parent, A2uiDuplicateTableRecovery.prepare(parent).input)
                val boundRoot = raw.replace("root=Column([heading,table])",
                    "root=Column([heading,table],padding=" + value + ")")
                assertEquals(boundRoot, A2uiDuplicateTableRecovery.prepare(boundRoot).input)
            }
        }
    }

    @Test fun flatBindingCellsLabelsMetadataAndParentsAreNotLiteralized() {
        val cases = listOf(
            program("[{item:\"{{/secret}}\",item:\"B\"}]"),
            program("[{item:\"prefix {{/secret}} suffix\",item:\"B\"}]"),
            program("[{item:\"prefix \${/secret} suffix\",item:\"B\"}]"),
            program("[{item:\"prefix {\$item.name} suffix\",item:\"B\"}]"),
            program("[{item:\"A\",item:\"B\"}]", tail = ",title=\"{{/heading}}\""),
            program("[{item:\"A\",item:\"B\"}]", tail = ",domain=\"prefix {{/domain}}\""),
            program("[{item:\"A\",item:\"B\"}]", cols = "[{key:\"item\",label:\"{{/label}}\"},{key:\"item\",label:\"Evidence\"}]"),
            program("[{item:\"A\",item:\"B\"}]").replace("root=Column([heading,table])", "root=Column([heading,parent])\nparent=Card([table],title=\"{{/entity}}\")"),
            program("[{item:\"A\",item:\"B\"}]").replace("root=Column([heading,table])", "root=Column([heading,parent])\nparent=Card([table],title=\"prefix {\$item.name}\")"),
        )
        cases.forEach { raw -> assertEquals(raw, A2uiDuplicateTableRecovery.prepare(raw).input) }
    }

    @Test fun quotedGapParentsReachTheTableWhileBareGapParentsRemainOpaque() {
        listOf("md", "sm").forEach { token ->
            val quoted = program("[{item:\"A\",item:\"B\"}]")
                .replace("root=Column([heading,table])", "root=Column([heading,parent],gap=\"" + token + "\")\nparent=Card([table])")
            assertTrue(A2uiDuplicateTableRecovery.prepare(quoted).changes.isNotEmpty())
            val bare = quoted.replace("gap=\"" + token + "\"", "gap=" + token)
            assertEquals(bare, A2uiDuplicateTableRecovery.prepare(bare).input)
        }
    }

    @Test fun quotedHeadersCommentsAndRawTripleStringsCannotCreateTableDefinitions() {
        val message = "first line\nfake=Table(columns=[])\nlast line"
        val rows = "[{item:" + JsonPrimitive(message) + ",item:\"Exact second cell\"}]"
        val raw = program(rows).replace("heading=Text", "// fake=Table(columns=[],rows=[])\nheading=Text")
        val props = projectedTable(raw)
        assertEquals(message, props.getAsJsonArray("rows").single().asJsonArray[0].asString)
        val rawLiteral = "r\"\"\"C:\\data\\report\nfake=Table()\n</a2ui>\"\"\""
        val rawProgram = program("[{item:" + rawLiteral + ",item:\"B\"}]")
        val expected = A2uiExpressCodec.literalPrefix(rawLiteral).first.asString
        assertEquals(expected, projectedTable(rawProgram).getAsJsonArray("rows").single().asJsonArray[0].asString)
    }

    @Test fun incompleteValuesWrongClosersAndUnparsedTailsRemainUnchanged() {
        val raw = program("[{item:\"A\",item:\"B\"}]")
        listOf(
            raw.replace("item:\"B\"", "item:\"Never closed"),
            raw.replace("rows=[{item:\"A\",item:\"B\"}]", "rows=[{item:\"A\",item:\"B\"]}"),
            raw.replace("}])", "}]}"),
            raw.replace("}])", "}]) unknown=\"metadata\""),
            raw.replace("}])", "}]) /* comment */ unknown=\"metadata\""),
        ).forEach { value ->
            assertEquals(value, A2uiDuplicateTableRecovery.prepare(value).input)
        }
    }

    @Test fun ordinaryUniqueKeyTablesAndGlobalDuplicateStateRemainUnchanged() {
        val ordinary = program("[{item:\"A\",evidence:\"B\"}]",
            cols = "[{key:\"item\",label:\"Item\"},{key:\"evidence\",label:\"Evidence\"}]")
        assertEquals(ordinary, A2uiDuplicateTableRecovery.prepare(ordinary).input)
        val stateOnly = "<a2ui>\n$/={prices:{price:\"199\",price:\"299\"}}\nroot=Text(\"Static answer\")\n</a2ui>"
        assertEquals(stateOnly, A2uiDuplicateTableRecovery.prepare(stateOnly).input)
    }

    @Test fun elementInputDepthAndReaderBudgetsDoNotAuthorizeBroaderRecovery() {
        val raw = program("[{item:\"A\",item:\"B\"}]")
        val tooLarge = raw + " ".repeat(A2uiExpressCodec.MAX_INPUT_CHARS)
        assertEquals(tooLarge, A2uiDuplicateTableRecovery.prepare(tooLarge).input)
        val tooDeep = program("[{item:" + "[".repeat(65) + "\"X\"" + "]".repeat(65) + ",item:\"B\"}]")
        assertEquals(tooDeep, A2uiDuplicateTableRecovery.prepare(tooDeep).input)
        val tooManyElements = raw.replace("</a2ui>", List(1_025) { index -> "extra" + index + "=Text(\"Other\")" }
            .joinToString("\n") + "\n</a2ui>")
        assertEquals(tooManyElements, A2uiDuplicateTableRecovery.prepare(tooManyElements).input)
        val tooManyRows = program("[" + List(2_100) { "{item:\"A\",item:\"B\"}" }.joinToString(",") + "]")
        assertEquals(tooManyRows, A2uiDuplicateTableRecovery.prepare(tooManyRows).input)
    }

    @Test fun existingMalformedTableFragmentSuppressionIsNotReversed() {
        val raw = requireNotNull(javaClass.getResourceAsStream("/recovery/BXP-003.raw.express"))
            .bufferedReader(Charsets.UTF_8).use { it.readText() }
        assertEquals(raw, A2uiDuplicateTableRecovery.prepare(raw).input)
        val values = leaves(graph(recover(raw)))
        assertFalse("Shatabdi Express also shows slight variation in duration, from about 2 h 15 m to 2 h 25 m." in values)
        assertFalse("Additional recovered text" in values)
    }
}
