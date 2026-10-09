package com.samsung.genuicraft.sdk

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiDeclaredTextRecovery
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

/** Generated-output recovery regressions. No native inference or source-text fallback. */
class GenUiDeclaredTextRecoveryTest {
    private fun prepared(raw: String, calls: Int = 64, chars: Int = 20_000, length: Int = 1_500) =
        A2uiDeclaredTextRecovery.prepare(raw, calls, chars, length)

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

    private fun reachableTexts(graph: JsonObject): List<String> {
        val result = mutableListOf<String>()
        val elements = graph.getAsJsonObject("elements")
        val active = mutableSetOf<String>()
        fun visit(id: String) {
            if (!active.add(id)) return
            val element = elements.getAsJsonObject(id) ?: return
            if (element.get("type")?.asString == "Text") result += element.getAsJsonObject("props").get("text").asString
            element.getAsJsonArray("children")?.forEach { visit(it.asString) }
            active -= id
        }
        visit(graph.get("root").asString)
        return result
    }

    @Test fun capturedCompleted046RetainsEmittedBodyAndCitationLiteralsWithoutAddingMissingSourceFacts() {
        val bytes = requireNotNull(javaClass.getResourceAsStream("/recovery/BXP-046.fp16.raw.express")).use { it.readBytes() }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        assertEquals("5384e77c986b78408d762bc49590cdf876eff03fa655581407a8a5c0b8fa7b6a", digest)
        val raw = bytes.toString(Charsets.UTF_8)
        val result = recover(raw)
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, result.repairKind)
        val output = graph(result)
        val values = reachableTexts(output)
        val ids = listOf("o", "p", "q", "w", "x", "y", "ac", "ad", "ae", "ai", "aj", "ak")
        val expected = ids.map { id ->
            val start = Regex("\\b" + id + "\\s*=\\s*Text\\(").find(raw)!!.range.last + 1
            val (literal, consumed) = A2uiExpressCodec.literalPrefix(raw.substring(start))
            val source = Regex("^\\s*,\\s*source\\s*:").find(raw.substring(start + consumed))!!
            val marker = A2uiExpressCodec.literalPrefix(raw.substring(start + consumed + source.range.last + 1)).first.asString
            id to (literal.asString to marker)
        }
        assertEquals(12, expected.size)
        assertEquals(1_971, expected.sumOf { it.second.first.length })
        var previous = -1
        expected.forEach { (id, pair) ->
            val (literal, marker) = pair
            val index = values.indexOf(literal)
            assertTrue("Declared paragraph is not reachable in emitted order: " + id, index > previous)
            assertEquals("Repeated or lost declared paragraph: " + id, 1, values.count { it == literal })
            if (marker.isNotEmpty()) {
                assertEquals("Citation lost its paragraph association: " + id, marker, values.getOrNull(index + 1))
            }
            previous = index
        }
        val expectedMarkers = setOf("[1]", "[3]", "[6]", "[7]", "[8]", "[10]", "[12]", "[13]")
        assertEquals(expectedMarkers, expected.flatMap { (_, pair) ->
            Regex("\\[[0-9]+]").findAll(pair.second).map { it.value }.toList()
        }.toSet())
        assertTrue(expectedMarkers.all { marker -> values.any { marker in it } })
        assertTrue(listOf("[2]", "[14]").all { marker -> leaves(output).any { marker in it } })
        assertTrue(leaves(output).any { it.contains("immunogenic toxicity") })
        assertTrue(leaves(output).any { it.contains("Cas nuclease") })
        assertFalse(leaves(output).any { it.contains("not interchangeable") })
        assertFalse("Additional recovered text" in values)
        assertFalse(result.repairKind == GenUiRepairKind.SOURCE_TEXT_FALLBACK)
        assertEquals(result.document, GenUiCompiler.compile(result.document.express))
    }

    @Test fun wrongOuterClosersRetainParagraphAndExactCitation() {
        listOf("}", "]", ")").forEach { closer ->
            val raw = "<a2ui>\nroot=Column([heading,detail])\nheading=Text(\"Clinical examples\",\"h2\")\n" +
                "detail=Text(\"Complete emitted paragraph\",source:\"[3][10]\"" + closer + "\n</a2ui>"
            val output = reachableTexts(graph(recover(raw)))
            assertEquals(listOf("Clinical examples", "Complete emitted paragraph", "[3][10]"), output)
        }
    }

    @Test fun sameLineAssignmentsBecomeSeparateDeclaredCallsIncludingTheFollowingTable() {
        val raw = """<a2ui>
root=Column([first,second,table])
first=Text("First exact paragraph",source:"[3]"},second=Text("Second exact paragraph",source:"[10]"],table=Table(columns=["K"],rows=[["V"]])
</a2ui>"""
        val transformed = prepared(raw)
        val decoded = A2uiExpressCodec.decode(transformed.input)
        assertEquals(listOf("first", "second", "table"),
            decoded.getAsJsonObject("elements").getAsJsonObject("root").getAsJsonArray("children").map { it.asString })
        val values = leaves(graph(recover(raw)))
        assertTrue("First exact paragraph" in values)
        assertTrue("Second exact paragraph" in values)
        assertTrue("V" in values)
        assertFalse("Additional recovered text" in values)
    }

    @Test fun stableDeclaredIdsCollisionSafeSyntheticIdsAndKnownRootOrderArePreserved() {
        val raw = """<a2ui>
root=Column([later,heading,earlier])
earlier=Text("Earlier definition",source:"[1]"})
later=Text("Later definition",source:"[2]"})
heading=Text("Middle heading","h2")
later_recovered_body=Text("Existing collision")
</a2ui>""".replace("})", "}")
        val first = prepared(raw)
        val second = prepared(raw)
        assertEquals(first, second)
        val decoded = A2uiExpressCodec.decode(first.input)
        val elements = decoded.getAsJsonObject("elements")
        assertTrue(elements.has("later") && elements.has("earlier"))
        assertTrue(elements.has("later_recovered_body"))
        assertTrue(elements.has("later_recovered_body_1"))
        assertEquals(listOf("later", "heading", "earlier"), elements.getAsJsonObject("root")
            .getAsJsonArray("children").map { it.asString })
        assertEquals(listOf("Later definition", "[2]", "Middle heading", "Earlier definition", "[1]"),
            reachableTexts(decoded))
    }

    @Test fun malformedActionConditionRepeatBindingAndUnknownTailsNeverBecomeUnconditionalText() {
        val tails = listOf(
            "onPress=openUrl(\"https://example.invalid/\")", "visible=false", "visible={check:\"unsafe\"}",
            "repeat={statePath:\"/rows\",template:\"bad\"}", "watch={\"/rows\":setState(\"/x\",1)}",
            "statePath=\"/rows\"", "unknown=\"metadata\"",
        )
        tails.forEach { tail ->
            val raw = "<a2ui>\nroot=Column([safe,bad])\nsafe=Text(\"Visible answer\")\n" +
                "bad=Text(\"Must remain dormant\",source:\"[1]\"," + tail + "}\n</a2ui>"
            val transformed = prepared(raw)
            assertFalse(tail, transformed.allowLiteralOnlyFallback)
            assertFalse(tail, "Must remain dormant" in leaves(graph(recover(raw))))
            assertTrue("Visible answer" in leaves(graph(recover(raw))))
        }
    }

    @Test fun unsafeOnlyGenerationDoesNotReenterGlobalLiteralRecovery() {
        val raw = """<a2ui>
root=Column([bad])
bad=Text("Must not become a body",source:"[1]",repeat={statePath:"/rows",template:"bad"})
</a2ui>"""
        assertFalse(prepared(raw).allowLiteralOnlyFallback)
        assertThrows(IllegalArgumentException::class.java) { recover(raw) }
    }

    @Test fun malformedBodyBindingsAndSourceBindingsAreNotLiteralized() {
        listOf("\"\$/secret\"", "\"\$item.name\"", "\"@source.a\"", "{\"\$state\":\"/secret\"}").forEach { expression ->
            val raw = "<a2ui>\nroot=Column([safe,bad])\nsafe=Text(\"Visible answer\")\nbad=Text(" +
                expression + ",source:\"[1]\"}\n</a2ui>"
            assertFalse(prepared(raw).allowLiteralOnlyFallback)
            assertFalse(expression, leaves(graph(recover(raw))).any { it in setOf("\$/secret", "\$item.name", "@source.a") })
        }
    }

    @Test fun guardedOrOpaqueParentsDoNotPromoteMalformedChildren() {
        listOf(
            "hidden=Column([bad],visible=false)",
            "hidden=Column([bad],repeat={statePath:\"/rows\",template:\"bad\"})",
            "hidden=Column([bad],onPress=openUrl(\"https://example.invalid/\"))",
            "hidden=Column([bad],visible:false)",
            "hidden=Modal(trigger=button,content=bad)",
            "hidden=Tabs(tabs=[{title:\"Hidden tab\",child:\"bad\"}])",
        ).forEach { parent ->
            val raw = "<a2ui>\nroot=Column([safe,hidden])\nsafe=Text(\"Visible answer\")\n" + parent +
                "\nbutton=Button(\"Open\")\nbad=Text(\"Dormant child\",source:\"[1]\"}\n</a2ui>"
            assertFalse(parent, prepared(raw).allowLiteralOnlyFallback)
            assertFalse(parent, "Dormant child" in leaves(graph(recover(raw))))
        }
    }

    @Test fun everyInlineGuardedAncestorKeepsItsMalformedChildOutOfCombinedSalvage() {
        val parents = listOf(
            "Column([bad],visible=false)",
            "Card([Column([bad],visible=false)])",
            "Column([bad],repeat={statePath:\"/rows\",template:\"bad\"})",
            "Column([bad],onPress=openUrl(\"https://example.invalid/\"))",
            "Column([bad],watch={\"/rows\":setState(\"/x\",1)})",
            "Modal(trigger=button,content=bad)",
            "Tabs(tabs=[{title:\"Dormant tab\",child:bad}])",
            "Card([Tabs(tabs=[{title:\"Dormant tab\",content:bad}])])",
        )
        parents.forEach { parent ->
            val root = "root=Column([safe," + parent + "])"
            // Confirm these are real codec-supported inline nodes, not an opaque-parent fallback.
            assertTrue(parent, A2uiExpressCodec.decodeStatements(listOf(root), requireRoot = false)
                .getAsJsonObject("elements").size() > 1)
            val raw = "<a2ui>\n" + root + "\nsafe=Text(\"Visible answer\")\nbutton=Button(\"Open\")\n" +
                "bad=Text(\"Dormant inline child\",source:\"[3]\"}\ndamaged=UnknownCall()\n</a2ui>"
            val transformed = prepared(raw)
            assertFalse(parent, transformed.allowLiteralOnlyFallback)
            assertFalse(parent, "Dormant inline child" in transformed.input)
            assertTrue(parent, runCatching { A2uiExpressCodec.decode(transformed.input) }.isFailure)
            val values = reachableTexts(graph(recover(raw)))
            assertTrue(parent, "Visible answer" in values)
            assertFalse(parent, "Dormant inline child" in values)
            assertFalse(parent, "[3]" in values)
        }
    }

    @Test fun passiveInlineContainersAndIndependentRootReferencesRemainEligibleDuringSalvage() {
        listOf(
            "root=Column([safe,Card([Column([body],gap=\"sm\")])])",
            "root=Column([safe,body,Column([body],visible=false)])",
        ).forEach { root ->
            val raw = "<a2ui>\n" + root + "\nsafe=Text(\"Visible answer\")\n" +
                "body=Text(\"Independent exact paragraph\",source:\"[3][10]\"}\ndamaged=UnknownCall()\n</a2ui>"
            assertTrue(prepared(raw).allowLiteralOnlyFallback)
            assertEquals(listOf("Visible answer", "Independent exact paragraph", "[3][10]"),
                reachableTexts(graph(recover(raw))))
        }
    }

    @Test fun supportedRuntimeTemplatesInBodiesAndVariantsAreNeverPromoted() {
        val templates = listOf(
            "\${/secret}", "Value \${/secret}", "\${\$item.name}", "Value {{\$item.name}}",
            "Value {\$item/name}", "Row {\$index + 1}", "Value \${index_1}",
        )
        templates.forEach { template ->
            listOf(
                "bad=Text(" + JsonPrimitive(template) + ",source:\"[3]\"}",
                "bad=Text(\"Must remain dynamic\",variant=" + JsonPrimitive(template) + ",source:\"[3]\"}",
            ).forEach { declaration ->
                val raw = "<a2ui>\nroot=Column([safe,bad])\nsafe=Text(\"Visible answer\")\n" + declaration +
                    "\ndamaged=UnknownCall()\n</a2ui>"
                assertFalse(template, prepared(raw).allowLiteralOnlyFallback)
                assertEquals(template, listOf("Visible answer"), reachableTexts(graph(recover(raw))))
            }
        }
        val currency = "Price is \$5.00; no binding or invented value."
        val raw = "<a2ui>\nroot=Column([body])\nbody=Text(" + JsonPrimitive(currency) + ",source:\"[3]\"}\n</a2ui>"
        assertEquals(listOf(currency, "[3]"), reachableTexts(graph(recover(raw))))
    }

    @Test fun runtimeObjectBindingsAndEmbeddedParentTemplatesGuardInlineDescendants() {
        val bindings = listOf("\$state", "\$bindState", "\$item", "\$bindItem", "\$index", "\$cond", "\$template", "\$computed")
            .map { key -> JsonObject().apply { addProperty(key, "/secret") }.toString() } +
            listOf(JsonPrimitive("Title \${/secret}").toString(), JsonPrimitive("Title {{\$item.name}}").toString())
        bindings.forEach { binding ->
            val root = "root=Column([safe,Card([bad],title=" + binding + ")])"
            assertTrue(A2uiExpressCodec.decodeStatements(listOf(root), requireRoot = false)
                .getAsJsonObject("elements").size() > 1)
            val raw = "<a2ui>\n" + root + "\nsafe=Text(\"Visible answer\")\n" +
                "bad=Text(\"Context-bound paragraph\",source:\"[3]\"}\ndamaged=UnknownCall()\n</a2ui>"
            assertFalse(binding, prepared(raw).allowLiteralOnlyFallback)
            assertFalse(binding, "Context-bound paragraph" in reachableTexts(graph(recover(raw))))
        }
    }

    @Test fun finalAndSameLineSemicolonsDoNotBecomePartOfTextTails() {
        val raw = "<a2ui>root=Column([first,second]);first=Text(\"First paragraph\",source:\"[3]\"};" +
            "second=Text(\"Second paragraph\",source:\"[10]\"};</a2ui>"
        assertEquals(listOf("First paragraph", "[3]", "Second paragraph", "[10]"),
            reachableTexts(graph(recover(raw))))
    }

    @Test fun commentsAreRemovedFromSyntaxWithoutChangingQuotedLiteralBytesDuringSalvage() {
        val message = "Keep // and /* literal */ ; ,fake=Text(\"inside prose\") exactly."
        val raw = "<a2ui>\nroot=Column([safe,body])\n" +
            "safe=Text/* syntax comment */(" + JsonPrimitive(message) + ") // trailing comment\n" +
            "/* multi-line comment\nfake=Text(\"Never code\")\n*/\n" +
            "body=Text(/* prefix */\"Exact paragraph\",/* tail */source:\"[3]\"};\n" +
            "damaged=UnknownCall()\n</a2ui>"
        val transformed = prepared(raw)
        assertTrue(transformed.input.contains(JsonPrimitive(message).toString()))
        assertFalse("Never code" in transformed.input)
        assertEquals(listOf(message, "Exact paragraph", "[3]"), reachableTexts(graph(recover(raw))))
    }

    @Test fun unterminatedBlockCommentsAreNotCompletedOrStrippedIntoValidRecoverySyntax() {
        val raw = "<a2ui>\nroot=Column([body])\nbody=Text(\"Must not infer a comment closer\",source:\"[3]\"} /* unfinished\n</a2ui>"
        val transformed = prepared(raw)
        assertEquals(raw, transformed.input)
        assertTrue(transformed.changes.isEmpty())
        assertFalse(transformed.allowLiteralOnlyFallback)
        assertThrows(IllegalArgumentException::class.java) { recover(raw) }
    }

    @Test fun anExplicitSafeReferenceCanShareAChildWithAGuardedBranch() {
        val raw = """<a2ui>
root=Column([bad,hidden])
hidden=Column([bad],visible=false)
bad=Text("Explicit root fact",source:"[1]"})
</a2ui>""".replace("})", "}")
        assertTrue("Explicit root fact" in leaves(graph(recover(raw))))
    }

    @Test fun duplicateTextDefinitionsAreAmbiguousRatherThanLastWriteWins() {
        val raw = """<a2ui>
root=Column([safe,bad])
safe=Text("Visible answer")
bad=Text("First conflicting value",source:"[1]"})
bad=Text("Second conflicting value",source:"[2]"})
</a2ui>""".replace("})", "}")
        val values = leaves(graph(recover(raw)))
        assertFalse("First conflicting value" in values)
        assertFalse("Second conflicting value" in values)
        assertTrue("Visible answer" in values)
    }

    @Test fun quotesCommentsAndEnvelopeTextDoNotCreateFalseDefinitions() {
        val message = "Keep ,fake=Text(\"not a definition\") and literal </a2ui>."
        val raw = "<a2ui>\nroot=Column([body])\n// fake=Text(\"comment\")\nbody=Text(" +
            JsonPrimitive(message) + ",source:\"[3]\"}\n</a2ui>"
        val values = reachableTexts(graph(recover(raw)))
        assertEquals(listOf(message, "[3]"), values)
        assertFalse("comment" in values)
        assertFalse("not a definition" in values)
    }

    @Test fun rawAndTripleQuotedLiteralBodiesUseProductionGrammar() {
        val literals = listOf(
            "r\"C:\\temp; /* literal */ // literal\"",
            "\"\"\"First line\nfake=Text(\"literal code\")\nLast line /* literal */\"\"\"",
            "r\"\"\"Raw line \\path; // literal\nfake=Text(\"raw code\")\nLast line\"\"\"",
        )
        literals.forEach { literal ->
            val expected = A2uiExpressCodec.literalPrefix(literal).first.asString
            val raw = "<a2ui>\nroot=Column([body]);body=Text(" + literal + ",source:\"[3]\"};</a2ui>"
            assertEquals(listOf(expected, "[3]"), reachableTexts(graph(recover(raw))))
        }
    }

    @Test fun unterminatedBodyOrCitationIsRejectedWithoutCompletingLiteralBytes() {
        listOf(
            "<a2ui>\nroot=Column([bad])\nbad=Text(\"Never closed\n</a2ui>",
            "<a2ui>\nroot=Column([bad])\nbad=Text(\"Complete body\",source:\"[1]\n</a2ui>",
        ).forEach { raw -> assertThrows(IllegalArgumentException::class.java) { recover(raw) } }
    }

    @Test fun parserIslandBudgetsDeclineOverBudgetContentAndDoNotRewriteOversizeInputs() {
        val raw = """<a2ui>
root=Column([safe,bad])
safe=Text("Visible answer")
bad=Text("This is over the literal length",source:"[1]"})
</a2ui>""".replace("})", "}")
        assertFalse("This is over the literal length" in prepared(raw, length = 4).input)
        assertFalse(prepared(raw, calls = 0).allowLiteralOnlyFallback)
        val oversized = raw + " ".repeat(A2uiExpressCodec.MAX_INPUT_CHARS)
        assertEquals(oversized, prepared(oversized).input)
        assertTrue(prepared(oversized).changes.isEmpty())
    }

    @Test fun existingDamagedTableFragmentsRemainSuppressedWhenStructuredContentSurvives() {
        val raw = requireNotNull(javaClass.getResourceAsStream("/recovery/BXP-003.raw.express"))
            .bufferedReader(Charsets.UTF_8).use { it.readText() }
        val values = leaves(graph(recover(raw)))
        assertFalse("Shatabdi Express also shows slight variation in duration, from about 2 h 15 m to 2 h 25 m." in values)
        assertFalse("Additional recovered text" in values)
        assertTrue("Shatabdi Express (120007)" in values)
    }
}
