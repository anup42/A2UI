package com.samsung.genuicraft.sdk

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiCanonicalGraph
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiExpressStreamCompilerTest {
    @Test fun closedTextCallPreviewsBeforeDelimiterWithoutCommittingTheTail() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([heading,later])\nheading=Text(\"Ready now\")"
        val early = compiler.accept(raw)!!
        assertEquals(listOf("heading"), children(early))
        assertEquals(setOf("root", "heading"), graph(early).getAsJsonObject("elements").keySet())
        assertValid(early)
        assertNull(compiler.accept(raw + " // comment still streaming"))
        assertNull(compiler.accept(raw + " // comment still streaming\n"))
        val later = compiler.accept(raw + " // comment still streaming\nlater=Text(\"Added later\")")!!
        assertEquals(listOf("heading", "later"), children(later))
        assertEquals("Ready now", graph(later).getAsJsonObject("elements")
            .getAsJsonObject("heading").getAsJsonObject("props").get("text").asString)
        assertValid(later)
    }

    @Test fun closedRootTextCanBeTheFirstPreviewWithoutANewline() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Text(\"First content\")"
        val preview = compiler.accept(raw)!!
        assertEquals(setOf("root"), graph(preview).getAsJsonObject("elements").keySet())
        assertNull(compiler.accept(raw + ";"))
        assertValid(preview)
    }

    @Test fun rootLateWaitsForExplicitReachabilityAndKeepsDetachedComponentsOut() {
        val compiler = GenUiExpressStreamCompiler()
        val prefix = "<a2ui>\ndetached=Text(\"Unused\")\nheading=Text(\"Used\")\n"
        assertNull(compiler.accept(prefix))
        val preview = compiler.accept(prefix + "root=Column([heading])")!!
        assertEquals(setOf("root", "heading"), graph(preview).getAsJsonObject("elements").keySet())
        assertFalse(preview.document.express.contains("Unused"))
        assertValid(preview)
    }

    @Test fun blankTextDoesNotBecomeAContentOnlyPreview() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([blank,actual])\nblank=Text(\"  \")\n"
        assertNull(compiler.accept(raw))
        val preview = compiler.accept(raw + "actual=Text(\"Visible\")")!!
        assertEquals(listOf("blank", "actual"), children(preview))
        assertValid(preview)
    }

    @Test fun boundCardTitleCountsAsContentWhenItsStateIsReady() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\n${'$'}/title=\"Summary\"\nroot=Card([],title={path:\"/title\"})"
        val preview = compiler.accept(raw)!!
        assertEquals(setOf("root"), graph(preview).getAsJsonObject("elements").keySet())
        assertValid(preview)
    }

    @Test fun forwardReferencesGainChildrenInOriginalOrderWithStableIds() {
        val compiler = GenUiExpressStreamCompiler()
        var raw = "<a2ui>\nroot=Column([first_heading,later_card])\n"
        assertNull(compiler.accept(raw))
        raw += "later_card=Card([late_detail],title=\"Summary\")\n"
        val card = compiler.accept(raw)!!
        assertEquals(listOf("later_card"), children(card))
        assertTrue(graph(card).getAsJsonObject("elements").has("later_card"))
        raw += "first_heading=Text(\"Heading\")\n"
        val heading = compiler.accept(raw)!!
        assertEquals(listOf("first_heading", "later_card"), children(heading))
        raw += "late_detail=Text(\"Detail\")\n"
        val detail = compiler.accept(raw)!!
        assertEquals(listOf("late_detail"), graph(detail).getAsJsonObject("elements")
            .getAsJsonObject("later_card").getAsJsonArray("children").map { it.asString })
        assertEquals(4, detail.readyComponentCount)
        assertNull(compiler.accept(raw))
        assertValid(detail)
    }

    @Test fun everyCharacterSplitPreservesStringsCommentsAndFinalCompilation() {
        val quotes = "\"\"\""
        val raw = "<a2ui>\nroot=Column([heading,code]) /* comment ; () */\n" +
            "heading=Text(\"A \\\"quote\\\"; ₹5,000 ☀ and </a2ui>\") // ignored\n" +
            "code=CodeBlock(r${quotes}first line\n<a2ui> literal </a2ui>\nsecond line${quotes},language=\"text\");\n</a2ui>"
        val compiler = GenUiExpressStreamCompiler()
        var last: GenUiStreamPreview? = null
        for (end in 1..raw.length) {
            compiler.accept(raw.substring(0, end))?.let { assertValid(it); last = it }
        }
        assertNotNull(last)
        assertEquals(GenUiCompiler.compile(raw), GenUiCompiler.compile(last!!.document.express))
        assertTrue(last!!.document.express.contains("second line"))
    }

    @Test fun truncationRetainsOnlyAlreadyClosedValuesWithoutRepair() {
        val compiler = GenUiExpressStreamCompiler()
        val ready = "<a2ui>\nroot=Column([a,b])\na=Text(\"Ready\")\n"
        val first = compiler.accept(ready)!!
        val unfinished = ready + "b=Text(\"This is still streaming"
        assertNull(compiler.accept(unfinished))
        assertFalse(first.document.express.contains("still streaming"))
        val complete = compiler.accept(unfinished + "\")\n")!!
        assertTrue(complete.document.express.contains("This is still streaming"))
    }

    @Test fun stateBeforeRootWaitsForIdentityAndLatePresentation() {
        val compiler = GenUiExpressStreamCompiler()
        var raw = "<a2ui>\n${'$'}/={rows:[{name:\"First\"},"
        assertNull(compiler.accept(raw))
        raw += "{name:\"Second\"}]}\n"
        assertNull(compiler.accept(raw))
        raw += "root=Column([table])\n"
        assertNull(compiler.accept(raw))
        raw += "table=Table(columns=[{key:\"name\",label:\"Name\"}],statePath=\"/rows\",domain=\"weather\",preferredPresentation=\"cards\")\n"
        val preview = compiler.accept(raw)!!
        assertEquals(2, graph(preview).getAsJsonObject("state").getAsJsonArray("rows").size())
        assertEquals("cards", graph(preview).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").get("preferredPresentation").asString)
    }

    @Test fun knownTableStreamsOnlyCompleteStateRowsAndPreservesDuplicates() {
        for (assignment in listOf("${'$'}/rows=[", "${'$'}/={rows:[")) {
            val compiler = GenUiExpressStreamCompiler()
            val start = "<a2ui>\nroot=Column([table])\n" +
                "table=Table(columns=[{key:\"name\",label:\"Name\"}],statePath=\"/rows\",domain=\"comparison\",preferredPresentation=\"cards\")\n"
            assertNull(compiler.accept(start))
            val first = start + assignment + "{name:\"Same\"},"
            val one = compiler.accept(first)!!
            assertEquals(1, graph(one).getAsJsonObject("state").getAsJsonArray("rows").size())
            val incomplete = first + "{name:\"Sa"
            assertNull(compiler.accept(incomplete))
            val two = compiler.accept(incomplete + "me\"},")!!
            assertEquals(2, graph(two).getAsJsonObject("state").getAsJsonArray("rows").size())
            assertValid(two)
        }
    }

    @Test fun nestedKnownBindingStreamsStateRows() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([table])\n" +
            "table=Table(columns=[\"Name\"],statePath=\"/data/rows\")\n" +
            "${'$'}/={other:\"Not a row\",data:{rows:[[\"First\"],"
        val preview = compiler.accept(raw)!!
        assertEquals(1, graph(preview).getAsJsonObject("state").getAsJsonObject("data").getAsJsonArray("rows").size())
        assertFalse(preview.document.express.contains("Not a row"))
    }

    @Test fun inlineRowsPreviewBeforeTableClosesAndLateMetadataUpdatesTheSameId() {
        val compiler = GenUiExpressStreamCompiler()
        val prefix = "<a2ui>\nroot=Column([table])\ntable=Table(columns=[\"Name\"],rows=[[\"First\"],[\"Second\"]]"
        val early = compiler.accept(prefix)!!
        assertEquals(2, graph(early).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").getAsJsonArray("rows").size())
        assertEquals(setOf("root", "table"), graph(early).getAsJsonObject("elements").keySet())
        val preview = compiler.accept(prefix + ",domain=\"weather\",preferredPresentation=\"cards\")\n")!!
        val props = graph(preview).getAsJsonObject("elements").getAsJsonObject("table").getAsJsonObject("props")
        assertEquals("weather", props.get("domain").asString)
        assertEquals("cards", props.get("preferredPresentation").asString)
        assertValid(early)
        assertValid(preview)
    }

    @Test fun inlineTableStreamsOnlyClosedRowsAcrossEveryCharacterSplit() {
        val compiler = GenUiExpressStreamCompiler()
        val unfinished = "<a2ui>\nroot=Column([table])\n" +
            "table=Table(columns=[{key:\"city\",label:\"City\"}],rows=[[\"Ben"
        for (end in 1..unfinished.length) {
            assertNull("A partial cell or row was displayed at character $end", compiler.accept(unfinished.substring(0, end)))
        }
        val first = unfinished + "galuru\"]"
        val one = compiler.accept(first)!!
        assertEquals(1, graph(one).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").getAsJsonArray("rows").size())
        assertFalse(one.document.express.contains("Delhi"))
        val unfinishedSecond = first + ",[\"Del"
        assertNull(compiler.accept(unfinishedSecond))
        val second = unfinishedSecond + "hi\"]"
        val two = compiler.accept(second)!!
        assertEquals(2, graph(two).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").getAsJsonArray("rows").size())
        val full = second + "],title=\"Forecast\",domain=\"weather\")\n</a2ui>"
        val finalPreview = compiler.accept(full)!!
        assertEquals("Forecast", graph(finalPreview).getAsJsonObject("elements")
            .getAsJsonObject("table").getAsJsonObject("props").get("title").asString)
        assertEquals(GenUiCompiler.compile(full), GenUiCompiler.compile(finalPreview.document.express))
        assertValid(one)
        assertValid(two)
        assertValid(finalPreview)
    }

    @Test fun inlineTableWaitsForColumnsWhenTheyFollowRows() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([table])\ntable=Table(rows=[[\"First\"]],columns="
        assertNull(compiler.accept(raw))
        val preview = compiler.accept(raw + "[\"Name\"]")!!
        assertEquals(1, graph(preview).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").getAsJsonArray("rows").size())
        assertValid(preview)
    }

    @Test fun unfinishedCellEndingInParenthesisDoesNotMaskClosedEarlierRow() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([table])\n" +
            "table=Table(columns=[\"Name\"],rows=[[\"First\"],[\"unfinished)"
        val preview = compiler.accept(raw)!!
        val rows = graph(preview).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").getAsJsonArray("rows")
        assertEquals(1, rows.size())
        assertEquals("First", rows[0].asJsonArray[0].asString)
        assertFalse(preview.document.express.contains("unfinished"))
        assertValid(preview)
    }

    @Test fun inlineTableKeepsClosedDisplayMetadataAndWaitsForRoot() {
        val table = "table=Table(title=\"Forecast\",domain=\"weather\",columns=[\"City\"]," +
            "/* known rows */rows=[[r\"\"\"First line\nSecond line\"\"\"]"
        val rootFirst = GenUiExpressStreamCompiler()
        val early = rootFirst.accept("<a2ui>\nroot=Column([table])\n" + table)!!
        val earlyProps = graph(early).getAsJsonObject("elements").getAsJsonObject("table").getAsJsonObject("props")
        assertEquals("Forecast", earlyProps.get("title").asString)
        assertEquals("weather", earlyProps.get("domain").asString)
        assertValid(early)

        val compiler = GenUiExpressStreamCompiler()
        val tail = "<a2ui>\n" + table
        assertNull(compiler.accept(tail))
        val ready = compiler.accept(tail + "])\nroot=Column([table])")!!
        val props = graph(ready).getAsJsonObject("elements").getAsJsonObject("table").getAsJsonObject("props")
        assertEquals("Forecast", props.get("title").asString)
        assertEquals("weather", props.get("domain").asString)
        assertValid(ready)
    }

    @Test fun inlineTableRetainsKnownVisibilityGuards() {
        val hidden = GenUiExpressStreamCompiler()
        val hiddenRaw = "<a2ui>\nroot=Column([table])\n" +
            "table=Table(visible=false,columns=[\"Name\"],rows=[[\"Secret\"]"
        assertNull(hidden.accept(hiddenRaw))

        val unresolved = GenUiExpressStreamCompiler()
        val table = "table=Table(visible={path:\"/show\"},columns=[\"Name\"],rows=[[\"Secret\"]"
        val conditional = "<a2ui>\nroot=Column([table])\n" + table
        assertNull(unresolved.accept(conditional))

        val falseState = GenUiExpressStreamCompiler()
        val stateBefore = "<a2ui>\n${'$'}/show=false\nroot=Column([table])\n"
        assertNull(falseState.accept(stateBefore + table))

        val trueState = GenUiExpressStreamCompiler()
        val ready = trueState.accept(stateBefore.replace("show=false", "show=true") + table.replace("Secret", "Shown"))!!
        assertEquals(1, graph(ready).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").getAsJsonArray("rows").size())
        assertValid(ready)
    }

    @Test fun lateCompletedVisibilityGuardClearsPreviouslyShownTable() {
        val compiler = GenUiExpressStreamCompiler()
        val first = "<a2ui>\nroot=Column([table])\n" +
            "table=Table(columns=[\"Name\"],rows=[[\"Secret\"]"
        val shown = compiler.accept(first)!!
        assertEquals(listOf("table"), children(shown))
        val hidden = compiler.accept(first + "],visible=false,title=\"")!!
        assertEquals(emptyList<String>(), children(hidden))
        assertEquals(setOf("root"), graph(hidden).getAsJsonObject("elements").keySet())
        assertFalse(hidden.document.express.contains("Secret"))
        assertValid(hidden)
    }

    @Test fun scalarNamedArgumentWaitsForSeparatorInsteadOfFreezingNumericPrefix() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([table])\n" +
            "table=Table(columns=[\"Name\"],rows=[[\"First\"]],primaryColumn=1"
        val early = compiler.accept(raw)!!
        assertFalse(graph(early).getAsJsonObject("elements").getAsJsonObject("table")
            .getAsJsonObject("props").has("primaryColumn"))
        val settled = compiler.accept(raw + "0,title=\"Ready\"")!!
        val props = graph(settled).getAsJsonObject("elements").getAsJsonObject("table").getAsJsonObject("props")
        assertEquals(10, props.get("primaryColumn").asInt)
        assertEquals("Ready", props.get("title").asString)
        assertValid(early)
        assertValid(settled)
    }

    @Test fun falseAndPendingVisibilityNeverPromoteHiddenChildrenOrOrphans() {
        val compiler = GenUiExpressStreamCompiler()
        var raw = "<a2ui>\nroot=Column([hidden,conditional,safe])\n" +
            "hidden=Column([secret],visible=false)\nsecret=Text(\"Secret\")\n" +
            "conditional=Column([detail],visible={path:\"/show\"})\ndetail=Text(\"Conditional\")\n" +
            "orphan=Text(\"Orphan\")\n"
        assertNull(compiler.accept(raw))
        raw += "${'$'}/show=false\n"
        assertNull(compiler.accept(raw))
        raw += "safe=Text(\"Safe\")\n"
        val preview = compiler.accept(raw)!!
        assertEquals(listOf("safe"), children(preview))
        assertEquals(setOf("root", "safe"), graph(preview).getAsJsonObject("elements").keySet())
        assertFalse(preview.document.express.contains("Secret"))
        assertFalse(preview.document.express.contains("Conditional"))
        assertFalse(preview.document.express.contains("Orphan"))
        val shown = compiler.accept(raw + "${'$'}/show=true\n")!!
        assertEquals(listOf("conditional", "safe"), children(shown))
    }

    @Test fun modalWaitsForNonChildDependenciesWhilePreviewActionsStayInert() {
        val compiler = GenUiExpressStreamCompiler()
        var raw = "<a2ui>\nroot=Column([title,dialog,action])\ntitle=Text(\"Title\")\n" +
            "dialog=Modal(trigger=trigger,content=body)\ntrigger=Button(\"Open\")\n" +
            "action=Button(\"Set\",onPress=setState(\"/answer\",true))\n"
        val first = compiler.accept(raw)!!
        assertEquals(listOf("title", "action"), children(first))
        assertFalse(graph(first).getAsJsonObject("elements").getAsJsonObject("action").has("on"))
        raw += "body=Text(\"Dialog body\")\n"
        val second = compiler.accept(raw)!!
        assertEquals(listOf("title", "dialog", "action"), children(second))
        raw += "${'$'}/answer=false\n"
        val third = compiler.accept(raw)!!
        assertEquals(listOf("title", "dialog", "action"), children(third))
        assertFalse(graph(third).getAsJsonObject("elements").getAsJsonObject("action").has("on"))
        assertValid(third)
    }

    @Test fun unsupportedClosedComponentDoesNotBlockValidSibling() {
        val compiler = GenUiExpressStreamCompiler()
        val raw = "<a2ui>\nroot=Column([unknown,safe])\nunknown=Unsupported(\"Do not salvage\")\nsafe=Text(\"Safe\")\n"
        val preview = compiler.accept(raw)!!
        assertEquals(listOf("safe"), children(preview))
        assertFalse(preview.document.express.contains("Do not salvage"))
    }

    @Test fun repeatWaitsForStateAndTemplateWithoutPromotingItemContent() {
        val compiler = GenUiExpressStreamCompiler()
        var raw = "<a2ui>\nroot=Column([heading,repeated])\nheading=Text(\"Items\")\n" +
            "repeated=List([],repeat={statePath:\"/items\",template:\"template\"})\n" +
            "template=Text(text={\"${'$'}item\":\"name\"})\n"
        val waiting = compiler.accept(raw)!!
        assertEquals(listOf("heading"), children(waiting))
        assertFalse(graph(waiting).getAsJsonObject("elements").has("template"))
        raw += "${'$'}/items=[{name:\"First\"}]\n"
        val ready = compiler.accept(raw)!!
        assertEquals(listOf("heading", "repeated"), children(ready))
        assertTrue(graph(ready).getAsJsonObject("elements").has("template"))
        assertValid(ready)
    }

    @Test fun resetsDuplicatesCyclesAndBoundsFailClosedWithoutThrowing() {
        val valid = "<a2ui>\nroot=Text(\"Ready\")\n"
        val reset = GenUiExpressStreamCompiler()
        assertNotNull(reset.accept(valid))
        assertNull(reset.accept("<a2ui>\nroot=Text(\"Changed\")\n"))
        assertNull(reset.accept(valid + "</a2ui>"))
        val duplicate = GenUiExpressStreamCompiler()
        assertNotNull(duplicate.accept(valid))
        assertNull(duplicate.accept(valid + "root=Text(\"Duplicate\")\n"))
        assertNull(GenUiExpressStreamCompiler().accept("<a2ui>\nroot=Column([loop])\nloop=Column([root])\n"))
        assertNull(GenUiExpressStreamCompiler().accept("<a2ui>\nroot=Text(\"" + "x".repeat(120_001)))
        assertNull(GenUiExpressStreamCompiler().accept("<a2ui>\nroot=List(items=" + "[".repeat(65)))
    }

    @Test(timeout = 5_000) fun sharedDescendantDagHasBoundedProjectionWork() {
        val levels = 24
        val raw = buildString {
            append("<a2ui>\nroot=Column([a0,b0])\n")
            for (level in 0 until levels) {
                for (prefix in listOf("a", "b")) {
                    append("$prefix$level=")
                    if (level == levels - 1) append("Text(\"Leaf\")\n")
                    else append("Column([a${level + 1},b${level + 1}])\n")
                }
            }
        }
        val preview = GenUiExpressStreamCompiler().accept(raw)!!
        assertEquals(1 + levels * 2, preview.readyComponentCount)
        assertValid(preview)
    }

    private fun graph(preview: GenUiStreamPreview): JsonObject = A2uiWireCodec.decode(JsonParser.parseString(preview.document.a2uiJson))
    private fun children(preview: GenUiStreamPreview): List<String> = graph(preview).getAsJsonObject("elements")
        .getAsJsonObject("root").getAsJsonArray("children").map { it.asString }
    private fun assertValid(preview: GenUiStreamPreview) {
        assertTrue(A2uiCanonicalGraph.validate(graph(preview)).isValid)
        assertEquals(preview.readyComponentCount, graph(preview).getAsJsonObject("elements").size())
        assertNotNull(GenUiCompiler.compile(preview.document.express))
    }
}
