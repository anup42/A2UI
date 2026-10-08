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

    @Test fun inlineRowsWaitForEntireTableCallIncludingLateMetadata() {
        val compiler = GenUiExpressStreamCompiler()
        val prefix = "<a2ui>\nroot=Column([table])\ntable=Table(columns=[\"Name\"],rows=[[\"First\"],[\"Second\"]]"
        assertNull(compiler.accept(prefix))
        val preview = compiler.accept(prefix + ",domain=\"weather\",preferredPresentation=\"cards\")\n")!!
        val props = graph(preview).getAsJsonObject("elements").getAsJsonObject("table").getAsJsonObject("props")
        assertEquals("weather", props.get("domain").asString)
        assertEquals("cards", props.get("preferredPresentation").asString)
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

    @Test fun modalWaitsForNonChildDependenciesAndActionBindings() {
        val compiler = GenUiExpressStreamCompiler()
        var raw = "<a2ui>\nroot=Column([title,dialog,action])\ntitle=Text(\"Title\")\n" +
            "dialog=Modal(trigger=trigger,content=body)\ntrigger=Button(\"Open\")\n" +
            "action=Button(\"Set\",onPress=setState(\"/answer\",true))\n"
        val first = compiler.accept(raw)!!
        assertEquals(listOf("title"), children(first))
        raw += "body=Text(\"Dialog body\")\n"
        val second = compiler.accept(raw)!!
        assertEquals(listOf("title", "dialog"), children(second))
        raw += "${'$'}/answer=false\n"
        val third = compiler.accept(raw)!!
        assertEquals(listOf("title", "dialog", "action"), children(third))
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
