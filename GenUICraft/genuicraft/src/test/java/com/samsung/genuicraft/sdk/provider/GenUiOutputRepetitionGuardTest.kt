package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import org.junit.Assert.*
import org.junit.Test

class GenUiOutputRepetitionGuardTest {
    private val guard = GenUiOutputRepetitionGuard()

    @Test fun `same child reference still stops at exactly twenty`() {
        val nineteen = "<a2ui>\nroot=Column([" + List(19) { "av" }.joinToString(",")
        assertNull(guard.inspect(nineteen))
        val stop = guard.inspect("$nineteen,av")
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("child reference 'av'"))
    }

    @Test fun `twenty six distinct alphabetic ids may precede their definitions`() {
        val ids = (1..26).map(::alphabeticId)
        ids.indices.forEach { last ->
            assertNull(guard.inspect("<a2ui>\nroot=Column([${ids.take(last + 1).joinToString(",")}"))
        }
        val closedList = "<a2ui>\nroot=Column([${ids.joinToString(",")}])"
        assertNull(guard.inspect(closedList))
        val document = closedList + ids.joinToString("", transform = { "\n$it=Text(\"Item $it\")" }) + "\n</a2ui>"
        assertNull(guard.inspect(document))
        assertTrue(GenUiCompiler.compile(document).a2uiJson.isNotBlank())
    }

    @Test fun `distinct double letter ids are not repeated definitions`() {
        val ids = (27..56).map(::alphabeticId)
        assertNull(guard.inspect("<a2ui>\nroot=Column([${ids.joinToString(",")}"))
    }

    @Test fun `repeated component definitions stop exactly at twentieth occurrence`() {
        val prefix = "<a2ui>\nroot=Column([a])\n"
        val nineteen = prefix + List(19) { "a=Text(\"Same\")" }.joinToString("\n")
        assertNull(guard.inspect(nineteen))
        val stop = guard.inspect("$nineteen\na=Text(\"Same\")")
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("component definition 'a'"))
        val semicolon = prefix + List(19) { "a=Text(\"Same\")" }.joinToString(";")
        assertNull(guard.inspect(semicolon))
        assertEquals(20, guard.inspect("$semicolon;a=Text(\"Same\")")?.repeatLimit)
    }

    @Test fun `bare content loops outside literals still stop at twenty`() {
        val prefix = "<a2ui>\n"
        assertNull(guard.inspect(prefix + List(19) { "loop" }.joinToString(" ")))
        val stop = guard.inspect(prefix + List(20) { "loop" }.joinToString(" "))
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("bare top-level content"))
        val phrase = List(20) { "unexpected output\n" }.joinToString("")
        assertEquals(20, guard.inspect(prefix + phrase)?.repeatLimit)
    }

    @Test fun `quoted source and repeated state values stay opaque`() {
        val rows = List(25) { "{name:\"Same\",value:\"09:00\"}" }.joinToString(",")
        assertNull(guard.inspect("<a2ui>\n\$/={events:[$rows]}\nroot=Text(\"Ready\")"))
        val repeated = List(25) { "av" }.joinToString(",")
        assertNull(guard.inspect("<a2ui>\nroot=Text(\"Column([$repeated])\")"))
        val prose = List(25) { "loop" }.joinToString(" ")
        assertNull(guard.inspect("<a2ui>\nroot=Text(\"$prose\")"))
    }

    @Test fun `literal closing tag does not disable repetition protection`() {
        val repeated = List(20) { "a" }.joinToString(",")
        val stop = guard.inspect("<a2ui>\nroot=Text(\"</a2ui>\")\nb=Column([$repeated")
        assertEquals(20, stop?.repeatLimit)
    }

    @Test fun `completed documents are not cancelled retroactively`() {
        val repeated = List(20) { "a" }.joinToString(",")
        assertNull(guard.inspect("<a2ui>\nroot=Column([$repeated])\n</a2ui>"))
    }

    @Test fun `oversized unique graphs remain rejected by terminal compiler limits`() {
        val ids = (1..1024).map { "item_$it" }
        val prefix = "<a2ui>\nroot=Column([${ids.joinToString(",")}"
        assertNull(guard.inspect(prefix))
        val document = prefix + "])" + ids.joinToString("", transform = { "\n$it=Text(\"Item\")" }) + "\n</a2ui>"
        val error = assertThrows(IllegalArgumentException::class.java) { GenUiCompiler.compile(document) }
        assertTrue(error.message.orEmpty().contains("1024"))
    }


    @Test fun `quoted real references preserve the same twentieth occurrence cutoff`() {
        val prefix = "<a2ui>\nroot=Column(["
        val nineteen = List(19) { "\"av\"" }.joinToString(",")
        assertNull(guard.inspect(prefix + nineteen))
        val stop = guard.inspect(prefix + nineteen + ",\"av\"")
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("child reference 'av'"))
    }

    @Test fun `inline source strings do not hide real repeated refs after the call`() {
        val misleading = "Column([" + List(25) { "example" }.joinToString(",") + "])"
        val prefix = "<a2ui>\nroot=Column([Text(\"$misleading\"),"
        assertNull(guard.inspect(prefix + List(19) { "av" }.joinToString(",")))
        assertEquals(20, guard.inspect(prefix + List(20) { "av" }.joinToString(","))?.repeatLimit)
    }


    @Test fun `sequential bare alphabetic words stop at twenty outside valid calls`() {
        val nineteen = (27..45).map(::alphabeticId).joinToString(" ")
        assertNull(guard.inspect("<a2ui>\n$nineteen"))
        val twenty = (27..46).map(::alphabeticId).joinToString(" ")
        val stop = guard.inspect("<a2ui>\n$twenty")
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("bare top-level identifiers 'aa' through 'at'"))
    }

    @Test fun `sequential alphabetic words inside literals and state remain data`() {
        val words = (27..56).map(::alphabeticId).joinToString(" ")
        assertNull(guard.inspect("<a2ui>\nroot=Text(\"$words\")"))
        assertNull(guard.inspect("<a2ui>\n\$/={prose:\"$words\"}\nroot=Text(\"Ready\")"))
    }

    @Test fun `compiler valid comments cannot invent reference loops or disable a real cutoff`() {
        val fake = "Column([" + List(25) { "example" }.joinToString(",") + "]) </a2ui>"
        val ids = ('a'..'z').map(Char::toString)
        listOf("// $fake", "# $fake", "/* $fake */").forEach { comment ->
            val prefix = "<a2ui>\n$comment\nroot=Column(["
            assertNull(guard.inspect(prefix + ids.joinToString(",")))
            val valid = prefix + ids.joinToString(",") + "])" +
                ids.joinToString("") { "\n$it=Text(\"Item $it\")" } + "\n</a2ui>"
            assertTrue(A2uiExpressCodec.decode(valid).getAsJsonObject("elements").has("root"))
            assertNull(guard.inspect(prefix + List(19) { "av" }.joinToString(",")))
            assertEquals(20, guard.inspect(prefix + List(20) { "av" }.joinToString(","))?.repeatLimit)
        }
    }

    @Test fun `comments between real references preserve the twentieth reference cutoff`() {
        val prefix = "<a2ui>\nroot=Column(["
        listOf("// between\n", "# between\n", "/* between */").forEach { comment ->
            val separator = ", $comment "
            val nineteen = prefix + List(19) { "av" }.joinToString(separator)
            assertNull(guard.inspect(nineteen))
            val twenty = nineteen + separator + "av"
            assertEquals(20, guard.inspect(twenty)?.repeatLimit)
            assertTrue(A2uiExpressCodec.decode(twenty + "])\nav=Text(\"Ready\")\n</a2ui>")
                .getAsJsonObject("elements").has("root"))
        }
    }

    @Test fun `unfinished comments stay opaque until a later callback resumes real references`() {
        val nineteen = "<a2ui>\nroot=Column([" + List(19) { "av" }.joinToString(",") + ","
        val fake = "Column([" + List(25) { "example" }.joinToString(",") + "]) </a2ui>"
        listOf("// $fake" to "\n", "# $fake" to "\n", "/* $fake" to " */").forEach { (comment, ending) ->
            assertNull(guard.inspect(nineteen + comment))
            assertEquals(20, guard.inspect(nineteen + comment + ending + "av")?.repeatLimit)
        }
    }

    @Test fun `raw and triple source literals remain opaque while later real references still stop`() {
        val triple = "\"\"\""
        val fake = "Column([" + List(25) { "example" }.joinToString(",") + "]) </a2ui>"
        val literals = listOf("r\"literal\\\"", triple + "\n\"\n$fake\n" + triple,
            "r" + triple + "\n\\\"\n$fake\n" + triple)
        literals.forEach { literal ->
            val prefix = "<a2ui>\nroot=Column([Text($literal),"
            assertNull(guard.inspect(prefix + List(19) { "av" }.joinToString(",")))
            val twenty = prefix + List(20) { "av" }.joinToString(",")
            assertEquals(20, guard.inspect(twenty)?.repeatLimit)
            assertTrue(A2uiExpressCodec.decode(twenty + "])\nav=Text(\"Ready\")\n</a2ui>")
                .getAsJsonObject("elements").has("root"))
        }
    }

    @Test fun `raw and triple reference literals are counted only after their literal prefix closes`() {
        listOf("r\"av\"", "\"\"\"av\"\"\"", "r\"\"\"av\"\"\"").forEach { literal ->
            val nineteen = "<a2ui>\nroot=Column([" + List(19) { literal }.joinToString(",") + ","
            literal.indices.forEach { length -> assertNull(guard.inspect(nineteen + literal.take(length))) }
            assertEquals(20, guard.inspect(nineteen + literal)?.repeatLimit)
            assertTrue(A2uiExpressCodec.decode(nineteen + literal + "])\nav=Text(\"Ready\")\n</a2ui>")
                .getAsJsonObject("elements").has("root"))
        }
    }

    private fun alphabeticId(value: Int): String {
        var remaining = value
        val result = StringBuilder()
        while (remaining > 0) {
            remaining--
            result.append(('a'.code + remaining % 26).toChar())
            remaining /= 26
        }
        return result.reverse().toString()
    }
}
