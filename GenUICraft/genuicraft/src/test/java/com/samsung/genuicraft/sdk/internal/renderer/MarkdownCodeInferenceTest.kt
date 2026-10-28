package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MarkdownCodeInferenceTest {
    @Test
    fun itineraryHeadingsAndProseNeverBecomeConsoleOutput() {
        val itinerary = """
            ## Overall flow
            - A balanced itinerary exploring temples, gardens, and royal landmarks.

            ## Course highlights
            - Discover the Mysore Palace and Sri Ranganathaswamy Temple. [1][2][3]
        """.trimIndent()
        val day = "## Day 1\nStart your trip by exploring the spiritual and natural highlights near Mysore."

        assertNull(inferCodeBlockFromPlainText(itinerary, emptyMap()))
        assertNull(inferCodeBlockFromPlainText(day, emptyMap()))
        assertNull(inferCodeBlockFromPlainText("# Overview\nThis is a summary of the plan.", emptyMap()))
        assertNull(inferCodeBlockFromPlainText("> A quoted travel tip\n> Follow it when you arrive.", emptyMap()))
    }

    @Test
    fun explicitMarkdownWinsOverWeakOutputAndCommandCuesButFencesStillWork() {
        assertNull(inferCodeBlockFromPlainText("Result: Trip ready\ngit status is unrelated prose.", mapOf("variant" to "markdown")))
        val fenced = parseFencedCodeBlock("```bash\n\$ git status\n```")
        assertNotNull(fenced)
        assertTrue(fenced!!.isConsole)
        assertEquals("git status", fenced.code.removePrefix("$ "))
    }

    @Test
    fun realShellAndPythonSnippetsRetainCodeRendering() {
        val shell = inferCodeBlockFromPlainText("\$ git status\nOn branch main", emptyMap())
        val rootShell = inferCodeBlockFromPlainText("# apt update\nReading package lists...", emptyMap())
        val directCommand = inferCodeBlockFromPlainText("adb devices\nList of devices attached", emptyMap())
        val python = inferCodeBlockFromPlainText("def greet(name):\n    print(name)", emptyMap())

        assertTrue(shell!!.isConsole)
        assertTrue(rootShell!!.isConsole)
        assertTrue(directCommand!!.isConsole)
        assertFalse(python!!.isConsole)
        assertEquals("python", python.language)
    }
}
