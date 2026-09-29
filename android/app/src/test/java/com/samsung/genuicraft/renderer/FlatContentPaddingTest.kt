package com.samsung.genuicraft.renderer

import com.samsung.genuicraft.renderer.flat.model.FlatElement
import com.samsung.genuicraft.renderer.flat.model.FlatSpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class FlatContentPaddingTest {
    @Test
    fun nestedAnswerWrappersUseHostGutterAndKeepVerticalSpacing() {
        val spec = FlatSpec("root", emptyMap(), mapOf(
            "root" to FlatElement("Column", emptyMap(), listOf("answer", "sources")),
            "answer" to FlatElement("Column", mapOf("padding" to 16), listOf("section")),
            "section" to FlatElement("Column", mapOf("margin" to 8, "paddingHorizontal" to 24, "paddingVertical" to 6), listOf("text", "card")),
            "text" to FlatElement("Text", mapOf("text" to "Flight options"), emptyList()),
            "card" to FlatElement("Card", mapOf("contentPadding" to 10), listOf("cardContent")),
            "cardContent" to FlatElement("Column", mapOf("padding" to 12), emptyList()),
            "sources" to FlatElement("Card", mapOf("title" to "Sources"), emptyList())
        ))
        val adjusted = spec.withCollapsedRootHorizontalPadding()
        val answer = adjusted.elements.getValue("answer").props
        assertEquals(0, answer["paddingHorizontal"])
        assertEquals(16, answer["paddingVertical"])
        assertFalse(answer.containsKey("padding"))
        val section = adjusted.elements.getValue("section").props
        assertEquals(0, section["paddingHorizontal"])
        assertEquals(0, section["marginHorizontal"])
        assertEquals(6, section["paddingVertical"])
        assertEquals(8, section["marginVertical"])
        listOf("text", "card", "cardContent", "sources").forEach {
            assertEquals(spec.elements[it], adjusted.elements[it])
        }
        assertEquals(16, spec.elements.getValue("answer").props["padding"])
        assertEquals(adjusted, adjusted.withCollapsedRootHorizontalPadding())
    }

    @Test
    fun rowCellsAndFixedWidthPanelsRetainTheirPaddingAndCyclesTerminate() {
        val spec = FlatSpec("root", emptyMap(), mapOf(
            "root" to FlatElement("Column", emptyMap(), listOf("row", "panel", "root")),
            "row" to FlatElement("Row", mapOf("padding" to 16), listOf("cell")),
            "cell" to FlatElement("Column", mapOf("padding" to 8), emptyList()),
            "panel" to FlatElement("Box", mapOf("width" to 160, "padding" to 12), emptyList())
        ))
        val adjusted = spec.withCollapsedRootHorizontalPadding()
        assertEquals(0, adjusted.elements.getValue("row").props["paddingHorizontal"])
        assertEquals(spec.elements["cell"], adjusted.elements["cell"])
        assertEquals(spec.elements["panel"], adjusted.elements["panel"])
    }
}
