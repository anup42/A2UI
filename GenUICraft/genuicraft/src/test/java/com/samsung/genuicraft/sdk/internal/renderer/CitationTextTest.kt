package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import com.samsung.genuicraft.sdk.GenUiSource
import org.junit.Assert.*
import org.junit.Test

class CitationTextTest {
    private val sources = listOf(
        GenUiSource("1", "https://www.irctc.co.in/", "First"),
        GenUiSource("11", "https://www.irctc.co.in/", "Eleventh", "Exact excerpt"),
    )

    @Test fun `adjacent and grouped numbers use exact IDs and preserve text and spans`() {
        val text = buildAnnotatedString {
            append("Trains [1][11], also [1, 11], unknown [2].")
            addStyle(SpanStyle(fontWeight = FontWeight.Bold), 0, 6)
        }
        var selected: GenUiSource? = null
        val annotated = annotateSourceCitations(text, sources, Color.Blue) { selected = it }
        assertEquals(text.text, annotated.text)
        assertEquals(text.spanStyles, annotated.spanStyles)
        val links = annotated.getLinkAnnotations(0, annotated.length)
        assertEquals(listOf("source:1", "source:11", "source:1", "source:11"),
            links.map { (it.item as LinkAnnotation.Clickable).tag })
        val eleventh = links[1].item as LinkAnnotation.Clickable
        eleventh.linkInteractionListener?.onClick(eleventh)
        assertEquals(sources[1], selected)
    }

    @Test fun `unknown IDs absent metadata duplicate IDs and inline code stay plain`() {
        assertTrue(annotateSourceCitations(AnnotatedString("[2] [99]"), sources, Color.Blue) {}.getLinkAnnotations(0, 8).isEmpty())
        assertEquals(AnnotatedString("[1]"), annotateSourceCitations(AnnotatedString("[1]"), emptyList(), Color.Blue) {})
        assertTrue(annotateSourceCitations(AnnotatedString("[1]"), sources + sources[0], Color.Blue) {}.getLinkAnnotations(0, 3).isEmpty())
        val code = AnnotatedString("[1]", spanStyle = SpanStyle(fontFamily = FontFamily.Monospace))
        assertEquals(code, annotateSourceCitations(code, sources, Color.Blue) {})
    }

    @Test fun `existing hyperlinks are not replaced by citation actions`() {
        val text = buildAnnotatedString {
            append("[1]")
            addLink(LinkAnnotation.Url("https://www.irctc.co.in/"), 0, 3)
        }
        assertEquals(text, annotateSourceCitations(text, sources, Color.Blue) {})
    }
}
