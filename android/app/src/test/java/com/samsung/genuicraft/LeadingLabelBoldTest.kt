package com.samsung.genuicraft

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class LeadingLabelBoldTest {
    private val sampleText =
        "Morning: Kinkaku-ji (Golden Pavilion). Access via taxi or direct bus to minimize walking. " +
            "Afternoon: Kyoto Railway Museum (adjacent to Umekoji-Kyotonishi Station)."

    @Test
    fun nativeRenderer_boldsLeadingLabels() {
        val parseMethod = GenUiNativeRenderer::class.java.getDeclaredMethod("parseBoldMarkdown", String::class.java)
        parseMethod.isAccessible = true

        val annotated = parseMethod.invoke(GenUiNativeRenderer, sampleText) as AnnotatedString
        assertBoldRange(annotated, "Morning:")
        assertBoldRange(annotated, "Afternoon:")
    }

    @Test
    fun nativeRenderer_boldsDottedLeadingLabels() {
        val parseMethod = GenUiNativeRenderer::class.java.getDeclaredMethod("parseBoldMarkdown", String::class.java)
        parseMethod.isAccessible = true

        val annotated = parseMethod.invoke(
            GenUiNativeRenderer,
            "Est. Visit: 2 - 3 hours\nFamily Cost: €44\nHigh: The site is family friendly."
        ) as AnnotatedString

        assertBoldRange(annotated, "Est. Visit:")
        assertBoldRange(annotated, "Family Cost:")
        assertBoldRange(annotated, "High:")
        assertFalse("Only the leading label should be bold", hasBoldRange(annotated, "2 - 3 hours"))
    }

    @Test
    fun nativeRenderer_doesNotBoldUrlLikeLeadingText() {
        val parseMethod = GenUiNativeRenderer::class.java.getDeclaredMethod("parseBoldMarkdown", String::class.java)
        parseMethod.isAccessible = true

        val annotated = parseMethod.invoke(GenUiNativeRenderer, "https://example.com:443/path") as AnnotatedString

        assertFalse("URL scheme should not be treated as a leading label", hasBoldRange(annotated, "https:"))
    }

    @Test
    fun flatSpecRenderer_boldsDottedLeadingLabels() {
        val parseMethod = Class
            .forName("com.samsung.genuicraft.renderer.FlatSpecRendererKt")
            .getDeclaredMethod("parseBoldMarkdown", String::class.java)
        parseMethod.isAccessible = true

        val annotated = parseMethod.invoke(
            null,
            "Est. Visit: 2 - 3 hours\nNo. Stops: 1\nAvg. Cost: €44"
        ) as AnnotatedString

        assertBoldRange(annotated, "Est. Visit:")
        assertBoldRange(annotated, "No. Stops:")
        assertBoldRange(annotated, "Avg. Cost:")
        assertFalse("Only the leading label should be bold", hasBoldRange(annotated, "2 - 3 hours"))
    }

    @Test
    fun flatSpecRenderer_doesNotBoldUrlLikeLeadingText() {
        val parseMethod = Class
            .forName("com.samsung.genuicraft.renderer.FlatSpecRendererKt")
            .getDeclaredMethod("parseBoldMarkdown", String::class.java)
        parseMethod.isAccessible = true

        val annotated = parseMethod.invoke(null, "https://example.com:443/path") as AnnotatedString

        assertFalse("URL scheme should not be treated as a leading label", hasBoldRange(annotated, "https:"))
    }

    @Test
    fun flatSpecRenderer_normalizesMojibakeText() {
        val parseMethod = Class
            .forName("com.samsung.genuicraft.renderer.FlatSpecRendererKt")
            .getDeclaredMethod("parseBoldMarkdown", String::class.java)
        parseMethod.isAccessible = true

        val annotated = parseMethod.invoke(
            null,
            "Morning: Visit the CitÃ© des Sciences and use the MÃ©tro."
        ) as AnnotatedString

        assertTrue("Expected CitÃ© to normalize to Cité", annotated.text.contains("Cité"))
        assertTrue("Expected MÃ©tro to normalize to Métro", annotated.text.contains("Métro"))
        assertFalse("Mojibake marker should not remain", annotated.text.contains("Ã"))
    }

    @Test
    fun htmlRenderer_wrapsLeadingLabelsWithStrong() {
        val formatMethod =
            GenUiHtmlRenderer::class.java.getDeclaredMethod("formatInlineText", String::class.java, File::class.java)
        formatMethod.isAccessible = true

        val formatted = formatMethod.invoke(GenUiHtmlRenderer, sampleText, null) as String
        assertTrue("Morning label should be wrapped with <strong>", formatted.contains("<strong>Morning:</strong>"))
        assertTrue("Afternoon label should be wrapped with <strong>", formatted.contains("<strong>Afternoon:</strong>"))
    }

    @Test
    fun htmlRenderer_wrapsDottedLeadingLabelsWithStrong() {
        val formatMethod =
            GenUiHtmlRenderer::class.java.getDeclaredMethod("formatInlineText", String::class.java, File::class.java)
        formatMethod.isAccessible = true

        val formatted = formatMethod.invoke(
            GenUiHtmlRenderer,
            "Est. Visit: 2 - 3 hours\nFamily Cost: €44\nHigh: The site is family friendly.",
            null
        ) as String

        assertTrue("Est. Visit label should be wrapped with <strong>", formatted.contains("<strong>Est. Visit:</strong>"))
        assertTrue("Family Cost label should be wrapped with <strong>", formatted.contains("<strong>Family Cost:</strong>"))
        assertTrue("High label should be wrapped with <strong>", formatted.contains("<strong>High:</strong>"))
    }

    @Test
    fun htmlRenderer_doesNotWrapUrlLikeLeadingText() {
        val formatMethod =
            GenUiHtmlRenderer::class.java.getDeclaredMethod("formatInlineText", String::class.java, File::class.java)
        formatMethod.isAccessible = true

        val formatted = formatMethod.invoke(GenUiHtmlRenderer, "https://example.com:443/path", null) as String

        assertFalse("URL scheme should not be wrapped as a label", formatted.contains("<strong>https:</strong>"))
    }

    private fun assertBoldRange(annotated: AnnotatedString, label: String) {
        assertTrue("$label label missing in parsed text", annotated.text.indexOf(label) >= 0)
        assertTrue("$label label should be bold", hasBoldRange(annotated, label))
    }

    private fun hasBoldRange(annotated: AnnotatedString, text: String): Boolean {
        val start = annotated.text.indexOf(text)
        if (start < 0) return false
        val end = start + text.length
        return annotated.spanStyles.any { range ->
            range.item.fontWeight in setOf(FontWeight.W600, FontWeight.W700, FontWeight.SemiBold, FontWeight.Bold) &&
                range.start <= start &&
                range.end >= end
        }
    }
}
