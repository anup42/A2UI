package com.samsung.genuicraft

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
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
        val morningStart = annotated.text.indexOf("Morning:")
        val afternoonStart = annotated.text.indexOf("Afternoon:")
        assertTrue("Morning label missing in parsed text", morningStart >= 0)
        assertTrue("Afternoon label missing in parsed text", afternoonStart >= 0)

        val hasMorningBold = annotated.spanStyles.any { range ->
            range.item.fontWeight == FontWeight.W700 &&
                range.start <= morningStart &&
                range.end >= morningStart + "Morning:".length
        }
        val hasAfternoonBold = annotated.spanStyles.any { range ->
            range.item.fontWeight == FontWeight.W700 &&
                range.start <= afternoonStart &&
                range.end >= afternoonStart + "Afternoon:".length
        }

        assertTrue("Morning label should be bold", hasMorningBold)
        assertTrue("Afternoon label should be bold", hasAfternoonBold)
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
}
