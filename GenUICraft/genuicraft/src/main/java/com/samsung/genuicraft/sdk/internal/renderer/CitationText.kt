package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.text.InlineTextContent
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.TextUnit
import com.samsung.genuicraft.sdk.GenUiSource

private val citationBracket = Regex("\\[([^\\]\\r\\n]{1,128})]")
private val numericCitationGroup = Regex("\\d+(?:\\s*,\\s*\\d+)+")

/** Add local preview actions without altering text, formatting, or pre-existing links. */
internal fun annotateSourceCitations(
    text: AnnotatedString,
    sources: List<GenUiSource>,
    color: Color,
    onPreview: (GenUiSource) -> Unit,
): AnnotatedString {
    val known = sources.groupBy { it.id }.filterValues { it.size == 1 }.mapValues { it.value.single() }
    if (known.isEmpty()) return text
    return buildAnnotatedString {
        append(text)
        fun addCitation(id: String, start: Int, end: Int) {
            val source = known[id] ?: return
            if (text.getLinkAnnotations(start, end).isNotEmpty()) return
            if (text.spanStyles.any { span ->
                span.start < end && span.end > start && span.item.fontFamily == FontFamily.Monospace
            }) return
            addLink(
                LinkAnnotation.Clickable(
                    tag = "source:$id",
                    styles = TextLinkStyles(SpanStyle(
                        color = color,
                        background = color.copy(alpha = .10f),
                        fontWeight = FontWeight.SemiBold,
                    )),
                    linkInteractionListener = { onPreview(source) },
                ),
                start, end,
            )
        }
        citationBracket.findAll(text.text).forEach { match ->
            val value = match.groupValues[1]
            if (value in known) {
                addCitation(value, match.range.first, match.range.last + 1)
            } else if (numericCitationGroup.matches(value)) {
                Regex("\\d+").findAll(value).forEach { number ->
                    val start = match.range.first + 1 + number.range.first
                    addCitation(number.value, start, start + number.value.length)
                }
            }
        }
    }
}

@Composable
internal fun CitationText(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = Color.Unspecified,
    fontSize: TextUnit = TextUnit.Unspecified,
    fontStyle: FontStyle? = null,
    fontWeight: FontWeight? = null,
    fontFamily: FontFamily? = null,
    letterSpacing: TextUnit = TextUnit.Unspecified,
    textDecoration: TextDecoration? = null,
    textAlign: TextAlign? = null,
    lineHeight: TextUnit = TextUnit.Unspecified,
    overflow: TextOverflow = TextOverflow.Clip,
    softWrap: Boolean = true,
    maxLines: Int = Int.MAX_VALUE,
    minLines: Int = 1,
    onTextLayout: (TextLayoutResult) -> Unit = {},
    style: TextStyle = LocalTextStyle.current,
) = CitationText(
    AnnotatedString(text), modifier, color, fontSize, fontStyle, fontWeight, fontFamily,
    letterSpacing, textDecoration, textAlign, lineHeight, overflow, softWrap, maxLines,
    minLines, emptyMap(), onTextLayout, style,
)

@Composable
internal fun CitationText(
    text: AnnotatedString,
    modifier: Modifier = Modifier,
    color: Color = Color.Unspecified,
    fontSize: TextUnit = TextUnit.Unspecified,
    fontStyle: FontStyle? = null,
    fontWeight: FontWeight? = null,
    fontFamily: FontFamily? = null,
    letterSpacing: TextUnit = TextUnit.Unspecified,
    textDecoration: TextDecoration? = null,
    textAlign: TextAlign? = null,
    lineHeight: TextUnit = TextUnit.Unspecified,
    overflow: TextOverflow = TextOverflow.Clip,
    softWrap: Boolean = true,
    maxLines: Int = Int.MAX_VALUE,
    minLines: Int = 1,
    inlineContent: Map<String, InlineTextContent> = emptyMap(),
    onTextLayout: (TextLayoutResult) -> Unit = {},
    style: TextStyle = LocalTextStyle.current,
) {
    val context = LocalSourceCitations.current
    val linkColor = MaterialTheme.colorScheme.primary
    val content = remember(text, context, linkColor, fontFamily, style.fontFamily) {
        if (context == null || fontFamily == FontFamily.Monospace || style.fontFamily == FontFamily.Monospace) text
        else annotateSourceCitations(text, context.sources, linkColor, context.onPreview)
    }
    Text(
        text = content, modifier = modifier, color = color, fontSize = fontSize,
        fontStyle = fontStyle, fontWeight = fontWeight, fontFamily = fontFamily,
        letterSpacing = letterSpacing, textDecoration = textDecoration, textAlign = textAlign,
        lineHeight = lineHeight, overflow = overflow, softWrap = softWrap, maxLines = maxLines,
        minLines = minLines, inlineContent = inlineContent, onTextLayout = onTextLayout, style = style,
    )
}
