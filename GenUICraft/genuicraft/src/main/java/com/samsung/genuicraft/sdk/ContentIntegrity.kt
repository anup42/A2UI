package com.samsung.genuicraft.sdk

import com.google.gson.JsonElement
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec
import com.samsung.genuicraft.sdk.internal.pipeline.RendererReferenceSemantics
import java.text.Normalizer
import java.util.Locale

/** Mechanical preservation checks; these are not a proof of semantic equivalence. */
internal object ContentIntegrity {
    private val citations = Regex("\\[\\d+(?:\\s*[,–-]\\s*\\d+)*]")
    private val urls = Regex("https?://[^\\s<>\"\\)]+")
    private val words = Regex("[\\p{L}\\p{N}]+")
    // Preserve signs, currency and adjacent units, including times, dates and percentages.
    private val numbers = Regex("[+−-]?[\\p{Sc}]?\\d+(?:[.,:/]\\d+)*(?:\\s*(?:%|°\\s*[CFK]|km|cm|mm|kg|mg|GB|MB|TB|dB|mAh|Hz|GHz|MHz|bps))?")
    private val visibleProps = mapOf(
        "Text" to listOf("text", "heading"), "Card" to listOf("title", "subtitle"),
        "List" to listOf("items"), "Table" to listOf("title", "columns", "rows"),
        "Chart" to listOf("title", "subtitle", "yLabel", "columns", "rows"),
        "CodeBlock" to listOf("title", "code"), "ConsoleLog" to listOf("title", "code"),
        "Formula" to listOf("title", "subtitle", "latex", "text", "result"),
        "Alert" to listOf("title", "message", "text", "timestamp", "source"),
        "Checklist" to listOf("title", "items", "disclaimer", "source"),
        "EmailPreview" to listOf("title", "subject", "from", "to", "cc", "bcc", "date", "timestamp", "body"),
        "AudioPlayer" to listOf("title", "description"), "Video" to listOf("title", "description"),
        "Image" to emptyList(), "Button" to listOf("label", "text"),
        "Stack" to emptyList(), "Divider" to emptyList(),
    )
    private val accessibilityTextProps = listOf(
        "ariaLabel", "accessibilityLabel", "contentDescription", "alt",
        "onClickLabel", "actionLabel", "stateDescription",
    )
    private val nestedTextFields = setOf(
        "text", "label", "value", "title", "name", "description", "alt", "ariaLabel",
        "accessibilityLabel", "contentDescription", "onClickLabel", "actionLabel",
        "stateDescription", "state",
    )
    private val nonContentControlLabels = setOf("source", "open source", "view source", "read source")

    fun suppliedUrls(request: GenUiRequest): Set<String> =
        (urls.findAll(request.text).map { it.value.trimEnd('.', ',', ';') } + request.sources.asSequence().map { it.url }).toSet()

    fun check(request: GenUiRequest, document: GenUiDocument): List<String> {
        val graph = A2uiExpressCodec.decode(document.express)
        val visible = mutableListOf<String>()
        val accessibility = mutableListOf<String>()
        val issues = mutableListOf<String>()
        val elements = graph.getAsJsonObject("elements")
        val reachable = mutableSetOf<String>()
        val renderOrder = mutableListOf<String>()
        fun visit(id: String) {
            if (!reachable.add(id)) {
                issues += "Conversion must render each element exactly once; repeated child: $id"
                return
            }
            renderOrder += id
            elements.getAsJsonObject(id)?.let { element ->
                RendererReferenceSemantics.references(element).forEach { visit(it.targetId) }
            }
        }
        visit(graph.get("root").asString)
        val unreachable = elements.keySet() - reachable
        if (unreachable.isNotEmpty()) issues += "Conversion must not hide source content in unreachable elements: ${unreachable.take(10)}"
        val allowedUrls = suppliedUrls(request)
        fun collect(value: JsonElement, into: MutableList<String>) {
            when {
                value.isJsonArray -> value.asJsonArray.forEach { collect(it, into) }
                value.isJsonObject -> value.asJsonObject.entrySet()
                    .filter { it.key in nestedTextFields }
                    .forEach { collect(it.value, into) }
                value.isJsonPrimitive -> into += LiteralTextCodec.decode(value.asString) ?: value.asString
            }
        }
        fun checkAction(value: JsonElement) {
            if (value.isJsonArray) { value.asJsonArray.forEach(::checkAction); return }
            val action = value.takeIf { it.isJsonObject }?.asJsonObject
            val params = action?.get("params")?.takeIf { it.isJsonObject }?.asJsonObject
            val url = params?.get("url")?.takeIf { it.isJsonPrimitive }?.asString
            if (action?.get("action")?.asString != "openUrl" || url !in allowedUrls ||
                url?.let { it.startsWith("https://", ignoreCase = true) || it.startsWith("http://", ignoreCase = true) } != true ||
                params?.keySet() != setOf("url")) {
                issues += "Generated actions may only open an exact supplied HTTP(S) URL."
            }
        }
        if (graph.getAsJsonObject("state").size() != 0) issues += "Conversion must use literal content, without generated state."
        elements.entrySet().forEach { (_, raw) ->
            val element = raw.asJsonObject
            val type = element.get("type").asString
            if (type !in visibleProps) issues += "Conversion cannot hide source content inside interactive $type components."
            if (listOf("visible", "watch", "repeat").any(element::has)) issues += "Conversion must not hide, repeat, or mutate source content."
            element.getAsJsonObject("on")?.entrySet()?.forEach { (_, action) -> checkAction(action) }
            element.getAsJsonObject("props")?.entrySet()?.forEach { (key, value) ->
                if (key in setOf("statePath", "rowsPath", "dataPath", "template", "itemTemplate")) issues += "Conversion must use literal source content."
            }
        }
        renderOrder.forEach { id ->
            val element = elements.getAsJsonObject(id)
            val type = element.get("type").asString
            val props = element.getAsJsonObject("props")
            visibleProps[type].orEmpty().forEach { key ->
                props.get(key)?.let { value ->
                    val allowedControl = type == "Button" && key in setOf("label", "text") &&
                        value.isJsonPrimitive &&
                        (LiteralTextCodec.decode(value.asString) ?: value.asString)
                            .trim().lowercase(Locale.ROOT) in nonContentControlLabels
                    if (!allowedControl) collect(value, visible)
                }
            }
            accessibilityTextProps.forEach { key ->
                props.get(key)?.let { value ->
                    val allowedControl = value.isJsonPrimitive &&
                        (LiteralTextCodec.decode(value.asString) ?: value.asString)
                            .trim().lowercase(Locale.ROOT) in nonContentControlLabels
                    if (!allowedControl) collect(value, accessibility)
                }
            }
            props.get("accessibility")?.let { collect(it, accessibility) }
        }
        val renderedText = visible.joinToString(" ")
        val accessibilityText = accessibility.joinToString(" ")
        val source = request.text
        val missingCitations = citations.findAll(source).map { it.value }.toSet().filterNot(renderedText::contains)
        fun clean(text: String) = text.replace(urls, "").replace(citations, "").replace(Regex("[*_`]"), "")
        fun facts(text: String) = numbers.findAll(clean(text)).map { it.value.replace(Regex("\\s+"), "").replace(",", "").replace('−', '-') }.groupingBy { it }.eachCount()
        val sourceFacts = facts(source)
        val outputFacts = facts(renderedText)
        val missingNumbers = sourceFacts.filter { (fact, count) -> (outputFacts[fact] ?: 0) < count }.keys
        val addedNumbers = outputFacts.filter { (fact, count) -> (sourceFacts[fact] ?: 0) < count }.keys
        val outputUrls = urls.findAll(document.express.replace("\\/", "/")).map { it.value.trimEnd('.', ',', ';') }.toSet()
        // Metadata URLs are attached by SourceAttribution after this model-output check.
        val requiredUrls = urls.findAll(source).map { it.value.trimEnd('.', ',', ';') }.toSet()
        val missingUrls = requiredUrls - outputUrls
        val inventedUrls = outputUrls - allowedUrls
        val sourceWords = tokens(clean(source))
        val outputWords = tokens(clean(renderedText))
        val missingWords = sourceWords.filter { (word, count) -> (outputWords[word] ?: 0) < count }.keys
        val addedWords = outputWords.filter { (word, count) -> (sourceWords[word] ?: 0) < count }.keys
        val sourceSequence = tokenSequence(clean(source))
        val outputSequence = tokenSequence(clean(renderedText))
        val accessibilitySequence = tokenSequence(clean(accessibilityText))
        val accessibilityInvented = !isSubsequence(accessibilitySequence, sourceSequence)
        return issues.distinct() + buildList {
            if (missingCitations.isNotEmpty()) add("Missing citation markers: ${missingCitations.take(10)}")
            if (missingNumbers.isNotEmpty()) add("Missing or changed numeric facts (including signs/units): ${missingNumbers.take(15)}")
            if (addedNumbers.isNotEmpty()) add("Added or duplicated numeric facts: ${addedNumbers.take(15)}")
            if (missingUrls.isNotEmpty()) add("Missing supplied URLs: ${missingUrls.take(5)}")
            if (inventedUrls.isNotEmpty()) add("URLs must come from the source: ${inventedUrls.take(5)}")
            if (missingWords.isNotEmpty()) add("Missing source wording: ${missingWords.take(25)}")
            if (addedWords.isNotEmpty()) add("Added or duplicated visible wording: ${addedWords.take(25)}")
            if (sourceSequence != outputSequence) add("Visible source wording order or multiplicity changed.")
            if (accessibilityInvented) add("Accessibility wording must preserve source order without invented content.")
        }
    }

    private fun tokens(text: String): Map<String, Int> = words.findAll(Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(Locale.ROOT))
        .map { it.value }.groupingBy { it }.eachCount()

    private fun tokenSequence(text: String): List<String> =
        words.findAll(Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(Locale.ROOT))
            .map { it.value }.toList()

    private fun isSubsequence(candidate: List<String>, source: List<String>): Boolean {
        if (candidate.isEmpty()) return true
        var sourceIndex = 0
        candidate.forEach { token ->
            while (sourceIndex < source.size && source[sourceIndex] != token) sourceIndex += 1
            if (sourceIndex == source.size) return false
            sourceIndex += 1
        }
        return true
    }
}
