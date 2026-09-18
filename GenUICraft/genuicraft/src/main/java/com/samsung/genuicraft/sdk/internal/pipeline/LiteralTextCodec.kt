package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

/**
 * Explicit literal escape for this catalog's string properties. The v0.9 wire profile accepts a
 * string, not the legacy {literalString: ...} object. An escaped value is still a normal wire
 * string; its reserved prefix tells this catalog's renderer not to evaluate the following text.
 * Prefix collisions are escaped again and decoded exactly once. Ordinary strings are unchanged.
 */
internal object LiteralTextCodec {
    private const val PREFIX = "\u001eGenUICraftLiteral:v1:"

    fun encode(value: String): String = if (
        value.startsWith(PREFIX) || value == "$" || value.startsWith("$/") || value.startsWith("\$state.") ||
        value.contains("\${") || Regex("\\{\\s*\\$(?:item|index)").containsMatchIn(value)
    ) PREFIX + value else value

    fun decode(value: String): String? = value.takeIf { it.startsWith(PREFIX) }?.substring(PREFIX.length)

    /** Protect a source value before insertion; do not call again on already protected data. */
    fun protectValue(value: JsonElement): JsonElement = protectValue(value, listItems = false)

    private fun protectValue(value: JsonElement, listItems: Boolean): JsonElement = when {
        value.isJsonPrimitive && value.asJsonPrimitive.isString -> JsonPrimitive(encode(value.asString))
        value.isJsonArray -> JsonArray().also { out -> value.asJsonArray.forEach { out.add(protectValue(it, listItems)) } }
        value.isJsonObject && value.asJsonObject.keySet() == setOf("path") -> value.deepCopy()
        value.isJsonObject && value.asJsonObject.keySet().any { it in expressionFields } -> value.deepCopy()
        value.isJsonObject -> JsonObject().also { out ->
            value.asJsonObject.entrySet().forEach { (key, item) ->
                // List.source is also a link target; Alert.source and other display text are not.
                val linkTarget = key in setOf("url", "href", "link") || listItems && key == "source"
                out.add(key, if (linkTarget) item.deepCopy() else protectValue(item, listItems))
            }
        }
        else -> value.deepCopy()
    }

    /** Conversion-only helper; ordinary output is returned byte-for-byte when no escape is needed. */
    fun protectVisibleContent(program: String): String {
        val graph = A2uiExpressCodec.decode(program)
        var changed = false
        graph.getAsJsonObject("elements").entrySet().forEach { (_, raw) ->
            val element = raw.asJsonObject
            val props = element.getAsJsonObject("props")
            val type = element.get("type").asString
            contentProperties[type].orEmpty().forEach { key ->
                props.get(key)?.let { value ->
                    val protected = protectValue(value, listItems = type == "List" && key == "items")
                    if (protected != value) { props.add(key, protected); changed = true }
                }
            }
        }
        return if (changed) A2uiExpressCodec.encode(graph) else program
    }

    private val expressionFields = setOf("call", "check", "\$state", "\$item", "\$computed", "\$cond", "\$template", "literalString")
    private val contentProperties = mapOf(
        "Text" to setOf("text", "heading"), "Card" to setOf("title", "subtitle"), "List" to setOf("items"),
        "Table" to setOf("title", "columns", "rows"), "Chart" to setOf("title", "subtitle", "yLabel", "columns", "rows"),
        "CodeBlock" to setOf("code", "title"), "ConsoleLog" to setOf("code", "title"),
        "Formula" to setOf("latex", "text", "title", "subtitle", "result"),
        "Alert" to setOf("message", "text", "title", "timestamp", "source"),
        "Checklist" to setOf("items", "title", "disclaimer", "source"),
        "EmailPreview" to setOf("subject", "body", "from", "to", "cc", "bcc", "date", "timestamp", "title"),
        "AudioPlayer" to setOf("description", "title"), "Video" to setOf("description", "title"),
        "Button" to setOf("label", "text"),
    )
}
