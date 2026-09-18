package com.samsung.genuicraft.sdk

import com.google.gson.Gson

/**
 * A prompt scaffold for the fixed source-block mapping, never a fallback output.
 *
 * The model still returns a complete program and chooses table presentation. The normal binding,
 * content, and compiler checks must accept its response before anything can be rendered.
 */
internal object SourceLayoutScaffold {
    const val HEADER = "Express scaffold (preserve all lines; replace table SELECT_DOMAIN and SELECT_PRESENTATION):"
    private val gson = Gson()

    fun create(root: String, elementIds: List<String>, blocks: List<Map<String, Any>>): String {
        require(elementIds.size == blocks.size) { "Scaffold element count must match source blocks." }
        return buildString {
            appendLine("<a2ui>")
            appendLine(root)
            blocks.forEachIndexed { index, block ->
                fun binding(role: String): String {
                    val token = block.getValue(role)
                    require(token is String && token.startsWith("@source.")) { "Scaffold fields must be source bindings." }
                    return gson.toJson(token)
                }
                append(elementIds[index]).append('=')
                when (block.getValue("kind")) {
                    "heading" -> append("Text(").append(binding("text")).append(",variant=\"heading\")")
                    "paragraph" -> append("Text(").append(binding("text")).append(')')
                    "list" -> append("List(items=").append(binding("items")).append(')')
                    "table" -> append("Table(columns=").append(binding("columns"))
                        .append(",rows=").append(binding("rows"))
                        .append(",domain=\"SELECT_DOMAIN\",preferredPresentation=\"SELECT_PRESENTATION\")")
                    "code" -> {
                        append("CodeBlock(code=").append(binding("code"))
                        if (block.containsKey("title")) append(",title=").append(binding("title"))
                        append(')')
                    }
                    "divider" -> append("Divider()")
                    else -> error("Unknown source block kind in scaffold.")
                }
                appendLine()
            }
            appendLine("</a2ui>")
        }
    }

    fun appendTo(userPrompt: String, scaffold: String): String = "$userPrompt\n$HEADER\n$scaffold"
}
