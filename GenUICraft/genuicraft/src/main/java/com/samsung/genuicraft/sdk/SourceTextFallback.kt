package com.samsung.genuicraft.sdk

import com.google.gson.Gson

/** Lossless deterministic A2UI used only after generated output and bounded repair are rejected. */
internal object SourceTextFallback {
    private val gson = Gson()

    fun compile(sourceText: String): GenUiDocument {
        val bindings = SourceBindings.from(sourceText)
        val ids = bindings.blocks.indices.map(::elementId)
        val program = buildString {
            appendLine("<a2ui>")
            appendLine("root=Column([${ids.joinToString(",")}],gap=\"md\")")
            bindings.blocks.forEachIndexed { index, block ->
                append(ids[index]).append('=').append(component(block)).appendLine()
            }
            append("</a2ui>")
        }
        return GenUiCompiler.compile(bindings.expand(program))
    }

    private fun component(block: Map<String, Any>): String {
        fun binding(role: String): String {
            val token = block.getValue(role)
            require(token is String && token.startsWith("@source.")) {
                "Source fallback fields must be typed source bindings."
            }
            return gson.toJson(token)
        }
        return when (block.getValue("kind")) {
            "heading" -> "Text(${binding("text")},variant=\"heading\")"
            "paragraph" -> "Text(${binding("text")})"
            "list" -> "List(items=${binding("items")})"
            "table" -> "Table(columns=${binding("columns")},rows=${binding("rows")},domain=\"generic\",preferredPresentation=\"table\")"
            "code" -> buildString {
                append("CodeBlock(code=").append(binding("code"))
                if (block.containsKey("title")) append(",title=").append(binding("title"))
                append(')')
            }
            "divider" -> "Divider()"
            else -> error("Unknown source block kind '${block["kind"]}'.")
        }
    }

    private fun elementId(index: Int): String {
        var value = index
        return buildString {
            do {
                insert(0, 'a' + value % 26)
                value = value / 26 - 1
            } while (value >= 0)
        }
    }
}
