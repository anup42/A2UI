package com.samsung.genuicraft.sdk

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec

/** Host-supplied attribution is attached deterministically, after model validation. */
internal object SourceAttribution {
    fun append(document: GenUiDocument, sources: List<GenUiSource>): GenUiDocument {
        if (sources.isEmpty()) return document
        val graph = A2uiExpressCodec.decode(document.express)
        val elements = graph.getAsJsonObject("elements")
        var next = 0
        fun id(): String { while (elements.has("sdk_source_${next}")) next++; return "sdk_source_${next++}" }
        val buttons = JsonArray()
        sources.forEach { source ->
            val buttonId = id()
            buttons.add(buttonId)
            elements.add(buttonId, JsonObject().apply {
                addProperty("type", "Button")
                add("props", JsonObject().apply {
                    addProperty("label", LiteralTextCodec.encode("[${source.id}] ${source.title?.takeIf(String::isNotBlank) ?: source.url}"))
                    addProperty("variant", "secondary")
                })
                add("on", JsonObject().apply { add("press", JsonObject().apply {
                    addProperty("action", "openUrl")
                    add("params", JsonObject().apply { addProperty("url", source.url) })
                }) })
            })
        }
        val sourceCard = id()
        elements.add(sourceCard, JsonObject().apply {
            addProperty("type", "Card")
            add("props", JsonObject().apply { addProperty("title", "Sources") })
            add("children", buttons)
        })
        val rootId = id()
        elements.add(rootId, JsonObject().apply {
            addProperty("type", "Stack")
            add("props", JsonObject().apply { addProperty("direction", "vertical"); addProperty("gap", "md") })
            add("children", JsonArray().apply { add(graph.get("root").asString); add(sourceCard) })
        })
        graph.addProperty("root", rootId)
        return GenUiCompiler.compile(A2uiExpressCodec.encode(graph))
    }
}
