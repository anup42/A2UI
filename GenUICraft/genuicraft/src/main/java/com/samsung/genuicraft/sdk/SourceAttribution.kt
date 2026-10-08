package com.samsung.genuicraft.sdk

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy

/** Host-supplied attribution is attached deterministically, after model validation. */
internal object SourceAttribution {
    private const val STATE_KEY = "__genuicraft_sources"

    /** Preview attribution adds only trusted metadata, preserving model component identities. */
    fun preview(document: GenUiDocument, sources: List<GenUiSource>): GenUiDocument {
        if (sources.isEmpty()) return document
        val graph = A2uiExpressCodec.decode(document.express)
        graph.getAsJsonObject("state").add(STATE_KEY, JsonArray().apply {
            sources.take(100).forEach { source -> add(JsonObject().apply {
                addProperty("id", source.id)
                addProperty("url", source.url)
                source.title?.let { addProperty("title", it) }
                source.description?.let { addProperty("description", it) }
            }) }
        })
        return document.copy(
            express = A2uiExpressCodec.encode(graph, shortenIds = false),
            a2uiJson = A2uiWireCodec.encode(graph, shortenIds = false).toString(),
        )
    }

    fun append(document: GenUiDocument, sources: List<GenUiSource>): GenUiDocument {
        if (sources.isEmpty()) return document
        val graph = A2uiExpressCodec.decode(document.express)
        val state = graph.getAsJsonObject("state") ?: JsonObject().also { graph.add("state", it) }
        state.add(STATE_KEY, JsonArray().apply {
            sources.forEach { source ->
                add(JsonObject().apply {
                    addProperty("id", source.id)
                    addProperty("url", source.url)
                    source.title?.let { addProperty("title", it) }
                    source.description?.let { addProperty("description", it) }
                })
            }
        })
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

    /** Reads persisted attribution, including SDK 0.5.1 source buttons that predate metadata. */
    fun read(document: GenUiDocument): List<GenUiSource> = runCatching {
        val input = document.express.ifBlank { document.a2uiJson }
        val graph = GenUiIrCodec.decode(JsonPrimitive(input)).canonicalGraph
        val state = graph.getAsJsonObject("state")
        val metadata = state?.get(STATE_KEY)
        val sources = if (metadata != null) {
            if (!metadata.isJsonArray || metadata.asJsonArray.size() > 100) return@runCatching emptyList()
            metadata.asJsonArray.mapNotNull { value ->
                if (!value.isJsonObject) return@mapNotNull null
                val source = value.asJsonObject
                sourceFrom(
                    source["id"].stringValue(), source["url"].stringValue(),
                    source["title"].stringValue(), source["description"].stringValue(),
                )
            }
        } else {
            val elements = graph.getAsJsonObject("elements")
            // Compilation normalizes element IDs, so older documents are recognized by
            // their source Card and exact labeled buttons, never by list position.
            val sourceButtonIds = elements.entrySet().flatMap { (_, value) ->
                val element = value.takeIf { it.isJsonObject }?.asJsonObject
                val title = element?.getAsJsonObject("props")?.get("title").stringValue()
                if (element?.get("type").stringValue() == "Card" && title.equals("Sources", true)) {
                    element?.getAsJsonArray("children")?.mapNotNull { it.stringValue() }.orEmpty()
                } else emptyList()
            }.toSet()
            elements.entrySet().mapNotNull { (id, value) ->
                if (id !in sourceButtonIds || !value.isJsonObject) return@mapNotNull null
                val element = value.asJsonObject
                if (element["type"].stringValue() != "Button") return@mapNotNull null
                val rawLabel = element.getAsJsonObject("props")?.get("label").stringValue() ?: return@mapNotNull null
                val label = LiteralTextCodec.decode(rawLabel) ?: rawLabel
                val match = Regex("^\\[([^\\]\\r\\n]{1,64})]\\s*(.*)$").matchEntire(label) ?: return@mapNotNull null
                val press = element.getAsJsonObject("on")?.getAsJsonObject("press") ?: return@mapNotNull null
                if (press["action"].stringValue() != "openUrl") return@mapNotNull null
                sourceFrom(match.groupValues[1], press.getAsJsonObject("params")?.get("url").stringValue(),
                    match.groupValues[2].takeIf(String::isNotBlank), null)
            }.take(100)
        }
        val ambiguousIds = sources.groupingBy { it.id }.eachCount().filterValues { it > 1 }.keys
        sources.filterNot { it.id in ambiguousIds }
    }.getOrDefault(emptyList())

    private fun JsonElement?.stringValue(): String? =
        this?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString

    private fun sourceFrom(id: String?, url: String?, title: String?, description: String?): GenUiSource? {
        if (id.isNullOrBlank() || id.length > 64 || url.isNullOrBlank() || url.length > 4096 ||
            (title?.length ?: 0) > 1000 || (description?.length ?: 0) > 4000) return null
        if (!url.startsWith("https://", true) && !url.startsWith("http://", true)) return null
        if (SafeContentPolicy.sanitizeActionUrl(url) != url) return null
        return GenUiSource(id, url, title, description)
    }
}
