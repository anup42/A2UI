package com.samsung.genuicraft.sdk

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy.MediaKind
import java.net.URI

/**
 * Imports a Bixby View response that already describes its UI. This path is deterministic: it
 * never calls a model, repairs source content, or converts an unsupported View into plain text.
 * Ordinary Markdown and other unrecognized input return null for the normal conversion path.
 */
object GenUiStructuredInput {
    const val PROVIDER = "sdk-structured-input"

    /** True only for a typed Bixby View node sequence or a narrow View wrapper around one. */
    @JvmStatic
    fun recognizes(text: String): Boolean {
        val parsed = runCatching { JsonParser.parseString(text) }.getOrNull()
        return if (parsed != null) looksLikeView(parsed) else looksLikeMalformedView(text)
    }

    /** Returns null for ordinary input; returns an explicit failure for a recognized bad View. */
    @JvmStatic
    fun convertOrNull(request: GenUiRequest): GenUiConversionResult? {
        val start = System.nanoTime()
        val parsed = runCatching { JsonParser.parseString(request.text) }.getOrNull()
        if (parsed == null && !looksLikeMalformedView(request.text)) return null
        if (parsed != null && !looksLikeView(parsed)) return null
        fun elapsed() = (System.nanoTime() - start) / 1_000_000
        return try {
            require(parsed != null) { "Malformed Bixby View JSON." }
            GenUiConverter.requestProblem(request)?.let { error(it) }
            val graph = importView(nodes(parsed))
            // Crossing the normal strict compiler keeps both persisted formats on the same
            // catalog/graph contract as model-produced documents.
            val wire = A2uiWireCodec.encode(graph, shortenIds = false).toString()
            val document = SourceAttribution.append(GenUiCompiler.compile(wire), request.sources)
                .copy(profile = "genuicraft_bixby_view_import_v1")
            GenUiConversionResult.Success(
                document = document,
                provider = PROVIDER,
                elapsedMs = elapsed(),
                attempts = 0,
                warnings = listOf("SDK structured input: imported Bixby View nodes without model inference."),
                repairKind = GenUiRepairKind.NONE,
            )
        } catch (failure: Exception) {
            GenUiConversionResult.Failure(
                message = "SDK structured input: ${failure.message ?: failure.javaClass.simpleName}",
                provider = PROVIDER,
                elapsedMs = elapsed(),
                attempts = 0,
            )
        }
    }

    private fun looksLikeMalformedView(text: String): Boolean {
        val compact = text.trimStart().take(512).replace(Regex("\\s+"), "")
        val nodeTypes = setOf("Markdown", "Divider", "CompoundCard")
        return nodeTypes.any { compact.startsWith("[{\"\$type\":\"$it\"") } ||
            compact.startsWith("[{\"type\":\"View\"") ||
            compact.startsWith("{\"\$type\":\"View\"") ||
            compact.startsWith("{\"\$type\":\"BixbyView\"") ||
            compact.startsWith("{\"type\":\"View\"")
    }

    private fun looksLikeView(value: JsonElement): Boolean = when {
        value.isJsonArray -> value.asJsonArray.any {
            it.isJsonObject && (it.asJsonObject.string("\$type") in VIEW_NODE_TYPES ||
                it.asJsonObject.string("type") == "View")
        }
        value.isJsonObject -> {
            val objectValue = value.asJsonObject
            when {
                objectValue.string("type") == "View" -> true
                objectValue.string("\$type") in setOf("View", "BixbyView") -> true
                objectValue.has("view") -> objectValue.get("view")?.let(::looksLikeView) == true
                else -> false
            }
        }
        else -> false
    }

    private fun nodes(value: JsonElement): JsonArray {
        if (value.isJsonArray) {
            val flattened = JsonArray()
            value.asJsonArray.forEachIndexed { index, raw ->
                val item = raw.requireObject("View envelope[$index]")
                when {
                    item.string("\$type") in setOf("View", "BixbyView") -> nodes(item).forEach { flattened.add(it) }
                    item.string("\$type") != null -> flattened.add(item)
                    item.string("type") == "View" -> nodes(item).forEach { flattened.add(it) }
                    item.string("type") == "Text" -> {
                        item.expectKeys(setOf("type", "value"), "View envelope[$index]")
                        flattened.add(JsonObject().apply {
                            addProperty("\$type", "Markdown")
                            addProperty("value", item.string("value") ?: error("View envelope[$index].value is required."))
                        })
                    }
                    else -> error("View envelope[$index] has unsupported type '${item.string("type") ?: "missing"}'.")
                }
            }
            return flattened
        }
        val wrapper = value.requireObject("View wrapper")
        if (wrapper.has("view")) {
            wrapper.expectKeys(setOf("view"), "View wrapper")
            return nodes(wrapper.get("view"))
        }
        if (wrapper.string("type") == "View") {
            wrapper.expectKeys(setOf("type", "value"), "View wrapper")
            val source = wrapper.requiredString("value", "View wrapper")
            val parsed = runCatching { JsonParser.parseString(source) }.getOrElse {
                error("View wrapper.value is not JSON: ${it.message}")
            }
            return nodes(parsed)
        }
        require(wrapper.string("\$type") in setOf("View", "BixbyView")) {
            "Unsupported View wrapper type."
        }
        wrapper.expectKeys(setOf("\$type", "contents", "analytics"), "View wrapper")
        return wrapper.get("contents").requireArray("View.contents")
    }

    private fun importView(source: JsonArray): JsonObject {
        require(source.size() > 0) { "View.contents is empty." }
        require(source.size() <= 1_023) { "View has too many nodes." }
        val elements = JsonObject()
        val children = JsonArray()
        elements.add("root", component("Stack", JsonObject().apply {
            addProperty("direction", "vertical")
            addProperty("gap", "none")
        }, children))
        source.forEachIndexed { index, raw ->
            val path = "View[$index]"
            val node = raw.requireObject(path)
            val id = "view_$index"
            val element = when (node.string("\$type")) {
                "Markdown" -> {
                    node.expectKeys(setOf("\$type", "value", "analytics"), path)
                    val value = node.string("value") ?: error("$path.value is required.")
                    component("Text", JsonObject().apply {
                        addProperty("text", LiteralTextCodec.encode(value))
                        // Preserve the upstream Markdown contract instead of guessing that
                        // leading heading markers are shell prompts or console output.
                        addProperty("variant", "markdown")
                    })
                }
                "Divider" -> {
                    node.expectKeys(setOf("\$type", "analytics"), path)
                    component("Divider", JsonObject())
                }
                "CompoundCard" -> placeCard(node, path)
                else -> error("$path has unsupported Bixby View type '${node.string("\$type") ?: "missing"}'.")
            }
            elements.add(id, element)
            children.add(id)
        }
        return JsonObject().apply {
            addProperty("root", "root")
            add("state", JsonObject())
            add("elements", elements)
        }
    }

    private fun placeCard(card: JsonObject, path: String): JsonObject {
        card.expectKeys(setOf("\$type", "contents", "analytics"), path)
        val contents = card.get("contents").requireArray("$path.contents")
        var title: String? = null
        var rating: String? = null
        var reviewCount: String? = null
        var address: String? = null
        var addressUrl: String? = null
        var images: JsonArray? = null
        var actions: JsonArray? = null
        var description: String? = null
        val seen = mutableSetOf<String>()
        contents.forEachIndexed { index, raw ->
            val itemPath = "$path.contents[$index]"
            val item = raw.requireObject(itemPath)
            val type = item.string("\$type") ?: error("$itemPath.\$type is required.")
            // A blank Paragraph and the terminal Divider are known decoration in this View.
            if (type != "Paragraph" && type != "Divider") {
                require(seen.add(type)) { "$path has duplicate $type blocks." }
            }
            when (type) {
                "TitleArea" -> {
                    item.expectKeys(setOf("\$type", "hAlign", "slot1", "analytics"), itemPath)
                    val slot = item.get("slot1").requireObject("$itemPath.slot1")
                    slot.expectKeys(setOf("\$type", "style", "text", "analytics"), "$itemPath.slot1")
                    require(slot.string("\$type") == "Paragraph") { "$itemPath.slot1 must be Paragraph." }
                    val original = slot.requiredString("text", "$itemPath.slot1")
                    title = original.removeSurrounding("**").takeIf(String::isNotBlank)
                        ?: error("$itemPath.slot1.text is empty.")
                }
                "SingleLine" -> {
                    item.expectKeys(setOf("\$type", "contents", "analytics"), itemPath)
                    val facts = item.get("contents").requireArray("$itemPath.contents")
                    facts.forEachIndexed factLoop@ { factIndex, factRaw ->
                        val factPath = "$itemPath.contents[$factIndex]"
                        val fact = factRaw.requireObject(factPath)
                        when (fact.string("\$type")) {
                            "Text" -> {
                                fact.expectKeys(setOf("\$type", "style", "color", "value", "analytics"), factPath)
                                val value = fact.string("value")?.takeIf(String::isNotBlank) ?: return@factLoop
                                when {
                                    value.matches(RATING) && rating == null -> rating = value
                                    value.matches(REVIEW_COUNT) && reviewCount == null -> reviewCount = value
                                    value == "★" -> Unit // Source star glyph; the card rating UI draws its own star.
                                    else -> error("$factPath contains unsupported place fact '$value'.")
                                }
                            }
                            "ImageLabel" -> {
                                fact.expectKeys(setOf("\$type", "url", "size", "analytics"), factPath)
                                val icon = fact.requiredString("url", factPath)
                                require(icon.contains("star", ignoreCase = true) && safeHttpUrl(icon)) {
                                    "$factPath is not a safe rating star icon."
                                }
                            }
                            "Spacer" -> fact.expectKeys(setOf("\$type", "size", "analytics"), factPath)
                            else -> error("$factPath has unsupported rating-row type '${fact.string("\$type") ?: "missing"}'.")
                        }
                    }
                }
                "ActionLinks" -> {
                    item.expectKeys(setOf("\$type", "items", "itemSize", "align", "analytics"), itemPath)
                    val links = item.get("items").requireArray("$itemPath.items")
                    require(links.size() == 1) { "$itemPath must have exactly one address link." }
                    val link = links[0].requireObject("$itemPath.items[0]")
                    link.expectKeys(setOf("label", "url", "truncate", "analytics"), "$itemPath.items[0]")
                    address = link.requiredString("label", "$itemPath.items[0]")
                    addressUrl = checkedUrl(link.requiredString("url", "$itemPath.items[0]"), "$itemPath.items[0].url")
                }
                "ImageList" -> {
                    item.expectKeys(setOf("\$type", "images", "backgroundColor", "objectFit", "analytics"), itemPath)
                    val photos = item.get("images").requireArray("$itemPath.images")
                    require(photos.size() > 0) { "$itemPath has no photos." }
                    images = JsonArray().also { out ->
                        photos.forEachIndexed { photoIndex, photoRaw ->
                            val photoPath = "$itemPath.images[$photoIndex]"
                            val photo = photoRaw.requireObject(photoPath)
                            photo.expectKeys(setOf("\$type", "url", "title", "onClickUrl", "analytics"), photoPath)
                            require(photo.string("\$type") == "Image") { "$photoPath must be Image." }
                            out.add(JsonObject().apply {
                                addProperty("url", checkedPhotoUrl(photo.requiredString("url", photoPath), "$photoPath.url"))
                                photo.string("title")?.let { addProperty("title", it) }
                                photo.string("onClickUrl")?.let { clickUrl ->
                                    addProperty("onClickUrl", checkedWebUrl(clickUrl, "$photoPath.onClickUrl"))
                                }
                            })
                        }
                    }
                }
                "LinkButtons" -> {
                    item.expectKeys(setOf("\$type", "items", "align", "size", "analytics"), itemPath)
                    val buttons = item.get("items").requireArray("$itemPath.items")
                    actions = JsonArray().also { out ->
                        buttons.forEachIndexed { buttonIndex, buttonRaw ->
                            val buttonPath = "$itemPath.items[$buttonIndex]"
                            val button = buttonRaw.requireObject(buttonPath)
                            button.expectKeys(setOf("label", "url", "onClick", "icon", "analytics"), buttonPath)
                            val label = button.requiredString("label", buttonPath)
                            val click = button.get("onClick")?.takeUnless(JsonElement::isJsonNull)
                                ?.requireObject("$buttonPath.onClick")
                            click?.expectKeys(setOf("url", "intent"), "$buttonPath.onClick")
                            val direct = button.string("url")
                            val onClick = click?.string("url")
                            require(direct == null || onClick == null || direct == onClick) {
                                "$buttonPath has conflicting action URLs."
                            }
                            val url = checkedUrl(direct ?: onClick ?: error("$buttonPath has no action URL."), "$buttonPath.url")
                            click?.string("intent")?.let { intent ->
                                require(intent == "url($url)") { "$buttonPath has unsupported action intent." }
                            }
                            button.get("icon")?.takeUnless(JsonElement::isJsonNull)?.requireObject("$buttonPath.icon")?.let { icon ->
                                icon.expectKeys(setOf("url", "lightThemeUrl", "darkThemeUrl"), "$buttonPath.icon")
                                icon.entrySet().forEach { (key, value) ->
                                    val iconUrl = value.takeUnless(JsonElement::isJsonNull)?.asString ?: return@forEach
                                    require(safeHttpUrl(iconUrl)) { "$buttonPath.icon.$key is not a safe HTTP(S) URL." }
                                }
                            }
                            out.add(JsonObject().apply { addProperty("label", label); addProperty("url", url) })
                        }
                    }
                }
                "Paragraph" -> {
                    item.expectKeys(setOf("\$type", "style", "text", "value", "analytics"), itemPath)
                    val text = item.string("text")
                    val value = item.string("value")
                    require(text == null || value == null || text == value) { "$itemPath has conflicting paragraph text." }
                    val visible = text ?: value
                    if (!visible.isNullOrBlank()) {
                        require(description == null) { "$path has multiple description paragraphs." }
                        description = visible
                    }
                }
                "Divider" -> item.expectKeys(setOf("\$type", "analytics"), itemPath)
                else -> error("$itemPath has unsupported card block '$type'.")
            }
        }
        require(seen.containsAll(setOf("TitleArea", "SingleLine", "ActionLinks", "ImageList", "LinkButtons"))) {
            "$path is not a complete supported place card."
        }
        return component("PlaceCard", JsonObject().apply {
            addProperty("title", LiteralTextCodec.encode(title ?: error("$path has no title.")))
            rating?.let { addProperty("rating", LiteralTextCodec.encode(it)) }
            reviewCount?.let { addProperty("reviewCount", LiteralTextCodec.encode(it)) }
            addProperty("address", LiteralTextCodec.encode(address ?: error("$path has no address.")))
            addProperty("addressUrl", addressUrl ?: error("$path has no address URL."))
            add("images", images ?: error("$path has no photos."))
            add("actions", actions ?: error("$path has no actions block."))
            description?.let { addProperty("description", LiteralTextCodec.encode(it)) }
        })
    }

    private fun component(type: String, props: JsonObject, children: JsonArray = JsonArray()): JsonObject =
        JsonObject().apply {
            addProperty("type", type)
            add("props", props)
            add("children", children)
        }

    private fun checkedUrl(url: String, path: String): String {
        require(url.length <= 4_096 && url == url.trim() && safeActionUrl(url)) {
            "$path is not an exact safe HTTP(S) or telephone URL."
        }
        return url
    }

    private fun checkedPhotoUrl(url: String, path: String): String {
        require(url.length <= 4_096 && url == url.trim() && url.startsWith("https://", ignoreCase = true) &&
            SafeContentPolicy.sanitizeMediaUrl(url, MediaKind.IMAGE) == url) {
            "$path is not an exact renderer-compatible HTTPS photo URL."
        }
        return url
    }

    private fun checkedWebUrl(url: String, path: String): String {
        require(url.length <= 4_096 && url == url.trim() && safeHttpUrl(url)) {
            "$path is not an exact safe HTTP(S) web URL."
        }
        return url
    }

    private fun safeActionUrl(url: String): Boolean {
        if (url.startsWith("tel:", ignoreCase = true)) {
            return SafeContentPolicy.sanitizeActionUrl(url) == url
        }
        return safeHttpUrl(url)
    }

    private fun safeHttpUrl(url: String): Boolean = runCatching {
        val uri = URI(url)
        uri.scheme?.lowercase() in setOf("http", "https") &&
            !uri.host.isNullOrBlank() && uri.userInfo == null &&
            SafeContentPolicy.sanitizeActionUrl(url) == url
    }.getOrDefault(false)

    private fun JsonElement?.requireObject(path: String): JsonObject =
        this?.takeIf(JsonElement::isJsonObject)?.asJsonObject ?: error("$path must be an object.")

    private fun JsonElement?.requireArray(path: String): JsonArray =
        this?.takeIf(JsonElement::isJsonArray)?.asJsonArray ?: error("$path must be an array.")

    private fun JsonObject.expectKeys(allowed: Set<String>, path: String) {
        val unknown = keySet() - allowed
        require(unknown.isEmpty()) { "$path has unsupported fields ${unknown.sorted()}." }
    }

    private fun JsonObject.string(key: String): String? {
        val value = get(key) ?: return null
        if (value.isJsonNull) return null
        require(value.isJsonPrimitive && value.asJsonPrimitive.isString) { "$key must be a string." }
        return value.asString
    }

    private fun JsonObject.requiredString(key: String, path: String): String =
        string(key)?.takeIf(String::isNotBlank) ?: error("$path.$key is required.")

    private val RATING = Regex("^(?:[0-4](?:\\.[0-9]+)?|5(?:\\.0+)?)$")
    private val REVIEW_COUNT = Regex("^\\([0-9][0-9,]*\\)$")
    private val VIEW_NODE_TYPES = setOf("Markdown", "Divider", "CompoundCard", "View", "BixbyView")
}
