package com.samsung.genuicraft.pipeline

import android.util.Log
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.net.URLDecoder
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale

internal object PipelineImageResolver {
    private const val LOG_TAG = "PipelineImageResolver"
    private const val USER_AGENT = "A2UI GenUICraft Android/1.0 image-resolver"
    private const val MAX_IMAGES_TO_VALIDATE = 6
    private const val CONNECT_TIMEOUT_MS = 2500
    private const val READ_TIMEOUT_MS = 2500

    data class Result(
        val jsonText: String,
        val resolvedCount: Int,
        val replacedCount: Int
    )

    fun repairImageUrls(
        jsonText: String,
        queryText: String,
        stage2Response: String
    ): Result {
        val parsed = runCatching { JsonParser.parseString(jsonText) }.getOrNull()
            ?: return Result(jsonText, resolvedCount = 0, replacedCount = 0)
        val payload = PipelineMediaSanitizer.normalizeGenUiPayload(parsed)
        if (!payload.isJsonObject || !FlatSpecContract.looksLikeFlatSpec(payload)) {
            return Result(jsonText, resolvedCount = 0, replacedCount = 0)
        }

        val elements = payload.asJsonObject.getAsJsonObject("elements")
            ?: return Result(jsonText, resolvedCount = 0, replacedCount = 0)
        val textIndex = buildTextIndex(elements)
        var checked = 0
        var resolved = 0
        var replaced = 0
        var hasRealImage = false

        elements.entrySet().forEach { (id, node) ->
            if (!node.isJsonObject) return@forEach
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty()
            if (!type.equals("Image", ignoreCase = true)) return@forEach

            val props = element.getAsJsonObject("props") ?: JsonObject().also {
                element.add("props", it)
            }
            val currentUrl = imageUrlFromProps(props).orEmpty()
            if (shouldValidateRemoteImage(currentUrl)) {
                hasRealImage = true
            }
            if (checked >= MAX_IMAGES_TO_VALIDATE) return@forEach
            if (!shouldValidateRemoteImage(currentUrl)) return@forEach

            checked += 1
            val currentIsReachable = isReachableImageUrl(currentUrl)
            if (currentIsReachable && props.has("fallbackUrl")) {
                return@forEach
            }

            val searchQuery = buildCommonsSearchQuery(
                imageElementId = id,
                currentUrl = currentUrl,
                props = props,
                queryText = queryText,
                stage2Response = stage2Response,
                textIndex = textIndex
            )
            val fallbackUrl = searchCommonsImageUrl(searchQuery)
            if (fallbackUrl.isNullOrBlank()) {
                return@forEach
            }

            props.addProperty("fallbackUrl", fallbackUrl)
            resolved += 1
            if (!currentIsReachable) {
                putImageUrl(props, fallbackUrl)
                replaced += 1
            }
        }

        if (!hasRealImage &&
            (PipelineMediaSanitizer.looksLikeTravelQuery(queryText) ||
                PipelineMediaSanitizer.looksLikeTravelContent(stage2Response))
        ) {
            val fallbackUrl = searchCommonsImageUrl(
                normalizeSearchQuery("$queryText ${firstContentTitle(stage2Response).orEmpty()} landmark travel")
            )
            if (!fallbackUrl.isNullOrBlank()) {
                val imageId = buildUniqueElementId(elements, "auto_travel_image")
                elements.add(
                    imageId,
                    JsonObject().apply {
                        addProperty("type", "Image")
                        add("props", JsonObject().apply {
                            addProperty("url", fallbackUrl)
                            addProperty("fit", "cover")
                            addProperty("aspectRatio", "16:9")
                            addProperty("alt", firstContentTitle(stage2Response) ?: "Trip image")
                        })
                        add("children", com.google.gson.JsonArray())
                    }
                )
                prependRootChild(payload.asJsonObject, imageId)
                resolved += 1
                replaced += 1
            }
        }

        return if (resolved > 0 || replaced > 0) {
            Result(payload.toString(), resolvedCount = resolved, replacedCount = replaced)
        } else {
            Result(jsonText, resolvedCount = 0, replacedCount = 0)
        }
    }

    internal fun shouldValidateRemoteImage(rawUrl: String): Boolean {
        val url = rawUrl.trim()
        if (url.isBlank()) return false
        val lower = url.lowercase(Locale.US)
        if (lower.startsWith("assets/") ||
            lower.startsWith("/assets/") ||
            lower.startsWith("../assets/") ||
            lower.startsWith("genuicraft:")
        ) {
            return false
        }
        if (!lower.startsWith("http://") && !lower.startsWith("https://")) {
            return false
        }
        if (
            lower.contains("cdn.jsdelivr.net") ||
            lower.contains("bootstrap-icons") ||
            lower.endsWith(".svg")
        ) {
            return false
        }
        return PipelineMediaSanitizer.looksLikeUsableInlineImageUrl(url)
    }

    internal fun buildCommonsSearchQueryForTest(
        currentUrl: String,
        queryText: String
    ): String = normalizeSearchQuery(
        listOfNotNull(
            filenameSearchHint(currentUrl),
            queryText
        ).joinToString(" ")
    )

    internal fun extractCommonsImageUrlForTest(rawJson: String): String? =
        extractCommonsImageUrl(rawJson)

    private fun buildTextIndex(elements: JsonObject): Map<String, String> {
        return elements.entrySet().mapNotNull { (id, node) ->
            if (!node.isJsonObject) return@mapNotNull null
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty()
            if (!type.equals("Text", ignoreCase = true)) return@mapNotNull null
            val props = element.getAsJsonObject("props") ?: return@mapNotNull null
            val text = jsonStringOrNull(props.get("text")).orEmpty().trim()
            if (text.isBlank()) null else id to text
        }.toMap()
    }

    private fun imageUrlFromProps(props: JsonObject): String? {
        return listOf("url", "src", "image", "source")
            .firstNotNullOfOrNull { key -> extractUrlFromJsonValue(props.get(key)) }
    }

    private fun putImageUrl(props: JsonObject, url: String) {
        val preferredKey = listOf("url", "src", "image", "source")
            .firstOrNull { props.has(it) }
            ?: "url"
        val current = props.get(preferredKey)
        if (current != null && current.isJsonObject) {
            current.asJsonObject.addProperty("url", url)
        } else {
            props.addProperty(preferredKey, url)
        }
        if (preferredKey != "url" && !props.has("url")) {
            props.addProperty("url", url)
        }
    }

    private fun extractUrlFromJsonValue(value: JsonElement?): String? {
        jsonStringOrNull(value)?.let { direct ->
            return direct.takeIf { it.isNotBlank() }
        }
        if (value != null && value.isJsonObject) {
            val obj = value.asJsonObject
            return listOf("uri", "url", "src", "path", "value", "source", "image")
                .firstNotNullOfOrNull { key -> jsonStringOrNull(obj.get(key))?.takeIf { it.isNotBlank() } }
        }
        return null
    }

    private fun buildCommonsSearchQuery(
        imageElementId: String,
        currentUrl: String,
        props: JsonObject,
        queryText: String,
        stage2Response: String,
        textIndex: Map<String, String>
    ): String {
        val propHints = listOf("alt", "title", "label", "caption", "description")
            .mapNotNull { key -> jsonStringOrNull(props.get(key)) }
            .filter { it.isNotBlank() }
        val siblingHint = nearbyTextHint(imageElementId, textIndex)
        val filenameHint = filenameSearchHint(currentUrl)
        val responseTitle = stage2Response
            .lineSequence()
            .map { it.trim().removePrefix("#").trim() }
            .firstOrNull { it.length in 8..90 && !it.contains("http", ignoreCase = true) }
        return normalizeSearchQuery(
            buildList {
                addAll(propHints)
                if (!siblingHint.isNullOrBlank()) add(siblingHint)
                if (!filenameHint.isNullOrBlank()) add(filenameHint)
                if (!responseTitle.isNullOrBlank()) add(responseTitle)
                add(queryText)
            }.joinToString(" ")
        )
    }

    private fun nearbyTextHint(imageElementId: String, textIndex: Map<String, String>): String? {
        val numericSuffix = Regex("""(\d+)$""").find(imageElementId)?.groupValues?.getOrNull(1)
        if (!numericSuffix.isNullOrBlank()) {
            textIndex.entries.firstOrNull { (id, _) -> id.endsWith(numericSuffix) }?.value?.let { return it }
        }
        val normalized = imageElementId.lowercase(Locale.US)
            .replace(Regex("""image|media|hero|photo|pic"""), "")
            .replace(Regex("""[_-]+"""), " ")
            .trim()
        return normalized.takeIf { it.length >= 3 }
    }

    private fun filenameSearchHint(rawUrl: String): String? {
        val uri = runCatching { URI(rawUrl.trim()) }.getOrNull() ?: return null
        val segment = uri.path
            ?.substringAfterLast('/')
            ?.takeIf { it.isNotBlank() }
            ?: return null
        val decoded = runCatching {
            URLDecoder.decode(segment, StandardCharsets.UTF_8.name())
        }.getOrDefault(segment)
        return decoded
            .substringBefore('?')
            .substringBefore('#')
            .replace(Regex("""\.(png|jpe?g|webp|gif)$""", RegexOption.IGNORE_CASE), "")
            .replace('_', ' ')
            .replace('-', ' ')
            .takeIf { it.length >= 3 }
    }

    private fun normalizeSearchQuery(raw: String): String {
        val blocked = setOf(
            "image", "photo", "picture", "media", "hero", "jpg", "jpeg", "png", "webp",
            "show", "with", "from", "into", "your", "this", "that", "the", "and", "for",
            "day", "days", "vacation", "itinerary"
        )
        return raw
            .replace(Regex("""https?://\S+"""), " ")
            .replace(Regex("""[^A-Za-z0-9\s'-]"""), " ")
            .split(Regex("""\s+"""))
            .map { it.trim('\'', '-') }
            .filter { it.length >= 3 && it.lowercase(Locale.US) !in blocked }
            .distinctBy { it.lowercase(Locale.US) }
            .take(8)
            .joinToString(" ")
            .ifBlank { "travel landmark" }
    }

    private fun searchCommonsImageUrl(searchQuery: String): String? {
        val encoded = URLEncoder.encode(searchQuery, StandardCharsets.UTF_8.name())
        val apiUrl = "https://commons.wikimedia.org/w/api.php" +
            "?action=query&generator=search&gsrsearch=$encoded&gsrnamespace=6&gsrlimit=8" +
            "&prop=imageinfo&iiprop=url|extmetadata|mime&format=json"
        val raw = readUrl(apiUrl) ?: return null
        return extractCommonsImageUrl(raw)
    }

    private fun extractCommonsImageUrl(raw: String): String? {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull() ?: return null
        val pages = root.getAsJsonObject("query")?.getAsJsonObject("pages") ?: return null
        return pages.entrySet()
            .asSequence()
            .mapNotNull { (_, page) ->
                val pageObj = page.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
                val imageInfo = pageObj.getAsJsonArray("imageinfo")?.firstOrNull()
                    ?.takeIf { it.isJsonObject }
                    ?.asJsonObject
                    ?: return@mapNotNull null
                val mime = jsonStringOrNull(imageInfo.get("mime")).orEmpty().lowercase(Locale.US)
                if (mime.isNotBlank() && !mime.startsWith("image/")) return@mapNotNull null
                jsonStringOrNull(imageInfo.get("url"))
                    ?: jsonStringOrNull(imageInfo.get("thumburl"))
            }
            .map { PipelineMediaSanitizer.sanitizeMediaUrlToken(it) }
            .firstOrNull { shouldValidateRemoteImage(it) }
    }

    private fun firstContentTitle(text: String): String? {
        return text.lineSequence()
            .map { it.trim().removePrefix("#").trim() }
            .firstOrNull { line ->
                line.length in 5..90 &&
                    !line.contains("http", ignoreCase = true) &&
                    !line.startsWith("Media:", ignoreCase = true) &&
                    !line.startsWith("Action:", ignoreCase = true)
            }
    }

    private fun buildUniqueElementId(elements: JsonObject, base: String): String {
        var index = 1
        while (true) {
            val candidate = "${base}_$index"
            if (!elements.has(candidate)) return candidate
            index += 1
        }
    }

    private fun prependRootChild(payload: JsonObject, childId: String) {
        val rootId = jsonStringOrNull(payload.get("root")).orEmpty()
        if (rootId.isBlank()) return
        val root = payload.getAsJsonObject("elements")?.getAsJsonObject(rootId) ?: return
        val children = root.get("children")?.takeIf { it.isJsonArray }?.asJsonArray
            ?: com.google.gson.JsonArray().also { root.add("children", it) }
        val existing = children.mapNotNull { child ->
            if (child.isJsonPrimitive && child.asJsonPrimitive.isString) child.asString else null
        }
        if (childId in existing) return
        root.add("children", com.google.gson.JsonArray().apply {
            add(childId)
            existing.forEach { add(it) }
        })
    }

    private fun isReachableImageUrl(rawUrl: String): Boolean {
        val normalized = rawUrl.trim()
        if (normalized.isBlank()) return false
        return runCatching {
            val connection = (URL(normalized).openConnection() as HttpURLConnection).apply {
                requestMethod = "HEAD"
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", USER_AGENT)
                setRequestProperty("Accept", "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8")
            }
            try {
                val code = connection.responseCode
                val contentType = connection.contentType.orEmpty().lowercase(Locale.US)
                code in 200..399 && (contentType.startsWith("image/") || contentType.isBlank())
            } finally {
                connection.disconnect()
            }
        }.getOrElse {
            Log.d(LOG_TAG, "Image validation failed for generated URL: ${it.message ?: it.javaClass.simpleName}")
            false
        }
    }

    private fun readUrl(rawUrl: String): String? {
        return runCatching {
            val connection = (URL(rawUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", USER_AGENT)
                setRequestProperty("Accept", "application/json")
            }
            try {
                val code = connection.responseCode
                if (code !in 200..299) return null
                connection.inputStream.bufferedReader().use { it.readText() }
            } finally {
                connection.disconnect()
            }
        }.getOrElse {
            Log.d(LOG_TAG, "Commons image search failed: ${it.message ?: it.javaClass.simpleName}")
            null
        }
    }

    private fun jsonStringOrNull(element: JsonElement?): String? {
        if (element == null || element.isJsonNull) return null
        if (element.isJsonPrimitive && element.asJsonPrimitive.isString) return element.asString
        if (element.isJsonObject) {
            val literal = element.asJsonObject.get("literalString")
            if (literal != null && literal.isJsonPrimitive && literal.asJsonPrimitive.isString) {
                return literal.asString
            }
        }
        return null
    }
}
