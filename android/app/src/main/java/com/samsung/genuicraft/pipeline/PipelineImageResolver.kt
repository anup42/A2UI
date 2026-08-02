package com.samsung.genuicraft.pipeline

import android.util.Log
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.security.SafeContentPolicy
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
    private const val MAX_TABLE_ROW_IMAGES = 4
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
        val payload = PipelineMediaSanitizer.normalizeCanonicalGraphPayload(parsed)
        if (!payload.isJsonObject || !A2uiCanonicalGraph.validate(payload.asJsonObject, requireReservedRoot = false).isValid) {
            return Result(jsonText, resolvedCount = 0, replacedCount = 0)
        }

        val elements = payload.asJsonObject.getAsJsonObject("elements")
            ?: return Result(jsonText, resolvedCount = 0, replacedCount = 0)
        val textIndex = buildTextIndex(elements)
        var checked = 0
        var resolved = 0
        var replaced = 0

        val tableRowImagesAdded = attachCommonsImagesToTravelTables(
            payload = payload.asJsonObject,
            queryText = queryText,
            stage2Response = stage2Response
        )
        if (tableRowImagesAdded > 0) {
            resolved += tableRowImagesAdded
            replaced += tableRowImagesAdded
        }

        elements.entrySet().forEach { (id, node) ->
            if (!node.isJsonObject) return@forEach
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty()
            if (!type.equals("Image", ignoreCase = true)) return@forEach

            val props = element.getAsJsonObject("props") ?: JsonObject().also {
                element.add("props", it)
            }
            val currentUrl = imageUrlFromProps(props).orEmpty()
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
        if (!lower.startsWith("https://")) {
            return false
        }
        return SafeContentPolicy.isSafeMediaUrl(url, SafeContentPolicy.MediaKind.IMAGE)
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

    private fun attachCommonsImagesToTravelTables(
        payload: JsonObject,
        queryText: String,
        stage2Response: String
    ): Int {
        val elements = payload.getAsJsonObject("elements") ?: return 0
        val state = payload.getAsJsonObject("state") ?: JsonObject()
        var added = 0
        val seenUrls = mutableSetOf<String>()
        elements.entrySet().forEach { (_, node) ->
            if (added >= MAX_TABLE_ROW_IMAGES || !node.isJsonObject) return@forEach
            val element = node.asJsonObject
            val type = jsonStringOrNull(element.get("type")).orEmpty()
            if (!type.equals("Table", ignoreCase = true)) return@forEach
            val props = element.getAsJsonObject("props") ?: return@forEach
            if (!looksLikeTravelTable(props, queryText, stage2Response)) return@forEach
            val rows = tableRowsArray(payload, state, props) ?: return@forEach
            if (rows.size() == 0) return@forEach

            ensureTableImageColumns(props)
            rows.forEach { row ->
                if (added >= MAX_TABLE_ROW_IMAGES) return@forEach
                if (!row.isJsonObject) return@forEach
                val rowObj = row.asJsonObject
                val existingImage = jsonStringOrNull(rowObj.get("image"))
                    ?: jsonStringOrNull(rowObj.get("photo"))
                    ?: jsonStringOrNull(rowObj.get("imageUrl"))
                    ?: jsonStringOrNull(rowObj.get("mediaImage"))
                if (!existingImage.isNullOrBlank() && shouldValidateRemoteImage(existingImage)) {
                    return@forEach
                }

                val searchQueries = travelRowSearchQueries(rowObj, queryText)
                if (searchQueries.isEmpty()) return@forEach
                val imageUrl = searchQueries
                    .asSequence()
                    .mapNotNull { searchCommonsImageUrl(it) }
                    .firstOrNull { it !in seenUrls }
                    ?: return@forEach
                seenUrls += imageUrl
                rowObj.addProperty("image", imageUrl)
                rowObj.addProperty("imageAlt", travelRowImageAlt(rowObj, queryText))
                added += 1
            }
        }
        return added
    }

    private fun looksLikeTravelTable(
        props: JsonObject,
        queryText: String,
        stage2Response: String
    ): Boolean {
        val domain = jsonStringOrNull(props.get("domain")).orEmpty().lowercase(Locale.US)
        if (domain in setOf("travel", "itinerary", "tourism", "trip")) return true
        val columns = props.getAsJsonArray("columns")
            ?.mapNotNull { column ->
                if (!column.isJsonObject) return@mapNotNull null
                val obj = column.asJsonObject
                listOfNotNull(jsonStringOrNull(obj.get("key")), jsonStringOrNull(obj.get("label")))
                    .joinToString(" ")
                    .lowercase(Locale.US)
            }
            .orEmpty()
        val joinedColumns = columns.joinToString(" ")
        val hasItineraryShape =
            joinedColumns.contains("day") &&
                (
                    joinedColumns.contains("morning") ||
                        joinedColumns.contains("afternoon") ||
                        joinedColumns.contains("evening") ||
                        joinedColumns.contains("food") ||
                        joinedColumns.contains("area") ||
                        joinedColumns.contains("place") ||
                        joinedColumns.contains("location")
                    )
        return hasItineraryShape &&
            (PipelineMediaSanitizer.looksLikeTravelQuery(queryText) ||
                PipelineMediaSanitizer.looksLikeTravelContent(stage2Response))
    }

    private fun ensureTableImageColumns(props: JsonObject) {
        val columns = props.getAsJsonArray("columns") ?: return
        val keys = columns.mapNotNull { column ->
            if (!column.isJsonObject) null else jsonStringOrNull(column.asJsonObject.get("key"))
        }.map { it.lowercase(Locale.US) }
        if ("image" !in keys) {
            columns.add(JsonObject().apply {
                addProperty("key", "image")
                addProperty("label", "Image")
            })
        }
        if ("imagealt" !in keys && "image_alt" !in keys) {
            columns.add(JsonObject().apply {
                addProperty("key", "imageAlt")
                addProperty("label", "Image Alt")
            })
        }
    }

    private fun tableRowsArray(
        payload: JsonObject,
        state: JsonObject,
        props: JsonObject
    ): com.google.gson.JsonArray? {
        props.getAsJsonArray("rows")?.let { return it }
        val statePath = jsonStringOrNull(props.get("statePath")).orEmpty()
        if (statePath.isBlank()) return null
        return jsonAtPointer(state, statePath)?.takeIf { it.isJsonArray }?.asJsonArray
            ?: jsonAtPointer(payload, statePath)?.takeIf { it.isJsonArray }?.asJsonArray
    }

    private fun jsonAtPointer(root: JsonElement, pointer: String): JsonElement? {
        val parts = pointer.trim()
            .removePrefix("#")
            .split('/')
            .filter { it.isNotBlank() }
        var current: JsonElement = root
        parts.forEach { rawPart ->
            val part = rawPart.replace("~1", "/").replace("~0", "~")
            current = when {
                current.isJsonObject -> current.asJsonObject.get(part) ?: return null
                current.isJsonArray -> {
                    val index = part.toIntOrNull() ?: return null
                    current.asJsonArray.takeIf { index in 0 until it.size() }?.get(index) ?: return null
                }
                else -> return null
            }
        }
        return current
    }

    internal fun travelRowSearchQueriesForTest(
        rowValues: Map<String, String>,
        queryText: String
    ): List<String> {
        val row = JsonObject().apply {
            rowValues.forEach { (key, value) -> addProperty(key, value) }
        }
        return travelRowSearchQueries(row, queryText)
    }

    private fun travelRowSearchQueries(row: JsonObject, queryText: String): List<String> {
        val destination = PipelineMediaSanitizer.extractTravelLocationKeyword(queryText)
            .takeUnless { it.equals("travel", ignoreCase = true) }
            .orEmpty()
        val rawValues = travelRowTextValues(row)
        val placePhrases = rawValues
            .flatMap(::extractTravelPlacePhrases)
            .distinctBy { it.lowercase(Locale.US) }
            .sortedWith(
                compareBy<String> { phrase ->
                    if (phrase.contains("coast", ignoreCase = true)) 1 else 0
                }.thenBy { it.length }
            )
        return buildList {
            placePhrases.forEach { phrase ->
                add("$destination $phrase")
            }
            val area = jsonStringOrNull(row.get("area"))
                ?: jsonStringOrNull(row.get("location"))
                ?: jsonStringOrNull(row.get("place"))
            if (!destination.isBlank() && !area.isNullOrBlank()) {
                add("$destination $area")
            }
            rawValues.take(2).forEach { value ->
                add("$destination $value")
            }
        }
            .map(::normalizeSearchQuery)
            .filter { it.isNotBlank() }
            .distinctBy { it.lowercase(Locale.US) }
    }

    private fun travelRowImageAlt(row: JsonObject, queryText: String): String {
        val destination = PipelineMediaSanitizer.extractTravelLocationKeyword(queryText)
        val phrase = travelRowTextValues(row)
            .flatMap(::extractTravelPlacePhrases)
            .firstOrNull()
            ?: jsonStringOrNull(row.get("area"))
            ?: jsonStringOrNull(row.get("place"))
            ?: jsonStringOrNull(row.get("location"))
            ?: destination
        return listOf(phrase, destination)
            .filter { it.isNotBlank() }
            .distinctBy { it.lowercase(Locale.US) }
            .joinToString(", ")
    }

    private fun travelRowTextValues(row: JsonObject): List<String> {
        val preferredKeys = listOf(
            "area", "place", "location", "title", "focus", "destination", "neighborhood",
            "morning", "afternoon", "evening"
        )
        return preferredKeys.mapNotNull { key ->
            jsonStringOrNull(row.get(key))?.takeIf { it.isNotBlank() }
        }
    }

    private fun extractTravelPlacePhrases(value: String): List<String> {
        val suffixes = "Beach|Beaches|Town|Market|Temple|Museum|Palace|Cape|Viewpoint|Island|Islands|Bay|Garden|Park|Hill|Hills|Buddha|Falls|Lake|Fort|Church|Cathedral|Mosque|Coast|Harbor|Harbour|Pier|Road|Street|Village"
        val pattern = Regex(
            """\b([A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*(?:\s+(?:or|and)\s+[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*)?\s+(?:$suffixes))\b"""
        )
        return pattern.findAll(value)
            .flatMap { match -> expandJoinedPlacePhrase(match.groupValues[1]).asSequence() }
            .map { it.trim().trim('.', ',', ';', ':') }
            .filter { it.length >= 4 }
            .toList()
    }

    private fun expandJoinedPlacePhrase(phrase: String): List<String> {
        val parts = phrase.split(Regex("""\s+(?:or|and)\s+"""))
        if (parts.size < 2) return listOf(phrase)
        val suffix = parts.last().substringAfterLast(' ').trim()
        if (suffix.isBlank()) return listOf(phrase)
        val expanded = parts.mapIndexed { index, part ->
            val clean = part.trim()
            if (index == parts.lastIndex || clean.endsWith(suffix, ignoreCase = true)) {
                clean
            } else {
                "$clean $suffix"
            }
        }
        return expanded + phrase
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
