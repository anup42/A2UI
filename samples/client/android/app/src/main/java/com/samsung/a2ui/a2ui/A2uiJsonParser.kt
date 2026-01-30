package com.samsung.a2ui.a2ui

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

object A2uiJsonParser {
  @OptIn(ExperimentalSerializationApi::class)
  private val json = Json {
    ignoreUnknownKeys = true
    isLenient = true
    allowTrailingComma = true
  }

  fun looksLikeJson(raw: String): Boolean {
    val trimmed = normalize(raw).trim()
    if (trimmed.isEmpty()) return false
    return try {
      when (json.parseToJsonElement(trimmed)) {
        is JsonObject, is JsonArray -> true
        else -> false
      }
    } catch (_: Exception) {
      false
    }
  }

  fun parseMessages(raw: String): List<ServerToClientMessage> {
    val trimmed = normalize(raw).trim()
    if (trimmed.isEmpty()) return emptyList()

    val initial = parseMessagesInternal(trimmed)
    if (initial.isNotEmpty()) return initial

    val unquoted = unwrapJsonString(trimmed)
    if (unquoted != null) {
      val fromUnquoted = parseMessagesInternal(unquoted)
      if (fromUnquoted.isNotEmpty()) return fromUnquoted
    }

    val sanitized = removeTrailingCommas(trimmed)
    if (sanitized != trimmed) {
      val fromSanitized = parseMessagesInternal(sanitized)
      if (fromSanitized.isNotEmpty()) return fromSanitized
    }

    val extracted = extractBalancedJson(trimmed)
    if (extracted != null && extracted != trimmed) {
      val fromExtracted = parseMessagesInternal(extracted)
      if (fromExtracted.isNotEmpty()) return fromExtracted
    }

    val fromFragments = parseMessageFragments(trimmed)
    if (fromFragments.isNotEmpty()) return fromFragments

    return emptyList()
  }

  private fun parseMessagesInternal(raw: String): List<ServerToClientMessage> {
    return try {
      val element = json.parseToJsonElement(raw)
      when (element) {
        is JsonArray -> element.mapNotNull { parseMessageElement(it) }
        is JsonObject -> {
          val nested = element["messages"] ?: element["a2ui"]
          if (nested is JsonArray) {
            nested.mapNotNull { parseMessageElement(it) }
          } else {
            listOfNotNull(parseMessageElement(element))
          }
        }
        is JsonPrimitive -> {
          if (element.isString) {
            parseMessagesInternal(element.content)
          } else {
            emptyList()
          }
        }
        else -> emptyList()
      }
    } catch (_: Exception) {
      parseJsonLines(raw)
    }
  }

  private fun unwrapJsonString(raw: String): String? {
    val trimmed = raw.trim()
    if (!trimmed.startsWith("\"") || !trimmed.endsWith("\"")) return null
    return try {
      val element = json.parseToJsonElement(trimmed) as? JsonPrimitive
      if (element != null && element.isString) element.content else null
    } catch (_: Exception) {
      null
    }
  }

  private fun removeTrailingCommas(raw: String): String {
    return raw.replace(Regex(",\\s*([}\\]])"), "$1")
  }

  private fun normalize(raw: String): String {
    var result = raw
    if (result.isNotEmpty() && result[0] == '\uFEFF') {
      result = result.drop(1)
    }
    result = stripInvalidControlChars(result)
    result = removeJsonComments(result)
    return result
  }

  private fun stripInvalidControlChars(raw: String): String {
    val builder = StringBuilder(raw.length)
    for (ch in raw) {
      if (ch == '\n' || ch == '\r' || ch == '\t' || ch >= ' ') {
        builder.append(ch)
      }
    }
    return builder.toString()
  }

  private fun removeJsonComments(raw: String): String {
    val builder = StringBuilder(raw.length)
    var inString = false
    var escape = false
    var i = 0
    while (i < raw.length) {
      val ch = raw[i]
      if (escape) {
        builder.append(ch)
        escape = false
        i++
        continue
      }
      if (ch == '\\' && inString) {
        builder.append(ch)
        escape = true
        i++
        continue
      }
      if (ch == '"') {
        builder.append(ch)
        inString = !inString
        i++
        continue
      }
      if (!inString && ch == '/' && i + 1 < raw.length) {
        val next = raw[i + 1]
        if (next == '/') {
          i += 2
          while (i < raw.length && raw[i] != '\n') {
            i++
          }
          continue
        }
        if (next == '*') {
          i += 2
          while (i + 1 < raw.length && !(raw[i] == '*' && raw[i + 1] == '/')) {
            i++
          }
          if (i + 1 < raw.length) {
            i += 2
          }
          continue
        }
      }
      builder.append(ch)
      i++
    }
    return builder.toString()
  }

  private fun extractBalancedJson(raw: String): String? {
    var index = 0
    while (index < raw.length) {
      val start = nextJsonStart(raw, index)
      if (start == -1) return null
      val candidate = balancedSubstring(raw, start)
      if (candidate != null) return candidate.trim()
      index = start + 1
    }
    return null
  }

  private fun parseMessageFragments(raw: String): List<ServerToClientMessage> {
    val messages = mutableListOf<ServerToClientMessage>()
    var index = 0
    while (index < raw.length) {
      val start = nextObjectStart(raw, index)
      if (start == -1) break
      val candidate = balancedSubstring(raw, start)
      if (candidate == null) {
        index = start + 1
        continue
      }
      if (containsMessageKey(candidate)) {
        try {
          val element = json.parseToJsonElement(candidate)
          val message = parseMessageElement(element)
          if (message != null) {
            messages.add(message)
          }
        } catch (_: Exception) {
          // Ignore malformed fragments.
        }
      }
      index = start + candidate.length
    }
    return messages
  }

  private fun nextObjectStart(raw: String, from: Int): Int {
    var inString = false
    var escape = false
    for (i in from until raw.length) {
      val ch = raw[i]
      if (escape) {
        escape = false
        continue
      }
      if (ch == '\\' && inString) {
        escape = true
        continue
      }
      if (ch == '"') {
        inString = !inString
        continue
      }
      if (!inString && ch == '{') {
        return i
      }
    }
    return -1
  }

  private fun containsMessageKey(raw: String): Boolean {
    return raw.contains("\"surfaceUpdate\"") ||
      raw.contains("\"beginRendering\"") ||
      raw.contains("\"dataModelUpdate\"") ||
      raw.contains("\"deleteSurface\"") ||
      raw.contains("\"updateComponents\"") ||
      raw.contains("\"createSurface\"") ||
      raw.contains("\"updateDataModel\"")
  }

  private fun nextJsonStart(raw: String, from: Int): Int {
    for (i in from until raw.length) {
      val ch = raw[i]
      if (ch == '{' || ch == '[') return i
    }
    return -1
  }

  private fun balancedSubstring(raw: String, start: Int): String? {
    val open = raw[start]
    val close = if (open == '{') '}' else ']'
    var depth = 0
    var inString = false
    var escape = false
    for (i in start until raw.length) {
      val ch = raw[i]
      if (escape) {
        escape = false
        continue
      }
      if (ch == '\\' && inString) {
        escape = true
        continue
      }
      if (ch == '"') {
        inString = !inString
        continue
      }
      if (!inString) {
        if (ch == open) {
          depth++
        } else if (ch == close) {
          depth--
          if (depth == 0) {
            return raw.substring(start, i + 1)
          }
        }
      }
    }
    return null
  }

  private fun parseJsonLines(raw: String): List<ServerToClientMessage> {
    val messages = mutableListOf<ServerToClientMessage>()
    val lines = raw.lines()
    for (line in lines) {
      val trimmed = line.trim()
      if (trimmed.isEmpty()) continue
      try {
        val element = json.parseToJsonElement(trimmed)
        val message = parseMessageElement(element)
        if (message != null) {
          messages.add(message)
        }
      } catch (_: Exception) {
        continue
      }
    }
    return messages
  }

  fun parseMessageElement(element: JsonElement): ServerToClientMessage? {
    val obj = element as? JsonObject ?: return null

    obj["createSurface"]?.let { create ->
      val createObj = create.jsonObject
      val surfaceId = createObj["surfaceId"]?.jsonPrimitive?.content ?: return null
      return ServerToClientMessage(beginRendering = BeginRenderingMessage(surfaceId, "root", emptyMap()))
    }

    obj["beginRendering"]?.let { begin ->
      val beginObj = begin.jsonObject
      val surfaceId = beginObj["surfaceId"]?.jsonPrimitive?.content ?: return null
      val root = beginObj["root"]?.jsonPrimitive?.content ?: return null
      val styles = beginObj["styles"]?.jsonObject?.entries?.associate { it.key to it.value.jsonPrimitive.content }
        ?: emptyMap()
      return ServerToClientMessage(beginRendering = BeginRenderingMessage(surfaceId, root, styles))
    }

    obj["updateComponents"]?.let { update ->
      val updateObj = update.jsonObject
      val surfaceId = updateObj["surfaceId"]?.jsonPrimitive?.content ?: return null
      val components = updateObj["components"]?.jsonArray?.mapNotNull { parseComponent(it) } ?: emptyList()
      return ServerToClientMessage(surfaceUpdate = SurfaceUpdateMessage(surfaceId, components))
    }

    obj["surfaceUpdate"]?.let { update ->
      val updateObj = update.jsonObject
      val surfaceId = updateObj["surfaceId"]?.jsonPrimitive?.content ?: return null
      val components = updateObj["components"]?.jsonArray?.mapNotNull { parseComponent(it) } ?: emptyList()
      return ServerToClientMessage(surfaceUpdate = SurfaceUpdateMessage(surfaceId, components))
    }

    obj["updateDataModel"]?.let { update ->
      val updateObj = update.jsonObject
      val surfaceId = updateObj["surfaceId"]?.jsonPrimitive?.content ?: return null
      val path = updateObj["path"]?.jsonPrimitive?.content
      val value = updateObj["value"]
      return ServerToClientMessage(dataModelUpdate = DataModelUpdateMessage(surfaceId, path, value = value))
    }

    obj["dataModelUpdate"]?.let { update ->
      val updateObj = update.jsonObject
      val surfaceId = updateObj["surfaceId"]?.jsonPrimitive?.content ?: "main"
      val path = updateObj["path"]?.jsonPrimitive?.content
        ?: updateObj["dataModelRoot"]?.jsonPrimitive?.content
      val contents = updateObj["contents"]?.jsonArray?.mapNotNull { it as? JsonObject } ?: emptyList()
      val value = updateObj["value"] ?: updateObj["data"]
      return ServerToClientMessage(dataModelUpdate = DataModelUpdateMessage(surfaceId, path, contents, value))
    }

    obj["deleteSurface"]?.let { delete ->
      val deleteObj = delete.jsonObject
      val surfaceId = deleteObj["surfaceId"]?.jsonPrimitive?.content ?: return null
      return ServerToClientMessage(deleteSurface = DeleteSurfaceMessage(surfaceId))
    }

    return null
  }

  private fun parseComponent(element: JsonElement): ComponentInstance? {
    val obj = element as? JsonObject ?: return null
    val id = obj["id"]?.jsonPrimitive?.content ?: return null
    val weight = obj["weight"]?.jsonPrimitive?.floatOrNull

    val componentValue = obj["component"]
    when (componentValue) {
      is JsonObject -> {
        if (componentValue.isEmpty()) {
          return ComponentInstance(id = id, weight = weight)
        }
        val entry = componentValue.entries.first()
        val type = entry.key
        val props = entry.value as? JsonObject
        return ComponentInstance(id = id, weight = weight, type = type, props = props)
      }
      is JsonPrimitive -> {
        if (!componentValue.isString) return ComponentInstance(id = id, weight = weight)
        val type = componentValue.content
        val props = JsonObject(obj.filterKeys { key ->
          key != "id" && key != "component" && key != "weight"
        })
        return ComponentInstance(id = id, weight = weight, type = type, props = props)
      }
      else -> return ComponentInstance(id = id, weight = weight)
    }
  }

  private val JsonPrimitive.floatOrNull: Float?
    get() = try {
      if (isString) null else content.toFloat()
    } catch (_: Exception) {
      null
    }
}
