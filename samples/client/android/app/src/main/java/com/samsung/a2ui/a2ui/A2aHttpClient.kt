package com.samsung.a2ui.a2ui

import android.util.Log
import java.util.UUID
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class A2aHttpClient {
  private val client = OkHttpClient()
  private val json = Json { ignoreUnknownKeys = true }
  private val logTag = "A2UI-A2A"

  fun sendTextPrompt(baseUrl: String, text: String): List<ServerToClientMessage> {
    val part = buildJsonObject { put("text", JsonPrimitive(text)) }
    return sendMessage(baseUrl, listOf(part))
  }

  fun sendUserAction(baseUrl: String, message: Map<String, Any?>): List<ServerToClientMessage> {
    val dataPart = buildJsonObject {
      put("data", buildJsonObject { put("data", toJsonElement(message)) })
      put("metadata", buildJsonObject { put("mimeType", JsonPrimitive(A2UI_MIME_TYPE)) })
    }
    return sendMessage(baseUrl, listOf(dataPart))
  }

  private fun sendMessage(baseUrl: String, parts: List<JsonObject>): List<ServerToClientMessage> {
    val requestJson = buildSendMessageRequest(parts)
    val body = json.encodeToString(JsonElement.serializer(), requestJson)

    val request = Request.Builder()
      .url(normalizeBaseUrl(baseUrl) + "/message:send")
      .addHeader("Content-Type", "application/json")
      .addHeader("X-A2A-Extensions", A2UI_EXTENSION_URI)
      .post(body.toRequestBody("application/json".toMediaType()))
      .build()

    Log.i(logTag, "POST ${request.url} parts=${parts.size}")
    return try {
      client.newCall(request).execute().use { response ->
        val responseBody = response.body?.string().orEmpty()
        Log.i(logTag, "Response code=${response.code} length=${responseBody.length}")
        if (!response.isSuccessful) {
          Log.w(logTag, "A2A error body: ${responseBody.take(500)}")
          return emptyList()
        }
        return parseA2aResponse(responseBody)
      }
    } catch (e: Exception) {
      Log.e(logTag, "A2A request failed: ${e.message}", e)
      emptyList()
    }
  }

  private fun buildSendMessageRequest(parts: List<JsonObject>): JsonObject {
    return buildJsonObject {
      put("message", buildJsonObject {
        put("messageId", JsonPrimitive(UUID.randomUUID().toString()))
        put("role", JsonPrimitive("ROLE_USER"))
        put("parts", JsonArray(parts))
        put("metadata", buildJsonObject {
          put("a2uiClientCapabilities", buildJsonObject {
            put("supportedCatalogIds", buildJsonArray {
              add(JsonPrimitive(STANDARD_CATALOG_ID))
            })
          })
        })
      })
    }
  }

  private fun parseA2aResponse(raw: String): List<ServerToClientMessage> {
    val element = try {
      json.parseToJsonElement(raw)
    } catch (_: Exception) {
      Log.w(logTag, "A2A response is not valid JSON.")
      return emptyList()
    }

    val messageElements = extractA2uiMessageElements(element)
    if (messageElements.isEmpty()) {
      Log.w(logTag, "A2A response contained no A2UI messages.")
      return emptyList()
    }

    val messages = messageElements.mapNotNull { A2uiJsonParser.parseMessageElement(it) }
    Log.i(logTag, "Parsed ${messages.size} A2UI messages.")
    return messages
  }

  private fun extractA2uiMessageElements(element: JsonElement): List<JsonObject> {
    when (element) {
      is JsonArray -> {
        val directMessages = element.mapNotNull { it as? JsonObject }.filter { isA2uiMessage(it) }
        if (directMessages.isNotEmpty()) return directMessages
        return extractFromParts(element)
      }
      is JsonObject -> {
        if (isA2uiMessage(element)) return listOf(element)

        val result = element["result"]
        if (result is JsonObject) {
          val resultMessages = extractFromResponseObject(result)
          if (resultMessages.isNotEmpty()) return resultMessages
        }

        val messages = extractFromResponseObject(element)
        if (messages.isNotEmpty()) return messages
      }
      else -> return emptyList()
    }

    return emptyList()
  }

  private fun extractFromResponseObject(obj: JsonObject): List<JsonObject> {
    val task = obj["task"] as? JsonObject
    if (task != null) {
      val parts = task["status"]?.jsonObject?.get("message")?.jsonObject?.get("parts")?.jsonArray
      if (parts != null) return extractFromParts(parts)
    }

    val message = obj["message"] as? JsonObject
    if (message != null) {
      val parts = message["parts"]?.jsonArray
      if (parts != null) return extractFromParts(parts)
    }

    return emptyList()
  }

  private fun extractFromParts(parts: JsonArray): List<JsonObject> {
    val messages = mutableListOf<JsonObject>()
    for (part in parts) {
      val partObj = part as? JsonObject ?: continue
      val payload = extractDataPayload(partObj) ?: continue
      when (payload) {
        is JsonObject -> if (isA2uiMessage(payload)) messages.add(payload)
        is JsonArray -> payload.mapNotNull { it as? JsonObject }.filter { isA2uiMessage(it) }.forEach { messages.add(it) }
        else -> continue
      }
    }
    return messages
  }

  private fun extractDataPayload(partObj: JsonObject): JsonElement? {
    val data = partObj["data"]
    val metadata = partObj["metadata"] as? JsonObject
    val mimeType = metadata?.get("mimeType")?.jsonPrimitive?.content
      ?: metadata?.get("mime_type")?.jsonPrimitive?.content

    if (mimeType != null && !mimeType.contains("a2ui")) {
      return null
    }

    if (data is JsonObject) {
      val inner = data["data"]
      return inner ?: data
    }

    return data
  }

  private fun isA2uiMessage(obj: JsonObject): Boolean {
    return obj.containsKey("beginRendering") ||
      obj.containsKey("surfaceUpdate") ||
      obj.containsKey("dataModelUpdate") ||
      obj.containsKey("deleteSurface")
  }

  private fun toJsonElement(value: Any?): JsonElement {
    return when (value) {
      null -> JsonNull
      is JsonElement -> value
      is String -> JsonPrimitive(value)
      is Number -> JsonPrimitive(value)
      is Boolean -> JsonPrimitive(value)
      is Map<*, *> -> buildJsonObject {
        for ((key, v) in value) {
          put(key.toString(), toJsonElement(v))
        }
      }
      is List<*> -> buildJsonArray {
        for (item in value) {
          add(toJsonElement(item))
        }
      }
      else -> JsonPrimitive(value.toString())
    }
  }

  private fun normalizeBaseUrl(baseUrl: String): String {
    return baseUrl.trim().removeSuffix("/")
  }

  private companion object {
    const val A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.8"
    const val A2UI_MIME_TYPE = "application/json+a2ui"
    const val STANDARD_CATALOG_ID =
      "https://github.com/google/A2UI/blob/main/specification/0.8/json/standard_catalog_definition.json"
  }
}
