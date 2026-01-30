package com.samsung.a2ui.a2ui

import android.content.Context
import android.util.Log
import java.io.IOException
import kotlinx.serialization.json.Json
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
import java.util.concurrent.TimeUnit
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

data class GeminiA2uiResult(
  val messages: List<ServerToClientMessage>,
  val error: String? = null,
  val requestJson: String? = null,
  val rawResponse: String? = null,
  val jsonPayload: String? = null
)

class GeminiA2uiClient(
  private val appContext: Context
) {
  private val client = OkHttpClient.Builder()
    .connectTimeout(15, TimeUnit.SECONDS)
    .readTimeout(30, TimeUnit.SECONDS)
    .writeTimeout(15, TimeUnit.SECONDS)
    .build()
  private val json = Json { ignoreUnknownKeys = true }
  private val promptCatalogJson by lazy {
    loadAssetJsonMinified("a2ui_prompt/standard_catalog_definition.json")
  }
  private val promptSchemaJson by lazy {
    loadAssetJsonMinified("a2ui_prompt/server_to_client_with_standard_catalog.json")
  }
  private val logTag = "A2UI-Gemini"

  fun sendPrompt(apiKey: String, model: String, text: String): GeminiA2uiResult {
    val prompt = buildA2uiPrompt(text)
    return sendRequest(apiKey, model, prompt, null)
  }

  fun sendUserAction(apiKey: String, model: String, action: Map<String, Any?>): GeminiA2uiResult {
    val actionJson = json.encodeToString(JsonElement.serializer(), toJsonElement(action))
    val userText = "User action event (JSON):\n$actionJson\nUpdate the UI and return A2UI JSON messages only."
    val prompt = GeminiPrompt(systemInstruction = SYSTEM_INSTRUCTION, userParts = listOf(userText))
    return sendRequest(apiKey, model, prompt, null)
  }

  private fun sendRequest(
    apiKey: String,
    model: String,
    prompt: GeminiPrompt,
    dataModel: JsonElement?
  ): GeminiA2uiResult {
    val primary = sendOnce(apiKey, model, prompt, "v1beta")
    val resolved = if (primary != null && shouldRetryWithV1(primary)) {
      Log.w(logTag, "Retrying with v1 endpoint for model=$model")
      val fallback = sendOnce(apiKey, model, prompt, "v1")
      fallback ?: primary
    } else {
      primary
    }

    return resolved?.let { ensureRenderable(it, dataModel) }
      ?: GeminiA2uiResult(emptyList(), "Gemini request failed: no response.")
  }

  private fun sendOnce(
    apiKey: String,
    model: String,
    prompt: GeminiPrompt,
    apiVersion: String
  ): GeminiA2uiResult? {
    val requestJson = buildRequestJson(apiVersion, prompt)
    val body = json.encodeToString(JsonElement.serializer(), requestJson)
    val request = Request.Builder()
      .url("https://generativelanguage.googleapis.com/$apiVersion/models/$model:generateContent?key=$apiKey")
      .addHeader("Content-Type", "application/json")
      .post(body.toRequestBody("application/json".toMediaType()))
      .build()

    Log.i(logTag, "POST Gemini api=$apiVersion model=$model promptParts=${prompt.userParts.size}")
    return try {
      client.newCall(request).execute().use { response ->
        val raw = response.body?.string().orEmpty()
        Log.i(logTag, "Gemini response api=$apiVersion code=${response.code} length=${raw.length}")
        if (!response.isSuccessful) {
          val message = if (raw.isNotBlank()) {
            "Gemini error: ${raw.take(500)}"
          } else {
            "Gemini request failed: HTTP ${response.code}"
          }
          Log.w(logTag, message)
          return GeminiA2uiResult(
            messages = emptyList(),
            error = message,
            requestJson = body,
            rawResponse = raw
          )
        }

        val text = extractCandidateText(raw)
          ?: return GeminiA2uiResult(
            messages = emptyList(),
            error = "Gemini response missing text.",
            requestJson = body,
            rawResponse = raw
          )
        Log.i(logTag, "Gemini text snippet: ${text.take(500)}")

        val jsonPayload = extractJsonPayload(text) ?: text
        Log.i(logTag, "Gemini JSON payload snippet: ${jsonPayload.take(500)}")
        var messages = A2uiJsonParser.parseMessages(jsonPayload)
        if (messages.isEmpty() && jsonPayload != text) {
          Log.w(logTag, "Retrying parse using full text payload.")
          messages = A2uiJsonParser.parseMessages(text)
        }
        if (messages.isNotEmpty() && jsonPayload != text) {
          val missingSurfaceUpdate = messages.none { it.surfaceUpdate != null }
          if (missingSurfaceUpdate && text.contains("\"surfaceUpdate\"")) {
            Log.w(logTag, "SurfaceUpdate missing; retrying parse using full text payload.")
            val fullMessages = A2uiJsonParser.parseMessages(text)
            if (fullMessages.size > messages.size) {
              messages = fullMessages
            }
          } else if (!hasRenderableMessages(messages) &&
            (text.contains("\"surfaceUpdate\"") || text.contains("\"beginRendering\""))) {
            Log.w(logTag, "Renderable messages missing; retrying parse using full text payload.")
            val fullMessages = A2uiJsonParser.parseMessages(text)
            if (fullMessages.size > messages.size) {
              messages = fullMessages
            }
          }
        }
        return if (messages.isEmpty()) {
          Log.w(logTag, "Gemini returned no valid A2UI messages.")
          logPayloadChunks(logTag, "Gemini JSON payload", jsonPayload)
          if (jsonPayload != text) {
            logPayloadChunks(logTag, "Gemini text payload", text)
          }
          GeminiA2uiResult(
            messages = emptyList(),
            error = "Gemini returned no valid A2UI messages.",
            requestJson = body,
            rawResponse = raw,
            jsonPayload = jsonPayload
          )
        } else {
          Log.i(logTag, "Parsed ${messages.size} A2UI messages.")
          GeminiA2uiResult(
            messages = messages,
            requestJson = body,
            rawResponse = raw,
            jsonPayload = jsonPayload
          )
        }
      }
    } catch (e: Exception) {
      Log.e(logTag, "Gemini request failed: ${e.message}", e)
      GeminiA2uiResult(
        messages = emptyList(),
        error = "Gemini request failed: ${e.message}",
        requestJson = body
      )
    }
  }

  private fun ensureRenderable(
    result: GeminiA2uiResult,
    dataModel: JsonElement?
  ): GeminiA2uiResult {
    if (hasRenderableMessages(result.messages)) {
      return result
    }
    val fallback = buildWeatherFallbackMessages(dataModel)
    if (fallback.isEmpty()) {
      return result
    }
    Log.w(logTag, "Generated fallback weather UI because no surfaceUpdate returned.")
    return GeminiA2uiResult(
      messages = result.messages + fallback,
      error = result.error,
      requestJson = result.requestJson,
      rawResponse = result.rawResponse,
      jsonPayload = result.jsonPayload
    )
  }

  private fun hasRenderableMessages(messages: List<ServerToClientMessage>): Boolean {
    return messages.any { it.surfaceUpdate != null || it.beginRendering != null }
  }

  private fun shouldRetryWithV1(result: GeminiA2uiResult): Boolean {
    val error = result.error ?: return false
    return error.contains("not found for API version v1beta") ||
      error.contains("not supported for generateContent")
  }

  private fun buildRequestJson(apiVersion: String, prompt: GeminiPrompt): JsonObject {
    val systemText = prompt.systemInstruction.trim()
    val mergedUserText = if (apiVersion == "v1") {
      (listOf(systemText) + prompt.userParts).joinToString("\n\n")
    } else {
      ""
    }

    val requestJson = buildJsonObject {
      if (apiVersion == "v1beta") {
        put(
          "systemInstruction",
          buildJsonObject {
            put(
              "parts",
              buildJsonArray {
                add(buildJsonObject { put("text", JsonPrimitive(systemText)) })
              }
            )
          }
        )
      }
      put(
        "contents",
        buildJsonArray {
          add(
            buildJsonObject {
              put("role", JsonPrimitive("user"))
              put(
                "parts",
                buildJsonArray {
                  if (apiVersion == "v1") {
                    add(buildJsonObject { put("text", JsonPrimitive(mergedUserText)) })
                  } else {
                    prompt.userParts.forEach { part ->
                      add(buildJsonObject { put("text", JsonPrimitive(part)) })
                    }
                  }
                }
              )
            }
          )
        }
      )
      put(
        "generationConfig",
        buildJsonObject {
          put("temperature", JsonPrimitive(0.2))
          put("maxOutputTokens", JsonPrimitive(1024))
        }
      )
    }

    return requestJson
  }

  private fun extractCandidateText(raw: String): String? {
    val element = try {
      json.parseToJsonElement(raw)
    } catch (_: Exception) {
      return null
    }

    val obj = element as? JsonObject ?: return null
    val candidates = obj["candidates"]?.jsonArray ?: return null
    val first = candidates.firstOrNull() as? JsonObject ?: return null
    val parts = first["content"]?.jsonObject?.get("parts")?.jsonArray ?: return null
    val text = parts.joinToString("\n") { part ->
      val objPart = part as? JsonObject
      val primitive = objPart?.get("text") as? JsonPrimitive
      primitive?.content.orEmpty()
    }
    return text.trim().ifBlank { null }
  }

  private fun extractJsonPayload(text: String): String? {
    val fenced = Regex("```(?:json)?\\s*([\\s\\S]*?)\\s*```").find(text)
    if (fenced != null) {
      return fenced.groupValues[1].trim()
    }
    return extractBalancedJson(text)
  }

  private fun extractBalancedJson(text: String): String? {
    var index = 0
    while (index < text.length) {
      val start = nextJsonStart(text, index)
      if (start == -1) return null
      val candidate = balancedSubstring(text, start)
      if (candidate != null) return candidate.trim()
      index = start + 1
    }
    return null
  }

  private fun nextJsonStart(text: String, from: Int): Int {
    for (i in from until text.length) {
      val ch = text[i]
      if (ch == '{' || ch == '[') return i
    }
    return -1
  }

  private fun balancedSubstring(text: String, start: Int): String? {
    val open = text[start]
    val close = if (open == '{') '}' else ']'
    var depth = 0
    var inString = false
    var escape = false
    for (i in start until text.length) {
      val ch = text[i]
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
            return text.substring(start, i + 1)
          }
        }
      }
    }
    return null
  }

  private fun logPayloadChunks(tag: String, label: String, payload: String) {
    val chunkSize = 3000
    var offset = 0
    var part = 1
    while (offset < payload.length) {
      val end = (offset + chunkSize).coerceAtMost(payload.length)
      Log.w(tag, "$label[$part]: ${payload.substring(offset, end)}")
      offset = end
      part++
    }
  }

  private fun buildA2uiPrompt(instructions: String): GeminiPrompt {
    val combinedInstructions = mutableListOf<String>()
    val trimmedInstructions = instructions.trim()
    if (trimmedInstructions.isNotEmpty()) {
      combinedInstructions.add(trimmedInstructions)
    }
    if (combinedInstructions.isEmpty()) {
      throw IllegalArgumentException("No instructions provided.")
    }
    val combinedText = combinedInstructions.joinToString("\" and \"")
    val userParts = listOf(
      PROMPT_INTRO,
      "The user's layout request is: \"$combinedText\"",
      "The Component Catalog you can use is: $promptCatalogJson",
      "The A2UI Protocol Message Schema: \"$promptSchemaJson\"",
      PROMPT_REQUIREMENTS,
      PROMPT_URL_RULES
    )
    return GeminiPrompt(systemInstruction = SYSTEM_INSTRUCTION, userParts = userParts)
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

  private fun buildWeatherFallbackMessages(dataModel: JsonElement?): List<ServerToClientMessage> {
    val root = dataModel as? JsonObject ?: return emptyList()
    if (!root.containsKey("current") && !root.containsKey("location")) return emptyList()

    val components = mutableListOf<ComponentInstance>()
    components.add(
      ComponentInstance(
        id = "root",
        type = "Column",
        props = buildJsonObject {
          put("children", explicitList("header", "statsCard"))
        }
      )
    )
    components.add(
      ComponentInstance(
        id = "header",
        type = "Column",
        props = buildJsonObject {
          put("children", explicitList("title", "location", "timestamp"))
        }
      )
    )
    components.add(textComponent("title", "Weather Update", usageHint = "h2"))
    components.add(textComponent("location", pathValue("/weather/location/name"), usageHint = "h4"))
    components.add(textComponent("timestamp", pathValue("/weather/current/time")))
    components.add(
      ComponentInstance(
        id = "statsCard",
        type = "Card",
        props = buildJsonObject {
          put("child", JsonPrimitive("statsColumn"))
        }
      )
    )
    components.add(
      ComponentInstance(
        id = "statsColumn",
        type = "Column",
        props = buildJsonObject {
          put(
            "children",
            explicitList(
              "tempRow",
              "feelsRow",
              "humidityRow",
              "windRow",
              "precipRow"
            )
          )
        }
      )
    )
    components.addAll(rowLabelValue("tempRow", "Temperature (C)", "/weather/current/temperatureC"))
    components.addAll(rowLabelValue("feelsRow", "Feels Like (C)", "/weather/current/apparentTemperatureC"))
    components.addAll(rowLabelValue("humidityRow", "Humidity (%)", "/weather/current/humidityPercent"))
    components.addAll(rowLabelValue("windRow", "Wind (km/h)", "/weather/current/windSpeedKmh"))
    components.addAll(rowLabelValue("precipRow", "Precip (mm)", "/weather/current/precipitationMm"))

    val surfaceUpdate = SurfaceUpdateMessage(surfaceId = "main", components = components)
    val begin = BeginRenderingMessage(surfaceId = "main", root = "root")
    return listOf(
      ServerToClientMessage(surfaceUpdate = surfaceUpdate),
      ServerToClientMessage(beginRendering = begin)
    )
  }

  private fun rowLabelValue(id: String, label: String, path: String): List<ComponentInstance> {
    val row = ComponentInstance(
      id = id,
      type = "Row",
      props = buildJsonObject {
        put("distribution", JsonPrimitive("spaceBetween"))
        put("alignment", JsonPrimitive("center"))
        put("children", explicitList("${id}_label", "${id}_value"))
      }
    )
    val labelComponent = textComponent("${id}_label", label)
    val valueComponent = textComponent("${id}_value", pathValue(path))
    return listOf(row, labelComponent, valueComponent)
  }

  private fun textComponent(
    id: String,
    text: String,
    usageHint: String? = null
  ): ComponentInstance {
    return textComponent(id, literalString(text), usageHint)
  }

  private fun textComponent(
    id: String,
    text: JsonObject,
    usageHint: String? = null
  ): ComponentInstance {
    return ComponentInstance(
      id = id,
      type = "Text",
      props = buildJsonObject {
        put("text", text)
        usageHint?.let { put("usageHint", JsonPrimitive(it)) }
      }
    )
  }

  private fun explicitList(vararg ids: String): JsonObject {
    return buildJsonObject {
      put("explicitList", buildJsonArray { ids.forEach { add(JsonPrimitive(it)) } })
    }
  }

  private fun literalString(value: String): JsonObject {
    return buildJsonObject { put("literalString", JsonPrimitive(value)) }
  }

  private fun pathValue(path: String): JsonObject {
    return buildJsonObject { put("path", JsonPrimitive(path)) }
  }

  private fun loadAssetJsonMinified(path: String): String {
    val raw = loadAssetText(path)
    val element = json.parseToJsonElement(raw)
    return json.encodeToString(JsonElement.serializer(), element)
  }

  private fun loadAssetText(path: String): String {
    return try {
      appContext.assets.open(path).bufferedReader().use { it.readText() }
    } catch (e: IOException) {
      throw IllegalStateException("Unable to load asset $path", e)
    }
  }

  private companion object {
    const val PROMPT_INTRO = """
You are creating a layout for a User Interface. It will be using a
format called A2UI which has several distinct schemas, each of which I will
provide to you. The user will be providing information about the UI they
would like to generate and your job is to create the JSON payloads as a
single array. Alternatively the user may provide a reference image and you
must try to understand it and match it as closely as possible.,

Here's everything you need:
"""

    const val PROMPT_REQUIREMENTS = """
Please return a valid A2UI Protocol Message object necessary to build the
user interface from scratch. If you choose to return multiple object you
must wrap them in an array and ensure there is a beginRendering message.
"""

    const val PROMPT_URL_RULES = """
If no data is provided create some. If there are any URLs you must
make them absolute and begin with a /. Nothing should ever be loaded from
a remote source
"""

    const val SYSTEM_INSTRUCTION = """
Please return a valid array
                        necessary to satisfy the user request. If no data is
                        provided create some. If there are any URLs you must
                        make them absolute and begin with a /.

                        Nothing should ever be loaded from a remote source.

                        You are working as part of an AI system, so no chit-chat and
                        no explaining what you're doing and why.DO NOT start with
                        "Okay", or "Alright" or any preambles. Just the output,
                        please.

                        ULTRA IMPORTANT: *Just* return the A2UI Protocol
                        Message object, do not wrap it in markdown. Just the object
                        please, nothing else!
"""
  }
}

private data class GeminiPrompt(
  val systemInstruction: String,
  val userParts: List<String>
)
