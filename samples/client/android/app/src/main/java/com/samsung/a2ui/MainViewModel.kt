package com.samsung.a2ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import android.util.Log
import com.samsung.a2ui.a2ui.GeminiA2uiClient
import com.samsung.a2ui.a2ui.GeminiA2uiResult
import com.samsung.a2ui.a2ui.A2uiJsonParser
import com.samsung.a2ui.a2ui.A2uiJsonWriter
import com.samsung.a2ui.a2ui.A2uiMessageProcessor
import com.samsung.a2ui.a2ui.A2aHttpClient
import com.samsung.a2ui.a2ui.ComponentNode
import com.samsung.a2ui.a2ui.ServerToClientMessage
import com.samsung.a2ui.a2ui.Surface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class MainViewModel(application: Application) : AndroidViewModel(application) {
  private val processor = A2uiMessageProcessor()
  private val client = A2aHttpClient()
  private val geminiClient = GeminiA2uiClient(application.applicationContext)
  private val logTag = "A2UI-VM"
  private val cachePrefs = application.getSharedPreferences("a2ui_cache", Application.MODE_PRIVATE)
  private val geminiPromptCache = LinkedHashMap<String, List<ServerToClientMessage>>()
  private val maxCacheEntries = 20
  private var lastGeminiDebug: GeminiA2uiResult? = null

  val messageProcessor: A2uiMessageProcessor
    get() = processor

  var serverUrl by mutableStateOf("http://10.0.2.2:10002")
    private set

  var backendMode by mutableStateOf(BackendMode.GEMINI_DIRECT)
    private set

  var rendererMode by mutableStateOf(RendererMode.LIT_WEBVIEW)
    private set

  // Supply the provider key at runtime; never commit credentials in the sample app.
  var geminiApiKey by mutableStateOf("")
    private set

  var geminiModel by mutableStateOf("gemini-2.5-flash-lite")
    private set

  var isLoading by mutableStateOf(false)
    private set

  var errorMessage by mutableStateOf<String?>(null)
    private set

  var cacheStatus by mutableStateOf<String?>(null)
    private set

  var surfaces by mutableStateOf<Map<String, Surface>>(emptyMap())
    private set

  init {
    loadCache()
  }

  var litPayload by mutableStateOf<LitPayload?>(null)
    private set

  fun updateServerUrl(url: String) {
    serverUrl = url.trim()
  }

  fun updateBackendMode(mode: BackendMode) {
    backendMode = mode
    if (mode != BackendMode.GEMINI_DIRECT) {
      rendererMode = RendererMode.NATIVE
    }
    Log.i(logTag, "Backend mode set to $mode")
  }

  fun updateRendererMode(mode: RendererMode) {
    rendererMode = mode
    Log.i(logTag, "Renderer mode set to $mode")
  }

  fun updateGeminiApiKey(key: String) {
    geminiApiKey = key.trim()
    Log.i(logTag, "Gemini API key updated (length=${geminiApiKey.length})")
  }

  fun updateGeminiModel(model: String) {
    geminiModel = model
    Log.i(logTag, "Gemini model set to \"$geminiModel\"")
  }

  fun applyLocalInput(rawInput: String) {
    val messages = A2uiJsonParser.parseMessages(rawInput)
    if (messages.isEmpty()) {
      errorMessage = "No A2UI messages found in the input."
      Log.w(logTag, "Local JSON parse produced no messages.")
      return
    }
    applyMessages(messages, reset = true)
  }

  fun submitInput(rawInput: String) {
    val input = rawInput.trim()
    if (input.isBlank()) {
      errorMessage = "Enter a prompt or A2UI JSON."
      Log.w(logTag, "submitInput called with blank input.")
      return
    }

    val isJson = A2uiJsonParser.looksLikeJson(input)
    if (!isJson && backendMode == BackendMode.A2A_SERVER && serverUrl.isBlank()) {
      errorMessage = "Set a server URL or switch to Gemini Direct."
      Log.w(logTag, "Missing server URL for A2A backend.")
      return
    }
    if (!isJson && backendMode == BackendMode.GEMINI_DIRECT && geminiApiKey.isBlank()) {
      errorMessage = "Enter a Gemini API key to use Gemini Direct."
      Log.w(logTag, "Missing Gemini API key.")
      return
    }
    if (!isJson && backendMode == BackendMode.GEMINI_DIRECT && geminiModel.isBlank()) {
      errorMessage = "Enter a Gemini model name."
      Log.w(logTag, "Missing Gemini model name.")
      return
    }

    errorMessage = null
    cacheStatus = null
    isLoading = true
    Log.i(
      logTag,
      "submitInput backend=$backendMode isJson=$isJson inputLength=${input.length} serverUrl=$serverUrl model=$geminiModel"
    )

    if (!isJson && backendMode == BackendMode.GEMINI_DIRECT) {
      val cacheKey = cacheKeyForPrompt(input)
      val cached = geminiPromptCache[cacheKey]
      if (cached != null) {
        Log.i(logTag, "Cache hit for prompt length=${input.length}.")
        cacheStatus = "Using cache"
        touchCacheEntry(cacheKey, cached)
        applyMessages(cached, reset = true)
        isLoading = false
        return
      }
    }

    viewModelScope.launch {
      val result = withContext(Dispatchers.IO) {
        when {
          isJson -> UiResult(A2uiJsonParser.parseMessages(input))
          backendMode == BackendMode.GEMINI_DIRECT ->
            fromGeminiResult(geminiClient.sendPrompt(geminiApiKey, geminiModel.trim(), input))
          else -> UiResult(client.sendTextPrompt(serverUrl, input))
        }
      }

      val hasMessages = result.messages.isNotEmpty()
      if (!result.error.isNullOrBlank()) {
        errorMessage = result.error
        Log.w(logTag, "submitInput error: ${result.error}")
      } else if (!hasMessages) {
        errorMessage = "No A2UI messages returned."
        Log.w(logTag, "submitInput returned zero messages.")
      }

      if (hasMessages) {
        Log.i(logTag, "submitInput received ${result.messages.size} messages.")
        applyMessages(result.messages, reset = true)
        if (!isJson && backendMode == BackendMode.GEMINI_DIRECT) {
          saveToCache(cacheKeyForPrompt(input), result.messages)
        }
      }

      isLoading = false
    }
  }

  fun sendUserAction(message: Map<String, Any?>) {
    if (backendMode == BackendMode.A2A_SERVER && serverUrl.isBlank()) {
      errorMessage = "Set a server URL to send actions."
      Log.w(logTag, "sendUserAction missing server URL.")
      return
    }
    if (backendMode == BackendMode.GEMINI_DIRECT && geminiApiKey.isBlank()) {
      errorMessage = "Enter a Gemini API key to send actions."
      Log.w(logTag, "sendUserAction missing Gemini API key.")
      return
    }
    if (backendMode == BackendMode.GEMINI_DIRECT && geminiModel.isBlank()) {
      errorMessage = "Enter a Gemini model name."
      Log.w(logTag, "sendUserAction missing Gemini model.")
      return
    }

    errorMessage = null
    isLoading = true
    Log.i(logTag, "sendUserAction backend=$backendMode")

    viewModelScope.launch {
      val result = withContext(Dispatchers.IO) {
        when (backendMode) {
          BackendMode.GEMINI_DIRECT -> fromGeminiResult(
            geminiClient.sendUserAction(geminiApiKey, geminiModel.trim(), message)
          )
          else -> UiResult(client.sendUserAction(serverUrl, message))
        }
      }
      val hasMessages = result.messages.isNotEmpty()
      if (!result.error.isNullOrBlank()) {
        errorMessage = result.error
        Log.w(logTag, "sendUserAction error: ${result.error}")
      }
      if (hasMessages) {
        Log.i(logTag, "sendUserAction received ${result.messages.size} messages.")
        applyMessages(result.messages, reset = false)
      } else if (result.error.isNullOrBlank()) {
        Log.w(logTag, "sendUserAction returned zero messages.")
      }
      isLoading = false
    }
  }

  fun updateBoundValue(node: ComponentNode, path: String, value: Any?, surfaceId: String) {
    processor.setData(node, path, value, surfaceId)
    surfaces = processor.getSurfaces().toMap()
  }

  private fun applyMessages(messages: List<ServerToClientMessage>, reset: Boolean) {
    if (reset) {
      processor.clearSurfaces()
      Log.i(logTag, "Surfaces cleared before applying messages.")
    }
    processor.processMessages(messages)
    surfaces = processor.getSurfaces().toMap()
    val uiJson = A2uiJsonWriter.toJson(messages)
    litPayload = LitPayload(
      json = uiJson,
      reset = reset,
      nonce = System.nanoTime()
    )
    Log.i(logTag, "Surfaces after apply: ${surfaces.keys}")
    lastGeminiDebug?.let { debug ->
      GeminiDebugStore.update(debug.requestJson, debug.rawResponse, uiJson)
      lastGeminiDebug = null
    }
  }

  private fun fromGeminiResult(result: GeminiA2uiResult): UiResult {
    lastGeminiDebug = result
    GeminiDebugStore.update(result.requestJson, result.rawResponse, null)
    return UiResult(result.messages, result.error)
  }

  fun clearCacheForPrompt(prompt: String) {
    val trimmed = prompt.trim()
    if (trimmed.isBlank()) {
      cacheStatus = "Enter a prompt to clear cache."
      return
    }
    val key = cacheKeyForPrompt(trimmed)
    val removed = geminiPromptCache.remove(key)
    cacheStatus = if (removed != null) {
      Log.i(logTag, "Cache cleared for prompt length=${trimmed.length}.")
      "Cache cleared"
    } else {
      Log.i(logTag, "No cache entry for prompt length=${trimmed.length}.")
      "No cache entry"
    }
    persistCache()
  }

  private fun cacheKeyForPrompt(prompt: String): String {
    return "${geminiModel.trim()}|${prompt.trim()}"
  }

  private fun loadCache() {
    val raw = cachePrefs.getString(CACHE_PREF_KEY, null) ?: return
    try {
      val root = JSONObject(raw)
      val entries = root.optJSONArray("entries") ?: return
      for (i in 0 until entries.length()) {
        val entry = entries.optJSONObject(i) ?: continue
        val key = entry.optString("key", "")
        val payload = entry.optString("payload", "")
        if (key.isBlank() || payload.isBlank()) continue
        val messages = A2uiJsonParser.parseMessages(payload)
        if (messages.isNotEmpty()) {
          geminiPromptCache[key] = messages
        }
      }
      Log.i(logTag, "Loaded ${geminiPromptCache.size} cached prompts.")
    } catch (e: Exception) {
      Log.w(logTag, "Failed to load cache: ${e.message}")
    }
  }

  private fun saveToCache(key: String, messages: List<ServerToClientMessage>) {
    geminiPromptCache.remove(key)
    geminiPromptCache[key] = messages
    trimCache()
    persistCache()
  }

  private fun touchCacheEntry(key: String, messages: List<ServerToClientMessage>) {
    geminiPromptCache.remove(key)
    geminiPromptCache[key] = messages
    persistCache()
  }

  private fun trimCache() {
    while (geminiPromptCache.size > maxCacheEntries) {
      val oldestKey = geminiPromptCache.keys.firstOrNull() ?: break
      geminiPromptCache.remove(oldestKey)
    }
  }

  private fun persistCache() {
    val entries = JSONArray()
    geminiPromptCache.forEach { (key, messages) ->
      val payload = A2uiJsonWriter.toJson(messages)
      val entry = JSONObject()
      entry.put("key", key)
      entry.put("payload", payload)
      entries.put(entry)
    }
    val root = JSONObject()
    root.put("entries", entries)
    cachePrefs.edit().putString(CACHE_PREF_KEY, root.toString()).apply()
  }

  companion object {
    private const val CACHE_PREF_KEY = "gemini_prompt_cache"
  }
}

enum class BackendMode {
  A2A_SERVER,
  GEMINI_DIRECT
}

enum class RendererMode {
  NATIVE,
  LIT_WEBVIEW
}

private data class UiResult(
  val messages: List<ServerToClientMessage>,
  val error: String? = null
)

data class LitPayload(
  val json: String,
  val reset: Boolean,
  val nonce: Long
)
