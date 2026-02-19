package com.a2ui.renderer

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ListView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File

class MainActivity : AppCompatActivity() {
    private data class LoadedItem(
        val title: String,
        val rawJson: String,
        val sourceDir: File?,
        val sourceLabel: String,
        val uiId: String?
    )

    private lateinit var pathInput: EditText
    private lateinit var statusText: TextView
    private lateinit var itemListView: ListView
    private lateinit var webView: WebView
    private lateinit var itemAdapter: ArrayAdapter<String>
    private val loadedItems = mutableListOf<LoadedItem>()
    private val appAssetBaseUrl = "https://appassets.androidplatform.net/"

    private val assetLoader by lazy {
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()
    }

    private val filePicker =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) {
                renderFromUri(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        pathInput = findViewById(R.id.pathInput)
        statusText = findViewById(R.id.statusText)
        itemListView = findViewById(R.id.itemListView)
        webView = findViewById(R.id.webView)
        val renderPathButton: Button = findViewById(R.id.renderPathButton)
        val pickFileButton: Button = findViewById(R.id.pickFileButton)
        val loadSampleButton: Button = findViewById(R.id.loadSampleButton)

        itemAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, mutableListOf())
        itemListView.adapter = itemAdapter
        itemListView.setOnItemClickListener { _, _, position, _ ->
            renderLoadedItem(position)
        }

        setupWebView()

        renderPathButton.setOnClickListener { renderFromPath() }
        pickFileButton.setOnClickListener {
            filePicker.launch(arrayOf("application/json", "text/plain", "*/*"))
        }
        loadSampleButton.setOnClickListener { renderBundledSample() }

        renderBundledSample()
    }

    private fun setupWebView() {
        webView.settings.javaScriptEnabled = false
        webView.settings.allowFileAccess = true
        webView.settings.allowContentAccess = true

        webView.webViewClient = object : WebViewClientCompat() {
            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest
            ): WebResourceResponse? {
                return assetLoader.shouldInterceptRequest(request.url)
            }

            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val uri = request.url
                val scheme = uri.scheme?.lowercase()
                if (scheme == "http" || scheme == "https") {
                    startActivity(Intent(Intent.ACTION_VIEW, uri))
                    return true
                }
                return false
            }
        }
    }

    private fun renderFromPath() {
        val path = pathInput.text?.toString()?.trim().orEmpty()
        if (path.isEmpty()) {
            showStatus(getString(R.string.error_missing_path), isError = true)
            return
        }

        try {
            val file = File(path)
            if (!file.exists() || !file.isFile) {
                showStatus(getString(R.string.error_path_not_found, path), isError = true)
                return
            }
            val raw = file.readText(Charsets.UTF_8)
            loadRecords(raw = raw, sourceDir = file.parentFile, sourceLabel = file.name)
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun renderFromUri(uri: Uri) {
        try {
            contentResolver.openInputStream(uri).use { input ->
                if (input == null) {
                    showStatus(getString(R.string.error_uri_open), isError = true)
                    return
                }
                val raw = input.bufferedReader(Charsets.UTF_8).use { it.readText() }
                pathInput.setText(uri.toString())
                loadRecords(raw = raw, sourceDir = null, sourceLabel = uri.toString())
            }
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun renderBundledSample() {
        try {
            val raw = assets.open("sample_genui.jsonl").bufferedReader(Charsets.UTF_8).use { it.readText() }
            pathInput.setText("")
            loadRecords(raw = raw, sourceDir = null, sourceLabel = "sample_genui.jsonl")
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun loadRecords(raw: String, sourceDir: File?, sourceLabel: String) {
        val records = parseRecords(raw, sourceDir, sourceLabel)
        if (records.isEmpty()) {
            showStatus("No valid JSON record found in $sourceLabel", isError = true)
            return
        }

        loadedItems.clear()
        loadedItems.addAll(records)

        itemAdapter.clear()
        itemAdapter.addAll(loadedItems.map { it.title })
        itemAdapter.notifyDataSetChanged()
        itemListView.visibility = if (loadedItems.size > 1) View.VISIBLE else View.GONE

        renderLoadedItem(0)
    }

    private fun renderLoadedItem(index: Int) {
        if (index !in loadedItems.indices) {
            return
        }
        val item = loadedItems[index]
        val result = GenUiHtmlRenderer.render(rawInput = item.rawJson, sourceDir = item.sourceDir)
        webView.loadDataWithBaseURL(appAssetBaseUrl, result.html, "text/html", "utf-8", null)

        val warningSuffix = if (result.warnings.isEmpty()) {
            ""
        } else {
            "\nWarnings:\n- " + result.warnings.joinToString("\n- ")
        }
        val idText = item.uiId ?: item.title
        val message =
            "Loaded ${loadedItems.size} item(s) from ${item.sourceLabel}. Rendering ${index + 1}/${loadedItems.size}: $idText"
        showStatus(message + warningSuffix, isError = false)
    }

    private fun showError(exc: Exception) {
        val msg = exc.message?.trim().takeUnless { it.isNullOrEmpty() } ?: exc.javaClass.simpleName
        val html = GenUiHtmlRenderer.renderErrorPage(msg)
        webView.loadDataWithBaseURL(appAssetBaseUrl, html, "text/html", "utf-8", null)
        showStatus(getString(R.string.error_render_failed, msg), isError = true)
    }

    private fun showStatus(message: String, isError: Boolean) {
        statusText.text = message
        statusText.setTextColor(if (isError) Color.parseColor("#B3261E") else Color.parseColor("#1F6F43"))
    }

    private fun parseRecords(raw: String, sourceDir: File?, sourceLabel: String): List<LoadedItem> {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) {
            return emptyList()
        }

        runCatching { JsonParser.parseString(trimmed) }.getOrNull()?.let { parsed ->
            return toLoadedItems(parsed, sourceDir, sourceLabel)
        }

        val items = mutableListOf<LoadedItem>()
        raw.lineSequence().forEachIndexed { index, line ->
            val lineTrimmed = line.trim()
            if (lineTrimmed.isEmpty()) {
                return@forEachIndexed
            }
            val parsed = runCatching { JsonParser.parseString(lineTrimmed) }.getOrNull() ?: return@forEachIndexed
            items += toLoadedItem(parsed, index + 1, sourceDir, sourceLabel)
        }
        return items
    }

    private fun toLoadedItems(
        parsed: JsonElement,
        sourceDir: File?,
        sourceLabel: String
    ): List<LoadedItem> {
        if (parsed.isJsonArray) {
            return parsed.asJsonArray.mapIndexed { index, element ->
                toLoadedItem(element, index + 1, sourceDir, sourceLabel)
            }
        }
        return listOf(toLoadedItem(parsed, 1, sourceDir, sourceLabel))
    }

    private fun toLoadedItem(
        element: JsonElement,
        index: Int,
        sourceDir: File?,
        sourceLabel: String
    ): LoadedItem {
        val obj = if (element.isJsonObject) element.asJsonObject else null
        val uiId = obj?.getString("ui_id")
            ?: obj?.getAsJsonArrayOrNull("genui_json")?.firstSurfaceId()
        val queryId = obj?.getString("query_id")
        val responseId = obj?.getString("response_id")

        val titleParts = mutableListOf<String>()
        titleParts += uiId ?: "Item $index"
        if (!queryId.isNullOrBlank()) {
            titleParts += queryId
        }
        if (!responseId.isNullOrBlank()) {
            titleParts += responseId
        }

        return LoadedItem(
            title = titleParts.joinToString(" | "),
            rawJson = element.toString(),
            sourceDir = sourceDir,
            sourceLabel = sourceLabel,
            uiId = uiId
        )
    }

    private fun JsonArray.firstSurfaceId(): String? {
        for (entry in this) {
            val msg = entry.asJsonObjectOrNull() ?: continue
            val create = msg.getAsJsonObjectOrNull("createSurface")
            val update = msg.getAsJsonObjectOrNull("updateComponents")
            val begin = msg.getAsJsonObjectOrNull("beginRendering")
            val surfaceId =
                create?.getString("surfaceId")
                    ?: update?.getString("surfaceId")
                    ?: begin?.getString("surfaceId")
            if (!surfaceId.isNullOrBlank()) {
                return surfaceId
            }
        }
        return null
    }

    private fun JsonElement.asJsonObjectOrNull(): JsonObject? =
        if (isJsonObject) asJsonObject else null

    private fun JsonObject.getAsJsonArrayOrNull(key: String) =
        get(key)?.takeIf { it.isJsonArray }?.asJsonArray

    private fun JsonObject.getAsJsonObjectOrNull(key: String) =
        get(key)?.takeIf { it.isJsonObject }?.asJsonObject

    private fun JsonObject.getString(key: String): String? {
        val value = get(key) ?: return null
        return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
    }
}
