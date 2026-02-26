package com.samsung.genuicraft

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat

class RenderActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_RECORD_INDEX = "record_index"
    }

    private val appAssetBaseUrl = "https://appassets.androidplatform.net/"

    private val assetLoader by lazy {
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()
    }

    private lateinit var titleText: TextView
    private lateinit var subtitleText: TextView
    private lateinit var warningText: TextView
    private lateinit var warningContainer: View
    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_render)

        titleText = findViewById(R.id.renderTitle)
        subtitleText = findViewById(R.id.renderSubtitle)
        warningText = findViewById(R.id.warningText)
        warningContainer = findViewById(R.id.warningContainer)
        webView = findViewById(R.id.webView)

        setupWebView()
        renderRequestedItem()
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

    private fun renderRequestedItem() {
        val index = intent.getIntExtra(EXTRA_RECORD_INDEX, -1)
        val session = RenderSessionStore.current()
        val record = session?.records?.getOrNull(index)

        if (record == null) {
            showError(getString(R.string.error_session_missing))
            return
        }

        titleText.text = record.uiId ?: record.title
        subtitleText.text = getString(
            R.string.rendering_position,
            index + 1,
            session.records.size,
            record.sourceLabel
        )

        val result = GenUiHtmlRenderer.render(rawInput = record.rawJson, sourceDir = record.sourceDir)
        webView.loadDataWithBaseURL(appAssetBaseUrl, result.html, "text/html", "utf-8", null)

        if (result.warnings.isEmpty()) {
            warningContainer.visibility = View.GONE
        } else {
            warningContainer.visibility = View.VISIBLE
            warningText.text = result.warnings.joinToString(separator = "\n") { "- $it" }
        }
    }

    private fun showError(message: String) {
        titleText.text = getString(R.string.render_error_title)
        subtitleText.text = message
        warningContainer.visibility = View.GONE
        webView.loadDataWithBaseURL(
            appAssetBaseUrl,
            GenUiHtmlRenderer.renderErrorPage(message),
            "text/html",
            "utf-8",
            null
        )
    }
}

