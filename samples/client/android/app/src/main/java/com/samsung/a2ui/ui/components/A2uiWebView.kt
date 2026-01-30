package com.samsung.a2ui.ui.components

import android.annotation.SuppressLint
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebChromeClient
import android.webkit.WebViewClient
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.ui.unit.dp
import com.samsung.a2ui.LitPayload
import org.json.JSONObject

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun A2uiWebView(
  payload: LitPayload?,
  modifier: Modifier = Modifier
) {
  val latestPayload = rememberUpdatedState(payload)
  val pageLoaded = remember { mutableStateOf(false) }
  val rendererReady = remember { mutableStateOf(false) }
  val statusText = if (rendererReady.value) "Lit renderer ready" else "Lit renderer loading..."

  Box(modifier = modifier) {
    AndroidView(
      factory = { context ->
        WebView(context).apply {
          settings.javaScriptEnabled = true
          settings.domStorageEnabled = true
          settings.allowFileAccessFromFileURLs = true
          settings.allowUniversalAccessFromFileURLs = true
          addJavascriptInterface(
            A2uiWebViewBridge(
              onReady = {
                if (!rendererReady.value) {
                  Log.i(LOG_TAG, "Lit renderer ready.")
                  rendererReady.value = true
                }
              },
              onStatus = { status ->
                Log.i(LOG_TAG, "Renderer status: $status")
              },
              onError = { error ->
                Log.e(LOG_TAG, "Renderer error: $error")
              }
            ),
            "A2uiBridge"
          )
          webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(consoleMessage: ConsoleMessage): Boolean {
              Log.d(
                LOG_TAG,
                "Console: ${consoleMessage.message()} (${consoleMessage.sourceId()}:${consoleMessage.lineNumber()})"
              )
              return true
            }
          }
          webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) {
              pageLoaded.value = true
              latestPayload.value?.let { sendPayload(view, it) }
            }
          }
          loadUrl("file:///android_asset/a2ui_lit/index.html")
        }
      },
      update = { view ->
        if (pageLoaded.value) {
          latestPayload.value?.let { sendPayload(view, it) }
        }
      },
      modifier = Modifier.fillMaxSize()
    )
    Text(
      text = statusText,
      style = MaterialTheme.typography.labelSmall,
      color = MaterialTheme.colorScheme.onSurface,
      modifier = Modifier
        .align(Alignment.TopEnd)
        .padding(8.dp)
        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f))
        .padding(horizontal = 8.dp, vertical = 4.dp)
    )
  }
}

private fun sendPayload(view: WebView, payload: LitPayload) {
  val quotedJson = JSONObject.quote(payload.json)
  val js = "window.renderA2uiMessages($quotedJson, ${payload.reset});"
  Log.i(LOG_TAG, "Sending payload to WebView (len=${payload.json.length}, reset=${payload.reset})")
  view.evaluateJavascript(js) { result ->
    Log.d(LOG_TAG, "evaluateJavascript result: $result")
  }
}

private const val LOG_TAG = "A2UI-WebView"

private class A2uiWebViewBridge(
  private val onReady: () -> Unit,
  private val onStatus: (String) -> Unit,
  private val onError: (String) -> Unit
) {
  private val handler = Handler(Looper.getMainLooper())

  @JavascriptInterface
  fun onRendererReady() {
    handler.post { onReady() }
  }

  @JavascriptInterface
  fun onRendererStatus(message: String) {
    handler.post { onStatus(message) }
  }

  @JavascriptInterface
  fun onRendererError(message: String) {
    handler.post { onError(message) }
  }
}
