package com.samsung.genuicraft

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat

class RenderActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_RECORD_INDEX = "record_index"
        const val APP_ASSET_BASE_URL = "https://appassets.androidplatform.net/"
    }

    private val assetLoader by lazy {
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val index = intent.getIntExtra(EXTRA_RECORD_INDEX, -1)
        val session = RenderSessionStore.current()
        val record = session?.records?.getOrNull(index)

        setContent {
            GenUiCraftTheme {
                RenderScreen(
                    session = session,
                    record = record,
                    index = index,
                    assetLoader = assetLoader,
                    onOpenExternalUrl = { openExternalUrl(it) }
                )
            }
        }
    }

    private fun openExternalUrl(url: String) {
        val uri = runCatching { Uri.parse(url) }.getOrNull() ?: return
        val scheme = uri.scheme?.lowercase()
        if (scheme != "http" && scheme != "https") {
            return
        }
        startActivity(Intent(Intent.ACTION_VIEW, uri))
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun RenderScreen(
    session: RenderSessionStore.Session?,
    record: GenUiRecord?,
    index: Int,
    assetLoader: WebViewAssetLoader,
    onOpenExternalUrl: (String) -> Unit
) {
    val topBarTitle = record?.uiId ?: record?.title ?: stringResource(id = R.string.app_name)
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = MaterialTheme.colorScheme.background,
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = topBarTitle,
                        style = MaterialTheme.typography.headlineSmall
                    )
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(genUiBackgroundBrush())
                .padding(innerPadding)
                .consumeWindowInsets(innerPadding)
                .padding(horizontal = 12.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (session == null || record == null || index < 0) {
                Text(
                    text = stringResource(id = R.string.render_error_title),
                    style = MaterialTheme.typography.headlineLarge,
                    color = MaterialTheme.colorScheme.error
                )
                Text(
                    text = stringResource(id = R.string.error_session_missing),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                return@Column
            }

            val modeLabel = when (session.renderMode) {
                RenderMode.NATIVE -> stringResource(id = R.string.render_mode_native)
                RenderMode.WEB -> stringResource(id = R.string.render_mode_web)
            }

            Text(
                text = stringResource(
                    id = R.string.rendering_position,
                    index + 1,
                    session.records.size,
                    record.sourceLabel
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = modeLabel,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.primary
            )

            when (session.renderMode) {
                RenderMode.WEB -> {
                    val webResult = remember(record.rawJson, record.sourceDir) {
                        GenUiHtmlRenderer.render(rawInput = record.rawJson, sourceDir = record.sourceDir)
                    }

                    if (webResult.warnings.isNotEmpty()) {
                        WarningCard(warnings = webResult.warnings)
                    }

                    WebRenderPane(
                        html = webResult.html,
                        assetLoader = assetLoader,
                        onOpenExternalUrl = onOpenExternalUrl,
                        modifier = Modifier.weight(1f)
                    )
                }

                RenderMode.NATIVE -> {
                    val nativeResult = remember(record.rawJson, record.sourceDir) {
                        GenUiNativeRenderer.render(rawInput = record.rawJson, sourceDir = record.sourceDir)
                    }

                    if (nativeResult.warnings.isNotEmpty()) {
                        WarningCard(warnings = nativeResult.warnings)
                    }

                    GenUiNativeRenderer.Render(
                        result = nativeResult,
                        sourceDir = record.sourceDir,
                        onOpenExternalUrl = onOpenExternalUrl,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
private fun WarningCard(warnings: List<String>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
        colors = genUiCardColors(GenUiCardTone.Error),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = "Warnings",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.error
            )
            warnings.forEach { warning ->
                Text(
                    text = "- $warning",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun WebRenderPane(
    html: String,
    assetLoader: WebViewAssetLoader,
    onOpenExternalUrl: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    AndroidView(
        modifier = modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.background),
        factory = { context ->
            WebView(context).apply {
                settings.javaScriptEnabled = false
                settings.allowFileAccess = true
                settings.allowContentAccess = true
                setBackgroundColor(ContextCompat.getColor(context, R.color.sys_color_background))

                webViewClient = object : WebViewClientCompat() {
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
                            onOpenExternalUrl(uri.toString())
                            return true
                        }
                        return false
                    }
                }
            }
        },
        update = { webView ->
            webView.loadDataWithBaseURL(RenderActivity.APP_ASSET_BASE_URL, html, "text/html", "utf-8", null)
        }
    )
}
