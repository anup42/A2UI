package com.samsung.genuicraft

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.MotionEvent
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
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import com.samsung.genuicraft.security.SafeContentPolicy
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

class RenderActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_RECORD_INDEX = "record_index"
        const val APP_ASSET_BASE_URL = "https://appassets.androidplatform.net/"
    }

    private var currentIndex by mutableIntStateOf(-1)
    private var swipeStartX = 0f
    private var swipeStartY = 0f
    private var swipeTracking = false

    private val assetLoader by lazy {
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()

        val session = RenderSessionStore.current()
        currentIndex = resolveInitialIndex(
            requestedIndex = savedInstanceState?.getInt(EXTRA_RECORD_INDEX)
                ?: intent.getIntExtra(EXTRA_RECORD_INDEX, -1),
            session = session
        )

        setContent {
            val index = currentIndex
            val record = session?.records?.getOrNull(index)
            GenUiCraftTheme {
                RenderScreen(
                    session = session,
                    record = record,
                    index = index,
                    assetLoader = assetLoader,
                    onOpenExternalUrl = { openExternalUrl(it) },
                    onLoadHtmlAsset = { loadHtmlAsset(it) }
                )
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putInt(EXTRA_RECORD_INDEX, currentIndex)
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                swipeStartX = ev.x
                swipeStartY = ev.y
                swipeTracking = isNavigationEdgeSwipeStart(ev.x)
            }
            MotionEvent.ACTION_CANCEL -> {
                swipeTracking = false
            }
        }

        val handled = super.dispatchTouchEvent(ev)

        if (ev.actionMasked == MotionEvent.ACTION_UP && swipeTracking) {
            handleHorizontalSwipe(endX = ev.x, endY = ev.y)
            swipeTracking = false
        }
        return handled
    }

    private fun resolveInitialIndex(requestedIndex: Int, session: RenderSessionStore.Session?): Int {
        val lastIndex = session?.records?.lastIndex ?: return requestedIndex
        if (lastIndex < 0) return -1
        return requestedIndex.coerceIn(0, lastIndex)
    }

    private fun isNavigationEdgeSwipeStart(startX: Float): Boolean {
        val density = resources.displayMetrics.density
        val edgeWidthPx = max(36f * density, resources.displayMetrics.widthPixels * 0.075f)
        return startX <= edgeWidthPx || startX >= resources.displayMetrics.widthPixels - edgeWidthPx
    }

    private fun handleHorizontalSwipe(endX: Float, endY: Float) {
        val session = RenderSessionStore.current() ?: return
        if (session.records.size <= 1 || currentIndex !in session.records.indices) return

        val dx = endX - swipeStartX
        val dy = endY - swipeStartY
        val density = resources.displayMetrics.density
        val minDistancePx = max(72f * density, resources.displayMetrics.widthPixels * 0.16f)
        val horizontalDominates = abs(dx) > abs(dy) * 1.35f
        if (abs(dx) < minDistancePx || !horizontalDominates) return

        val nextIndex = if (dx < 0f) {
            (currentIndex + 1).coerceAtMost(session.records.lastIndex)
        } else {
            (currentIndex - 1).coerceAtLeast(0)
        }
        if (nextIndex != currentIndex) {
            currentIndex = nextIndex
        }
    }

    private fun openExternalUrl(url: String) {
        val safeUrl = SafeContentPolicy.sanitizeActionUrl(url) ?: return
        val uri = runCatching { Uri.parse(safeUrl) }.getOrNull() ?: return
        val action = if (uri.scheme.equals("tel", ignoreCase = true)) {
            Intent.ACTION_DIAL
        } else {
            Intent.ACTION_VIEW
        }
        startActivity(Intent(action, uri))
    }

    private fun loadHtmlAsset(assetPath: String): String =
        assets.open(assetPath).bufferedReader(Charsets.UTF_8).use { it.readText() }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun RenderScreen(
    session: RenderSessionStore.Session?,
    record: GenUiRecord?,
    index: Int,
    assetLoader: WebViewAssetLoader,
    onOpenExternalUrl: (String) -> Unit,
    onLoadHtmlAsset: (String) -> String
) {
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 12.dp
        DeviceSizeClass.Medium -> 18.dp
        DeviceSizeClass.Expanded -> 24.dp
    }

    val topBarTitle = record?.uiId ?: record?.title ?: stringResource(id = R.string.app_name)
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
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
                    containerColor = genUiTopBarContainerColor(),
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        }
    ) { innerPadding ->
        GenUiScreenBackground(
            modifier = Modifier.fillMaxSize()
        ) { backgroundModifier ->
            Column(
                modifier = backgroundModifier
                    .padding(innerPadding)
                    .consumeWindowInsets(innerPadding)
                    .padding(horizontal = horizontalPadding, vertical = 14.dp),
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
                val positionLabel = stringResource(
                    id = R.string.rendering_position,
                    index + 1,
                    session.records.size,
                    record.sourceLabel
                )

                when (session.renderMode) {
                    RenderMode.WEB -> {
                        RenderMetadataHeader(
                            positionLabel = positionLabel,
                            modeLabel = modeLabel
                        )

                        val baseHtml = remember(record.rawJson, record.sourceDir, record.htmlAssetPath) {
                            val htmlAssetPath = record.htmlAssetPath
                            if (!htmlAssetPath.isNullOrBlank()) {
                                runCatching { onLoadHtmlAsset(htmlAssetPath) }
                                    .getOrElse { exc ->
                                        renderHtmlAssetError(
                                            path = htmlAssetPath,
                                            message = exc.message ?: exc.javaClass.simpleName
                                        )
                                    }
                            } else {
                                GenUiHtmlRenderer.render(
                                    rawInput = record.rawJson,
                                    sourceDir = record.sourceDir
                                ).html
                            }
                        }
                        val colorScheme = MaterialTheme.colorScheme
                        val themedHtml = remember(
                            baseHtml,
                            colorScheme.background,
                            colorScheme.surface,
                            colorScheme.surfaceContainerLowest,
                            colorScheme.surfaceContainerLow,
                            colorScheme.surfaceContainer,
                            colorScheme.surfaceContainerHigh,
                            colorScheme.surfaceVariant,
                            colorScheme.onSurface,
                            colorScheme.onSurfaceVariant,
                            colorScheme.primary,
                            colorScheme.primaryContainer,
                            colorScheme.secondary,
                            colorScheme.secondaryContainer,
                            colorScheme.tertiary,
                            colorScheme.tertiaryContainer,
                            colorScheme.outline,
                            colorScheme.outlineVariant,
                            colorScheme.error,
                            deviceConfig.screenWidthDp,
                            deviceConfig.screenHeightDp,
                            deviceConfig.smallestWidthDp,
                            deviceConfig.fontScale,
                            deviceConfig.isLandscape,
                            deviceConfig.localeTag,
                            deviceConfig.layoutDirection,
                            deviceConfig.widthClass,
                            deviceConfig.heightClass
                        ) {
                            applyDynamicHtmlPalette(baseHtml, colorScheme, deviceConfig)
                        }

                        WebRenderPane(
                            html = themedHtml,
                            assetLoader = assetLoader,
                            onOpenExternalUrl = onOpenExternalUrl,
                            deviceConfig = deviceConfig,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    RenderMode.NATIVE -> {
                        val nativeResult = remember(record.rawJson, record.sourceDir) {
                            GenUiNativeRenderer.render(rawInput = record.rawJson, sourceDir = record.sourceDir)
                        }

                        GenUiNativeRenderer.Render(
                            result = nativeResult,
                            sourceDir = record.sourceDir,
                            onOpenExternalUrl = onOpenExternalUrl,
                            modifier = Modifier.weight(1f),
                            headerContent = {
                                RenderMetadataHeader(
                                    positionLabel = positionLabel,
                                    modeLabel = modeLabel
                                )
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun RenderMetadataHeader(
    positionLabel: String,
    modeLabel: String
) {
    Text(
        text = positionLabel,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
    Text(
        text = modeLabel,
        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
        color = MaterialTheme.colorScheme.primary
    )
}

@Composable
private fun WebRenderPane(
    html: String,
    assetLoader: WebViewAssetLoader,
    onOpenExternalUrl: (String) -> Unit,
    deviceConfig: DeviceUiConfig,
    modifier: Modifier = Modifier
) {
    AndroidView(
        modifier = modifier
            .fillMaxWidth()
            .background(Color.Transparent),
        factory = { context ->
            WebView(context).apply {
                settings.javaScriptEnabled = false
                settings.allowFileAccess = true
                settings.allowContentAccess = true
                settings.useWideViewPort = true
                settings.loadWithOverviewMode = true
                settings.builtInZoomControls = false
                settings.displayZoomControls = false
                settings.textZoom = deviceConfig.webViewTextZoomPercent()
                setBackgroundColor(android.graphics.Color.TRANSPARENT)

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
            webView.settings.textZoom = deviceConfig.webViewTextZoomPercent()
            webView.setBackgroundColor(android.graphics.Color.TRANSPARENT)
            webView.loadDataWithBaseURL(RenderActivity.APP_ASSET_BASE_URL, html, "text/html", "utf-8", null)
        }
    )
}

private fun applyDynamicHtmlPalette(
    html: String,
    colorScheme: ColorScheme,
    deviceConfig: DeviceUiConfig
): String {
    val dark = colorScheme.background.luminance() < 0.5f
    val background = colorScheme.surfaceContainerLowest
    val backgroundVariant = lerp(
        colorScheme.surfaceContainerLow,
        colorScheme.primaryContainer,
        if (dark) 0.18f else 0.12f
    )
    val surface = colorScheme.surfaceContainerHigh.toCssRgba(if (dark) 0.58f else 0.52f)
    val surfaceLow = colorScheme.surfaceContainerLow.toCssRgba(if (dark) 0.54f else 0.48f)
    val surfaceContainer = colorScheme.surfaceContainer.toCssRgba(if (dark) 0.58f else 0.52f)
    val surfaceHigh = colorScheme.surfaceContainerHighest.toCssRgba(if (dark) 0.62f else 0.56f)
    val toneOnTone = colorScheme.onSurface.toCssRgba(if (dark) 0.10f else 0.06f)
    val toneOnToneHigh = colorScheme.onSurface.toCssRgba(if (dark) 0.14f else 0.10f)
    val tableSurface = colorScheme.surfaceContainerHigh.toCssRgba(if (dark) 0.56f else 0.50f)
    val tableStripe = colorScheme.onSurface.toCssRgba(if (dark) 0.040f else 0.026f)
    val tableHeader = colorScheme.onSurface.toCssRgba(if (dark) 0.070f else 0.040f)
    val layoutDirection = if (deviceConfig.layoutDirection == LayoutDirection.Rtl) "rtl" else "ltr"
    val orientation = if (deviceConfig.isLandscape) "landscape" else "portrait"
    val bodyPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 10
        DeviceSizeClass.Medium -> 14
        DeviceSizeClass.Expanded -> 18
    }
    val tableMinWidth = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> max(340, deviceConfig.screenWidthDp + 18)
        DeviceSizeClass.Medium -> 460
        DeviceSizeClass.Expanded -> 580
    }
    val webTextZoom = deviceConfig.webViewTextZoomPercent()

    val overrideStyle = """
        <style id="genuicraft-dynamic-palette">
          :root {
            --device-screen-width-dp: ${deviceConfig.screenWidthDp};
            --device-screen-height-dp: ${deviceConfig.screenHeightDp};
            --device-smallest-width-dp: ${deviceConfig.smallestWidthDp};
            --device-density: ${deviceConfig.density.toCssFloat()};
            --device-font-scale: ${deviceConfig.fontScale.toCssFloat()};
            --device-text-zoom: ${webTextZoom}%;
            --device-locale: "${deviceConfig.localeTag}";
            --device-layout-direction: ${layoutDirection};
            --device-orientation: ${orientation};
            --device-width-class: ${deviceConfig.widthClass.cssToken()};
            --device-height-class: ${deviceConfig.heightClass.cssToken()};

            --sys-color-background: ${background.toCssRgba(if (dark) 0.20f else 0.28f)};
            --sys-color-background-variant: ${backgroundVariant.toCssRgba(if (dark) 0.24f else 0.32f)};
            --sys-color-surface: $surface;
            --sys-color-surface-bright: $surface;
            --sys-color-surface-brightest: $surface;
            --sys-color-surface-variant: ${colorScheme.surfaceVariant.toCssRgba(if (dark) 0.44f else 0.56f)};
            --sys-color-surface-fixed: $surfaceHigh;
            --sys-color-surface-fixed-variant: ${colorScheme.onSurface.toCssRgba(if (dark) 0.78f else 0.84f)};

            --sys-color-surface-container-lowest: ${background.toCssRgba(if (dark) 0.32f else 0.40f)};
            --sys-color-surface-container-low: $surfaceLow;
            --sys-color-surface-container: $surfaceContainer;
            --sys-color-surface-container-high: $surfaceHigh;
            --sys-color-surface-container-higher: $surfaceHigh;
            --sys-color-surface-container-highest: $surfaceHigh;
            --sys-color-surface-container-fixed: $surface;
            --sys-color-surface-container-fixed-variant: ${colorScheme.onSurface.toCssRgba(if (dark) 0.78f else 0.84f)};

            --sys-color-tone-on-tone: $toneOnTone;
            --sys-color-tone-on-tone-high: $toneOnToneHigh;

            --sys-color-on-surface-container-highest: ${colorScheme.onSurface.toCssHex()};
            --sys-color-on-surface-container-high: ${colorScheme.onSurface.toCssHex()};
            --sys-color-on-surface-container: ${colorScheme.onSurfaceVariant.toCssHex()};
            --sys-color-on-surface-container-low: ${colorScheme.onSurfaceVariant.toCssHex()};
            --sys-color-on-surface-container-lowest: ${colorScheme.onSurfaceVariant.toCssHex()};

            --sys-color-outline: ${colorScheme.outline.toCssHex()};
            --sys-color-outline-low: ${colorScheme.outlineVariant.toCssHex()};
            --sys-color-outline-high: ${colorScheme.outline.toCssHex()};
            --sys-color-outline-highest: ${colorScheme.outline.toCssHex()};
            --sys-color-media-border: ${if (dark) "#636368" else "#B7B7BB"};

            --sys-color-primary: ${colorScheme.primary.toCssHex()};
            --sys-color-primary-high: ${colorScheme.primary.toCssHex()};
            --sys-color-primary-bright: ${colorScheme.primaryContainer.toCssHex()};
            --sys-color-functional-red: ${colorScheme.error.toCssHex()};
            --sys-color-functional-red-bright: ${colorScheme.error.toCssHex()};
            --sys-color-functional-green: ${colorScheme.secondary.toCssHex()};
            --sys-color-functional-green-bright: ${colorScheme.secondaryContainer.toCssHex()};
            --sys-color-functional-orange: ${colorScheme.tertiary.toCssHex()};
            --sys-color-functional-orange-bright: ${colorScheme.tertiaryContainer.toCssHex()};
          }

          html {
            direction: ${layoutDirection} !important;
            -webkit-text-size-adjust: ${webTextZoom}% !important;
            text-size-adjust: ${webTextZoom}% !important;
          }

          body {
            padding: ${bodyPadding}px !important;
          }

          body::before {
            opacity: 0.08 !important;
          }

          body::after {
            opacity: 0.06 !important;
          }

          .surface,
          .card,
          .booking-card,
          .media-card,
          .table-wrap {
            -webkit-backdrop-filter: blur(18px) saturate(1.08);
            backdrop-filter: blur(18px) saturate(1.08);
          }

          .table-wrap {
            background: $tableSurface !important;
            border-color: ${colorScheme.outline.toCssRgba(if (dark) 0.78f else 0.84f)} !important;
          }

          .data-table {
            background: transparent !important;
            min-width: ${tableMinWidth}px !important;
          }

          .data-table thead th {
            background: $tableHeader !important;
          }

          .data-table tbody tr:nth-child(even) {
            background: $tableStripe !important;
          }

          .data-table tbody tr:nth-child(odd) {
            background: transparent !important;
          }
        </style>
    """.trimIndent()

    return if (html.contains("</head>", ignoreCase = true)) {
        html.replace("</head>", "$overrideStyle\n</head>", ignoreCase = true)
    } else {
        html + overrideStyle
    }
}

private fun renderHtmlAssetError(path: String, message: String): String {
    val safePath = path
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    val safeMessage = message
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    return """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body { margin: 0; padding: 18px; font-family: sans-serif; color: #f8fafc; background: transparent; }
            .card { border: 1px solid rgba(255,255,255,.18); border-radius: 18px; padding: 16px; background: rgba(15,23,42,.78); }
            h1 { font-size: 20px; margin: 0 0 10px; }
            p { color: #cbd5e1; line-height: 1.5; }
            code { color: #93c5fd; }
          </style>
        </head>
        <body>
          <section class="card">
            <h1>HTML preview unavailable</h1>
            <p>Could not load <code>$safePath</code>.</p>
            <p>$safeMessage</p>
          </section>
        </body>
        </html>
    """.trimIndent()
}

private fun Color.toCssHex(): String {
    val r = (red * 255f).roundToInt().coerceIn(0, 255)
    val g = (green * 255f).roundToInt().coerceIn(0, 255)
    val b = (blue * 255f).roundToInt().coerceIn(0, 255)
    return String.format("#%02X%02X%02X", r, g, b)
}

private fun Color.toCssRgba(alphaValue: Float): String {
    val r = (red * 255f).roundToInt().coerceIn(0, 255)
    val g = (green * 255f).roundToInt().coerceIn(0, 255)
    val b = (blue * 255f).roundToInt().coerceIn(0, 255)
    val a = alphaValue.coerceIn(0f, 1f)
    return "rgba($r, $g, $b, $a)"
}

private fun Float.toCssFloat(): String {
    return String.format(Locale.US, "%.2f", this)
}
