package com.samsung.genuicraft.sdk

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import com.google.gson.Gson
import com.samsung.genuicraft.sdk.internal.renderer.FlatDiagnostic
import com.samsung.genuicraft.sdk.internal.renderer.FlatRenderEvent
import com.samsung.genuicraft.sdk.internal.renderer.FlatRendererHost
import com.samsung.genuicraft.sdk.internal.renderer.FlatSpecContent
import com.samsung.genuicraft.sdk.internal.renderer.GenUiNativeRenderer
import com.samsung.genuicraft.sdk.internal.renderer.LocalFlatSpecTextHorizontalPadding
import com.samsung.genuicraft.sdk.internal.renderer.LocalSourceCitations
import com.samsung.genuicraft.sdk.internal.renderer.SourceCitationContext
import com.samsung.genuicraft.sdk.internal.renderer.SourceCitationDialog
import com.samsung.genuicraft.sdk.internal.theme.GenUiRendererTheme

/** Renders a compiled document without starting a model or provider. */
@Composable
fun GenUiContent(
    document: GenUiDocument,
    modifier: Modifier = Modifier,
    onAction: (GenUiAction) -> Unit = {},
) {
    GenUiContent(document, modifier, onAction, showSources = true)
}

/** Renders a document with optional host-owned source disclosure. */
@Composable
fun GenUiContent(
    document: GenUiDocument,
    modifier: Modifier = Modifier,
    onAction: (GenUiAction) -> Unit = {},
    showSources: Boolean,
) = GenUiContentImpl(document, modifier, onAction, showSources)

@Composable
internal fun GenUiContentImpl(
    document: GenUiDocument,
    modifier: Modifier = Modifier,
    onAction: (GenUiAction) -> Unit = {},
    showSources: Boolean = true,
    onSourcePreviewChanged: (Boolean) -> Unit = {},
    embeddedMode: Boolean = false,
) {
    val renderResult = remember(document.a2uiJson, document.express) {
        val input = document.a2uiJson.ifBlank { document.express }
        GenUiNativeRenderer.render(input, sourceDir = null)
    }
    val callbackHost = remember(onAction) { CallbackRendererHost(onAction) }
    val sources = remember(document.express, document.a2uiJson) { SourceAttribution.read(document) }
    var previewSource by remember(document.express, document.a2uiJson) {
        mutableStateOf<GenUiSource?>(null)
    }
    val citationContext = SourceCitationContext(
        sources = sources,
        onPreview = { source ->
            onSourcePreviewChanged(true)
            previewSource = source
        },
        showSources = showSources,
    )

    GenUiRendererTheme {
        val contentPadding = if (embeddedMode) 4.dp else 16.dp
        if (renderResult.errorMessage != null) {
            Text(
                text = "Unable to render GenUI: ${renderResult.errorMessage}",
                color = MaterialTheme.colorScheme.error,
                modifier = modifier.padding(horizontal = if (embeddedMode) 0.dp else 16.dp, vertical = contentPadding),
            )
            return@GenUiRendererTheme
        }

        Surface(
            modifier = modifier.fillMaxWidth(),
            color = if (embeddedMode) Color.Transparent else MaterialTheme.colorScheme.background,
        ) {
            CompositionLocalProvider(
                LocalFlatSpecTextHorizontalPadding provides 0.dp,
                LocalSourceCitations provides citationContext,
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(
                        horizontal = if (embeddedMode) 0.dp else 16.dp,
                        vertical = contentPadding,
                    ),
                ) {
                    renderResult.surfaces.forEach { surface ->
                        val spec = surface.canonicalSpec ?: return@forEach
                        FlatSpecContent(
                            spec = spec,
                            host = callbackHost,
                            onEvent = { event -> dispatchExternalEvent(event, onAction) },
                            collapseRootHorizontalPadding = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
            }
        }
        SourceCitationDialog(
            source = previewSource,
            onDismiss = {
                previewSource = null
                onSourcePreviewChanged(false)
            },
            onOpenUrl = callbackHost::openUrl,
        )
    }
}

internal class CallbackRendererHost(
    private val callback: (GenUiAction) -> Unit,
) : FlatRendererHost {
    override fun openUrl(url: String) {
        callback(GenUiAction(name = "openUrl", parameters = mapOf("url" to url)))
    }

    override val imageLoader: ImageLoader? = null

    override fun onDiagnostic(diagnostic: FlatDiagnostic) = Unit
}

internal fun dispatchExternalEvent(
    event: FlatRenderEvent,
    callback: (GenUiAction) -> Unit,
) {
    if (
        event.kind != FlatRenderEvent.Kind.ACTION ||
        event.action != "emitEvent" ||
        event.success != true
    ) {
        return
    }
    val eventName = event.params["name"]?.toString()?.trim().orEmpty()
    if (eventName.isEmpty()) return
    callback(
        GenUiAction(
            name = eventName,
            parameters = event.params
                .filterKeys { it != "name" }
                .mapValues { (_, value) -> actionParameter(value) },
        )
    )
}

private fun actionParameter(value: Any?): String = when (value) {
    null -> ""
    is String -> value
    is Number, is Boolean, is Char -> value.toString()
    else -> Gson().toJson(value)
}
