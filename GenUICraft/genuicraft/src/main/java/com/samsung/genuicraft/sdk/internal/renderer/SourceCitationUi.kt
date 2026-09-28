package com.samsung.genuicraft.sdk.internal.renderer

import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.sdk.GenUiSource

/** Attribution belongs to one compiled answer, never to the renderer process or conversation. */
internal data class SourceCitationContext(
    val sources: List<GenUiSource>,
    val onPreview: (GenUiSource) -> Unit,
    val showSources: Boolean = true,
)

internal val LocalSourceCitations = staticCompositionLocalOf<SourceCitationContext?> { null }

internal fun sourceDisplayDomain(url: String): String =
    runCatching { Uri.parse(url).host?.removePrefix("www.").orEmpty() }
        .getOrDefault("")
        .ifBlank { url }

@Composable
internal fun SourceCitationDialog(
    source: GenUiSource?,
    onDismiss: () -> Unit,
    onOpenUrl: (String) -> Unit,
) {
    if (source == null) return
    val domain = sourceDisplayDomain(source.url)
    val maxBodyHeight = (LocalConfiguration.current.screenHeightDp * 0.5f).dp
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = source.title?.takeIf(String::isNotBlank) ?: domain,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = maxBodyHeight)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "[${source.id}] $domain",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                source.description?.takeIf(String::isNotBlank)?.let { description ->
                    Text(description, style = MaterialTheme.typography.bodyMedium)
                }
                Text(
                    text = source.url,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        },
        confirmButton = {
            TextButton(onClick = { onDismiss(); onOpenUrl(source.url) }) {
                Text("Open website")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        },
    )
}
