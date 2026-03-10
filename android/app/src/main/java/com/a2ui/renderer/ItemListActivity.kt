package com.samsung.genuicraft

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
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
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.LocalOverscrollConfiguration
import androidx.compose.foundation.interaction.MutableInteractionSource

class ItemListActivity : AppCompatActivity() {
    private var session by mutableStateOf<RenderSessionStore.Session?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        session = RenderSessionStore.current()

        setContent {
            GenUiCraftTheme {
                ItemListScreen(
                    session = session,
                    onItemClick = { position ->
                        startActivity(
                            Intent(this, RenderActivity::class.java)
                                .putExtra(RenderActivity.EXTRA_RECORD_INDEX, position)
                        )
                    }
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        session = RenderSessionStore.current()
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
private fun ItemListScreen(
    session: RenderSessionStore.Session?,
    onItemClick: (Int) -> Unit
) {
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 16.dp
        DeviceSizeClass.Medium -> 24.dp
        DeviceSizeClass.Expanded -> 30.dp
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = stringResource(id = R.string.list_title),
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
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .consumeWindowInsets(innerPadding)
        ) { backgroundModifier ->
            Column(
                modifier = backgroundModifier
                .padding(horizontal = horizontalPadding, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
            val sourceText = session?.sourceLabel ?: stringResource(id = R.string.list_empty_source)
            val countText = stringResource(id = R.string.list_count, session?.records?.size ?: 0)
            val modeText = stringResource(
                id = R.string.list_mode,
                when (session?.renderMode) {
                    RenderMode.NATIVE -> stringResource(id = R.string.render_mode_native)
                    else -> stringResource(id = R.string.render_mode_web)
                }
            )

            Text(
                text = sourceText,
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "$countText • $modeText",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            if (session == null || session.records.isEmpty()) {
                Text(
                    text = stringResource(id = R.string.list_empty_message),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 12.dp)
                )
                return@Column
            }

            CompositionLocalProvider(LocalOverscrollConfiguration provides null) {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(bottom = 20.dp)
                ) {
                    itemsIndexed(
                        items = session.records,
                        key = { index, item -> "${item.uiId ?: item.title}_$index" }
                    ) { index, item ->
                        RecordItemCard(
                            index = index,
                            item = item,
                            deviceConfig = deviceConfig,
                            onClick = { onItemClick(index) }
                        )
                    }
                }
            }
            }
        }
    }
}

@Composable
private fun RecordItemCard(
    index: Int,
    item: GenUiRecord,
    deviceConfig: DeviceUiConfig,
    onClick: () -> Unit
) {
    val allowExpanded = deviceConfig.allowLargeTextListLayout()
    val cardContainer = genUiCardContainerColor(GenUiCardTone.Neutral)
    val cardBorder = genUiCardBorderColor()
    val interactionSource = remember { MutableInteractionSource() }

    var resolvedMaxLines by remember(item.title, allowExpanded) { mutableIntStateOf(if (allowExpanded) 2 else 1) }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onClick
            ),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = CardDefaults.cardColors(containerColor = cardContainer),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, cardBorder)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = (index + 1).toString().padStart(2, '0'),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                )
                Text(
                    text = item.title,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = resolvedMaxLines,
                    overflow = TextOverflow.Clip,
                    onTextLayout = { layoutResult ->
                        if (allowExpanded) {
                            resolvedMaxLines = if (layoutResult.lineCount > 1) 2 else 1
                        }
                    },
                    modifier = Modifier.weight(1f)
                )
            }

            Text(
                text = buildMetaText(item),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = if (resolvedMaxLines == 2) 4 else 3,
                overflow = TextOverflow.Clip
            )

            Spacer(modifier = Modifier.height(2.dp))

            Text(
                text = item.sourceLabel,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )
        }
    }
}

private fun buildMetaText(item: GenUiRecord): String {
    if (!item.summary.isNullOrBlank()) {
        return if (!item.uiId.isNullOrBlank()) {
            "id: ${item.uiId}  |  ${item.summary}"
        } else {
            item.summary
        }
    }

    val parts = mutableListOf<String>()
    if (!item.uiId.isNullOrBlank()) {
        parts += "id: ${item.uiId}"
    }
    if (!item.queryId.isNullOrBlank()) {
        parts += "query: ${item.queryId}"
    }
    if (!item.responseId.isNullOrBlank()) {
        parts += "response: ${item.responseId}"
    }
    return if (parts.isEmpty()) "Tap to render this UI" else parts.joinToString("  |  ")
}
