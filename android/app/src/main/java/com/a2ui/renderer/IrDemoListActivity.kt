package com.samsung.genuicraft

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

class IrDemoListActivity : AppCompatActivity() {
    private var session by mutableStateOf<IrDemoSessionStore.Session?>(null)
    private var errorMessage by mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        loadDefaultIrDemo()

        setContent {
            GenUiCraftTheme {
                IrDemoListScreen(
                    session = session,
                    errorMessage = errorMessage,
                    onItemClick = { index ->
                        startActivity(
                            Intent(this, IrDemoRenderActivity::class.java)
                                .putExtra(IrDemoRenderActivity.EXTRA_RECORD_INDEX, index)
                        )
                    }
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        session = IrDemoSessionStore.current()
    }

    private fun loadDefaultIrDemo() {
        try {
            val records = IrDemoRecordLoader.loadDefault(assets)
            IrDemoSessionStore.update(
                sourceLabel = getString(R.string.ir_demo_source),
                records = records
            )
            session = IrDemoSessionStore.current()
            errorMessage = null
        } catch (exc: Exception) {
            val msg = exc.message?.trim().takeUnless { it.isNullOrBlank() } ?: exc.javaClass.simpleName
            errorMessage = getString(R.string.ir_demo_error_load, msg)
            session = null
        }
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun IrDemoListScreen(
    session: IrDemoSessionStore.Session?,
    errorMessage: String?,
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
                        text = stringResource(id = R.string.ir_demo_list_title),
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
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = session?.sourceLabel ?: stringResource(id = R.string.ir_demo_source),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = stringResource(id = R.string.ir_demo_item_count, session?.records?.size ?: 0),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                if (!errorMessage.isNullOrBlank()) {
                    Text(
                        text = errorMessage,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                    return@Column
                }

                if (session == null || session.records.isEmpty()) {
                    Text(
                        text = stringResource(id = R.string.ir_demo_list_empty),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    return@Column
                }

                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(bottom = 20.dp)
                ) {
                    itemsIndexed(
                        items = session.records,
                        key = { index, item -> "${item.queryId}_${item.responseId}_$index" }
                    ) { index, item ->
                        IrDemoItemCard(
                            index = index,
                            record = item,
                            onClick = { onItemClick(index) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun IrDemoItemCard(
    index: Int,
    record: IrDemoRecord,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = genUiCardColors(GenUiCardTone.Neutral),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = (index + 1).toString().padStart(2, '0'),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    text = record.queryText,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
            }

            Text(
                text = previewResponse(record.responseText),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )

            val footer = buildString {
                append(record.queryId)
                if (!record.responseId.isNullOrBlank()) {
                    append("  |  ")
                    append(record.responseId)
                }
            }
            Text(
                text = footer,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )
        }
    }
}

private fun previewResponse(text: String): String {
    val normalized = text
        .replace("\r", "\n")
        .lineSequence()
        .map { it.trim() }
        .firstOrNull { it.isNotBlank() }
        .orEmpty()
        .replace(Regex("\\s+"), " ")
        .trim()

    return normalized.take(220)
}

