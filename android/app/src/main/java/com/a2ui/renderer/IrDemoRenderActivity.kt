package com.samsung.genuicraft

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

private sealed interface IrDemoRenderUiState {
    data class Loading(val message: String) : IrDemoRenderUiState
    data class Success(val result: GenUiStagePipeline.PipelineResult) : IrDemoRenderUiState
    data class Failure(val message: String) : IrDemoRenderUiState
}

class IrDemoRenderActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_RECORD_INDEX = "extra_record_index"
    }

    private var session by mutableStateOf<IrDemoSessionStore.Session?>(null)
    private var record by mutableStateOf<IrDemoRecord?>(null)
    private var uiState by mutableStateOf<IrDemoRenderUiState>(
        IrDemoRenderUiState.Loading(message = "")
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        val index = intent.getIntExtra(EXTRA_RECORD_INDEX, -1)
        session = IrDemoSessionStore.current()
        record = session?.records?.getOrNull(index)

        val selected = record
        if (selected == null) {
            uiState = IrDemoRenderUiState.Failure(getString(R.string.ir_demo_error_missing))
        } else {
            runStage3ForRecord(selected)
        }

        setContent {
            GenUiCraftTheme {
                IrDemoRenderScreen(
                    session = session,
                    record = record,
                    uiState = uiState,
                    onOpenExternalUrl = { openExternalUrl(it) }
                )
            }
        }
    }

    private fun runStage3ForRecord(record: IrDemoRecord) {
        val pipeline = GenUiStagePipeline(this)
        uiState = IrDemoRenderUiState.Loading(getString(R.string.ir_demo_status_initializing))
        lifecycleScope.launch {
            val outcome = pipeline.executeStage3FromResponse(
                queryText = record.queryText,
                stage2ResponseText = record.responseText
            ) { update ->
                val message = when (update.stage) {
                    GenUiStagePipeline.Stage.STAGE3 -> getString(R.string.ir_demo_status_stage3)
                    GenUiStagePipeline.Stage.STAGE4 -> getString(R.string.ir_demo_status_stage4)
                    else -> update.message
                }
                uiState = IrDemoRenderUiState.Loading(message)
            }

            uiState = when (outcome) {
                is GenUiStagePipeline.Outcome.Success -> {
                    IrDemoRenderUiState.Success(outcome.result)
                }

                is GenUiStagePipeline.Outcome.Failure -> {
                    val stageName = outcome.stage.name.removePrefix("STAGE")
                    IrDemoRenderUiState.Failure(
                        getString(R.string.ir_demo_error_failed, stageName, outcome.message)
                    )
                }
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
private fun IrDemoRenderScreen(
    session: IrDemoSessionStore.Session?,
    record: IrDemoRecord?,
    uiState: IrDemoRenderUiState,
    onOpenExternalUrl: (String) -> Unit
) {
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 12.dp
        DeviceSizeClass.Medium -> 18.dp
        DeviceSizeClass.Expanded -> 24.dp
    }

    val topBarTitle = record?.responseId ?: record?.queryId ?: stringResource(id = R.string.ir_demo_render_title)
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
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .consumeWindowInsets(innerPadding)
        ) { backgroundModifier ->
            Column(
                modifier = backgroundModifier
                    .padding(horizontal = horizontalPadding, vertical = 14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (session != null) {
                    Text(
                        text = session.sourceLabel,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                if (record != null) {
                    Text(
                        text = stringResource(id = R.string.ir_demo_query_prefix),
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = record.queryText,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                when (uiState) {
                    is IrDemoRenderUiState.Loading -> {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                            colors = genUiCardColors(GenUiCardTone.Primary),
                            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 14.dp, vertical = 12.dp),
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                CircularProgressIndicator(
                                    strokeWidth = 2.dp,
                                    modifier = Modifier.padding(top = 2.dp)
                                )
                                Text(
                                    text = uiState.message,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        }
                    }

                    is IrDemoRenderUiState.Failure -> {
                        Text(
                            text = uiState.message,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }

                    is IrDemoRenderUiState.Success -> {
                        if (uiState.result.warnings.isNotEmpty()) {
                            IrDemoWarningCard(warnings = uiState.result.warnings)
                        }
                        GenUiNativeRenderer.Render(
                            result = uiState.result.renderResult,
                            sourceDir = null,
                            onOpenExternalUrl = onOpenExternalUrl,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun IrDemoWarningCard(warnings: List<String>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
        colors = genUiCardColors(GenUiCardTone.Warning),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            warnings.forEach { warning ->
                Text(
                    text = warning,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

