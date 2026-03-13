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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
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
    private var debugMode by mutableStateOf(false)
    private var pipelineLogs by mutableStateOf<List<String>>(emptyList())
    private var generatedIrJson by mutableStateOf<String?>(null)
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
                    logs = pipelineLogs,
                    debugMode = debugMode,
                    generatedIrJson = generatedIrJson,
                    onDebugModeChange = { debugMode = it },
                    onOpenExternalUrl = { openExternalUrl(it) }
                )
            }
        }
    }

    private fun appendPipelineLog(message: String) {
        val sanitized = sanitizeIrDemoLogText(message)
        if (sanitized.isBlank()) {
            return
        }
        pipelineLogs = (pipelineLogs + sanitized).takeLast(40)
    }

    private fun runStage3ForRecord(record: IrDemoRecord) {
        val pipeline = GenUiStagePipeline(this)
        generatedIrJson = null
        pipelineLogs = emptyList()
        appendPipelineLog("IR demo started for ${record.queryId}")
        val provider = InferenceBackendSettings.getProvider(this)
        appendPipelineLog("Backend: ${provider.name.lowercase()}")
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            val localUrl = InferenceBackendSettings.getLocalServerBaseUrl(this)
            appendPipelineLog("Local server URL: ${localUrl.ifBlank { "<empty>" }}")
        }

        uiState = IrDemoRenderUiState.Loading(getString(R.string.ir_demo_status_initializing))
        lifecycleScope.launch {
            val outcome = pipeline.executeStage3FromResponse(
                queryText = record.queryText,
                stage2ResponseText = record.responseText
            ) { update ->
                appendPipelineLog(update.message)
                val message = when (update.stage) {
                    GenUiStagePipeline.Stage.STAGE3 -> getString(R.string.ir_demo_status_stage3)
                    GenUiStagePipeline.Stage.STAGE4 -> getString(R.string.ir_demo_status_stage4)
                    else -> update.message
                }
                uiState = IrDemoRenderUiState.Loading(message)
            }

            uiState = when (outcome) {
                is GenUiStagePipeline.Outcome.Success -> {
                    generatedIrJson = outcome.result.stage3Json
                    appendPipelineLog("Pipeline completed successfully.")
                    if (outcome.result.warnings.isNotEmpty()) {
                        outcome.result.warnings.forEach { warning ->
                            appendPipelineLog("Warning: $warning")
                        }
                    }
                    IrDemoRenderUiState.Success(outcome.result)
                }

                is GenUiStagePipeline.Outcome.Failure -> {
                    generatedIrJson = outcome.stage3Json
                    appendPipelineLog("Pipeline failed.")
                    appendPipelineLog("Error: ${outcome.message}")
                    if (!outcome.stage2Response.isNullOrBlank()) {
                        appendPipelineLog(
                            "Response preview: ${previewIrDemoLog(outcome.stage2Response)}"
                        )
                    }
                    if (!outcome.stage3Json.isNullOrBlank()) {
                        appendPipelineLog(
                            "IR JSON preview: ${previewIrDemoLog(outcome.stage3Json)}"
                        )
                    }
                    IrDemoRenderUiState.Failure(
                        outcome.message
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
    logs: List<String>,
    debugMode: Boolean,
    generatedIrJson: String?,
    onDebugModeChange: (Boolean) -> Unit,
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
                actions = {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                        modifier = Modifier.padding(end = 6.dp)
                    ) {
                        Text(
                            text = "Debug",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Switch(
                            checked = debugMode,
                            onCheckedChange = onDebugModeChange
                        )
                    }
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

                if (debugMode && logs.isNotEmpty()) {
                    IrDemoLogCard(logs = logs)
                }

                if (debugMode) {
                    IrDemoDebugCard(
                        title = "Response",
                        content = record?.responseText.orEmpty(),
                        monospace = false
                    )
                    if (!generatedIrJson.isNullOrBlank()) {
                        IrDemoDebugCard(
                            title = "Generated IR JSON",
                            content = generatedIrJson,
                            monospace = true
                        )
                    }
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
private fun IrDemoDebugCard(
    title: String,
    content: String,
    monospace: Boolean
) {
    if (content.isBlank()) {
        return
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
        colors = genUiCardColors(GenUiCardTone.Neutral),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = content,
                style = MaterialTheme.typography.bodySmall.copy(
                    fontFamily = if (monospace) FontFamily.Monospace else FontFamily.Default
                ),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = 260.dp)
                    .verticalScroll(rememberScrollState())
            )
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

@Composable
private fun IrDemoLogCard(logs: List<String>) {
    val recent = if (logs.size <= 12) logs else logs.takeLast(12)
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
        colors = genUiCardColors(GenUiCardTone.Neutral),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "Logs",
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface
            )
            recent.forEach { line ->
                Text(
                    text = line,
                    style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

private fun sanitizeIrDemoLogText(value: String): String {
    return value
        .replace(Regex("""(?i)\bstage\s*[34]\b\s*[:\-]?\s*"""), "")
        .replace('\n', ' ')
        .replace(Regex("""\s{2,}"""), " ")
        .trim()
}

private fun previewIrDemoLog(value: String): String {
    return sanitizeIrDemoLogText(value).take(240)
}
