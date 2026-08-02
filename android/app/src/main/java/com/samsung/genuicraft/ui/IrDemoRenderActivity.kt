package com.samsung.genuicraft

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.security.SafeContentPolicy
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File

private sealed interface IrDemoRenderUiState {
    data class Loading(val message: String) : IrDemoRenderUiState
    data class Success(val result: GenUiStagePipeline.PipelineResult) : IrDemoRenderUiState
    data class Failure(val message: String) : IrDemoRenderUiState
}

private data class IrDemoGenerationDebugState(
    val streamText: String = "",
    val runtimeBackend: String? = null,
    val outputTokens: Int? = null,
    val outputTokensPerSecond: Double? = null,
    val metricsAreEstimated: Boolean = true,
    val complete: Boolean = false,
)

private object IrDemoRenderSessionCache {
    var recordIndex: Int = -1
    var debugMode: Boolean = false
    var logs: List<String> = emptyList()
    var generatedIrJson: String? = null
    var successResult: GenUiStagePipeline.PipelineResult? = null
    var loadingMessage: String? = null
    var failureMessage: String? = null
    var generationDebugState: IrDemoGenerationDebugState = IrDemoGenerationDebugState()
}

private val IrDemoDebugJsonGson = GsonBuilder()
    .disableHtmlEscaping()
    .setPrettyPrinting()
    .create()

class IrDemoRenderActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_RECORD_INDEX = "extra_record_index"
        const val EXTRA_FORCE_FRESH = "extra_force_fresh"
    }

    private var session by mutableStateOf<IrDemoSessionStore.Session?>(null)
    private var record by mutableStateOf<IrDemoRecord?>(null)
    private var debugMode by mutableStateOf(false)
    private var pipelineLogs by mutableStateOf<List<String>>(emptyList())
    private var generatedIrJson by mutableStateOf<String?>(null)
    private var generationDebugState by mutableStateOf(IrDemoGenerationDebugState())
    private var currentRecordIndex: Int = -1
    private var uiState by mutableStateOf<IrDemoRenderUiState>(
        IrDemoRenderUiState.Loading(message = "")
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        val index = intent.getIntExtra(EXTRA_RECORD_INDEX, -1)
        val forceFreshLaunch = intent.getBooleanExtra(EXTRA_FORCE_FRESH, false) && savedInstanceState == null
        currentRecordIndex = index
        if (forceFreshLaunch) {
            clearSessionCache()
        }
        session = IrDemoSessionStore.current()
        record = session?.records?.getOrNull(index)

        val selected = record
        if (selected == null) {
            uiState = IrDemoRenderUiState.Failure(getString(R.string.ir_demo_error_missing))
            persistSessionCache()
        } else {
            val restored = restoreFromSessionCache()
            if (!restored) {
                if (selected.hasSavedIr) {
                    renderSavedRecord(selected)
                } else {
                    runStage3ForRecord(selected)
                }
            }
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
                    generationDebugState = generationDebugState,
                    onDebugModeChange = {
                        debugMode = it
                        persistSessionCache()
                    },
                    onOpenExternalUrl = { openExternalUrl(it) }
                )
            }
        }
    }

    private fun clearSessionCache() {
        IrDemoRenderSessionCache.recordIndex = -1
        IrDemoRenderSessionCache.debugMode = false
        IrDemoRenderSessionCache.logs = emptyList()
        IrDemoRenderSessionCache.generatedIrJson = null
        IrDemoRenderSessionCache.successResult = null
        IrDemoRenderSessionCache.loadingMessage = null
        IrDemoRenderSessionCache.failureMessage = null
        IrDemoRenderSessionCache.generationDebugState = IrDemoGenerationDebugState()
    }

    private fun restoreFromSessionCache(): Boolean {
        if (IrDemoRenderSessionCache.recordIndex != currentRecordIndex) {
            return false
        }

        debugMode = IrDemoRenderSessionCache.debugMode
        pipelineLogs = IrDemoRenderSessionCache.logs
        generatedIrJson = IrDemoRenderSessionCache.generatedIrJson
        generationDebugState = IrDemoRenderSessionCache.generationDebugState

        IrDemoRenderSessionCache.successResult?.let {
            uiState = IrDemoRenderUiState.Success(it)
            if (generationDebugState.runtimeBackend.isNullOrBlank()) {
                generationDebugState = generationDebugState.copy(
                    runtimeBackend = it.stage3RuntimeBackend,
                    outputTokens = it.stage3OutputTokens,
                    outputTokensPerSecond = it.stage3OutputTokensPerSecond,
                    metricsAreEstimated = false,
                    complete = true,
                )
            }
            ensureIrGenerationTimingLog(
                stageDurationsMs = it.stageDurationsMs,
                stageStreamDurationsMs = it.stageStreamDurationsMs
            )
            return true
        }
        IrDemoRenderSessionCache.failureMessage?.takeIf { it.isNotBlank() }?.let {
            uiState = IrDemoRenderUiState.Failure(it)
            return true
        }
        IrDemoRenderSessionCache.loadingMessage?.takeIf { it.isNotBlank() }?.let {
            uiState = IrDemoRenderUiState.Loading(it)
        }
        return false
    }

    private fun persistSessionCache() {
        IrDemoRenderSessionCache.recordIndex = currentRecordIndex
        IrDemoRenderSessionCache.debugMode = debugMode
        IrDemoRenderSessionCache.logs = pipelineLogs
        IrDemoRenderSessionCache.generatedIrJson = generatedIrJson
        IrDemoRenderSessionCache.generationDebugState = generationDebugState
        when (val state = uiState) {
            is IrDemoRenderUiState.Success -> {
                IrDemoRenderSessionCache.successResult = state.result
                IrDemoRenderSessionCache.loadingMessage = null
                IrDemoRenderSessionCache.failureMessage = null
            }

            is IrDemoRenderUiState.Loading -> {
                IrDemoRenderSessionCache.successResult = null
                IrDemoRenderSessionCache.loadingMessage = state.message
                IrDemoRenderSessionCache.failureMessage = null
            }

            is IrDemoRenderUiState.Failure -> {
                IrDemoRenderSessionCache.successResult = null
                IrDemoRenderSessionCache.loadingMessage = null
                IrDemoRenderSessionCache.failureMessage = state.message
            }
        }
    }

    private fun appendPipelineLog(message: String) {
        val sanitized = sanitizeIrDemoLogText(message)
        if (sanitized.isBlank()) {
            return
        }
        pipelineLogs = pipelineLogs + sanitized
        persistSessionCache()
    }

    private fun applyGenerationDebugUpdate(update: GenUiStagePipeline.StageUpdate) {
        val isStreamEvent = update.stage3StreamText != null
        val hasDiagnostics = isStreamEvent ||
            update.llmOutputTokens != null ||
            update.llmOutputTokensPerSecond != null ||
            !update.llmRuntimeBackend.isNullOrBlank()
        if (!hasDiagnostics) {
            return
        }
        generationDebugState = generationDebugState.copy(
            streamText = update.stage3StreamText ?: generationDebugState.streamText,
            runtimeBackend = update.llmRuntimeBackend ?: generationDebugState.runtimeBackend,
            outputTokens = update.llmOutputTokens ?: generationDebugState.outputTokens,
            outputTokensPerSecond = update.llmOutputTokensPerSecond
                ?: generationDebugState.outputTokensPerSecond,
            metricsAreEstimated = if (isStreamEvent) {
                update.streamMetricsAreEstimated
            } else {
                generationDebugState.metricsAreEstimated
            },
            complete = when {
                isStreamEvent -> update.stage3StreamComplete
                !update.stage3Json.isNullOrBlank() -> true
                else -> generationDebugState.complete
            },
        )
    }

    private fun ensureIrGenerationTimingLog(
        stageDurationsMs: Map<GenUiStagePipeline.Stage, Long>,
        stageStreamDurationsMs: Map<GenUiStagePipeline.Stage, Long>,
        fallbackDurationMs: Long? = null,
        beforeFailure: Boolean = false
    ) {
        val alreadyLogged = pipelineLogs.any {
            it.startsWith("IR generation time:", ignoreCase = true) ||
                it.startsWith("IR generation time before failure:", ignoreCase = true)
        }
        if (alreadyLogged) {
            return
        }

        val durationMs = stageDurationsMs[GenUiStagePipeline.Stage.STAGE3]
            ?: fallbackDurationMs
        val streamPart = stageStreamDurationsMs[GenUiStagePipeline.Stage.STAGE3]
            ?.let { " (stream ${formatIrDemoDuration(it)})" }
            .orEmpty()
        val prefix = if (beforeFailure) {
            "IR generation time before failure"
        } else {
            "IR generation time"
        }
        val value = if (durationMs != null && durationMs > 0L) {
            "${formatIrDemoDuration(durationMs)}$streamPart"
        } else {
            "unavailable"
        }
        appendPipelineLog("$prefix: $value")
    }

    private fun renderSavedRecord(record: IrDemoRecord) {
        val payload = IrDemoRecordRepository.buildRenderablePayload(record)
        val savedIr = record.genUiJson?.trim().orEmpty()
        generatedIrJson = savedIr.takeIf { it.isNotBlank() }
        pipelineLogs = emptyList()
        generationDebugState = IrDemoGenerationDebugState()
        if (payload.isNullOrBlank() || savedIr.isBlank()) {
            uiState = IrDemoRenderUiState.Failure("Saved Demo item is missing GenUI IR.")
            persistSessionCache()
            return
        }

        val renderResult = GenUiNativeRenderer.renderLegacyForComparison(
            rawInput = payload,
            sourceDir = record.savedSourceDir()
        )
        if (renderResult.errorMessage != null) {
            uiState = IrDemoRenderUiState.Failure(renderResult.errorMessage)
            persistSessionCache()
            return
        }

        uiState = IrDemoRenderUiState.Success(
            GenUiStagePipeline.PipelineResult(
                queryText = record.queryText,
                stage2Prompt = "Saved Demo artifact",
                stage2Response = record.responseText,
                stage3Prompt = "Saved GenUI IR",
                stage3SystemPrompt = null,
                stage3Json = savedIr,
                stage3InputTokens = null,
                stage3OutputTokens = null,
                stageDurationsMs = emptyMap(),
                stageStreamDurationsMs = emptyMap(),
                usedFallback = false,
                warnings = renderResult.warnings,
                renderResult = renderResult
            )
        )
        persistSessionCache()
    }

    private fun runStage3ForRecord(record: IrDemoRecord) {
        val pipeline = GenUiStagePipeline(this)
        val queryText = decodeIrDemoQueryText(record.queryText)
        generatedIrJson = null
        pipelineLogs = emptyList()
        generationDebugState = IrDemoGenerationDebugState()
        uiState = IrDemoRenderUiState.Loading(getString(R.string.ir_demo_status_initializing))
        persistSessionCache()
        val runStartedAtMs = System.currentTimeMillis()
        appendPipelineLog("IR demo started for ${record.queryId}")
        val provider = InferenceBackendSettings.getIrProvider(this)
        appendPipelineLog("IR backend: ${provider.name.lowercase()}")
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            val localUrl = InferenceBackendSettings.getLocalServerBaseUrl(this)
            appendPipelineLog("Local server URL: ${localUrl.ifBlank { "<empty>" }}")
        }

        lifecycleScope.launch {
            val outcome = pipeline.executeStage3FromResponse(
                queryText = queryText,
                stage2ResponseText = record.responseText
            ) { update ->
                val isStreamEvent = update.stage3StreamText != null
                if (!isStreamEvent) {
                    appendPipelineLog(update.message)
                    update.debugLog?.let(::appendPipelineLog)
                }
                applyGenerationDebugUpdate(update)
                if (!update.stage3Json.isNullOrBlank()) {
                    generatedIrJson = update.stage3Json
                }
                val message = when (update.stage) {
                    GenUiStagePipeline.Stage.STAGE3 -> getString(R.string.ir_demo_status_stage3)
                    GenUiStagePipeline.Stage.STAGE4 -> getString(R.string.ir_demo_status_stage4)
                    else -> update.message
                }
                uiState = IrDemoRenderUiState.Loading(message)
                persistSessionCache()
            }

            uiState = when (outcome) {
                is GenUiStagePipeline.Outcome.Success -> {
                    generatedIrJson = outcome.result.stage3Json
                    generationDebugState = generationDebugState.copy(
                        runtimeBackend = outcome.result.stage3RuntimeBackend
                            ?: generationDebugState.runtimeBackend,
                        outputTokens = outcome.result.stage3OutputTokens
                            ?: generationDebugState.outputTokens,
                        outputTokensPerSecond = outcome.result.stage3OutputTokensPerSecond
                            ?: generationDebugState.outputTokensPerSecond,
                        complete = true,
                    )
                    ensureIrGenerationTimingLog(
                        stageDurationsMs = outcome.result.stageDurationsMs,
                        stageStreamDurationsMs = outcome.result.stageStreamDurationsMs,
                        fallbackDurationMs = System.currentTimeMillis() - runStartedAtMs
                    )
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
                    generationDebugState = generationDebugState.copy(complete = true)
                    ensureIrGenerationTimingLog(
                        stageDurationsMs = outcome.stageDurationsMs,
                        stageStreamDurationsMs = outcome.stageStreamDurationsMs,
                        fallbackDurationMs = System.currentTimeMillis() - runStartedAtMs,
                        beforeFailure = true
                    )
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
            persistSessionCache()
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
    generationDebugState: IrDemoGenerationDebugState,
    onDebugModeChange: (Boolean) -> Unit,
    onOpenExternalUrl: (String) -> Unit
) {
    val deviceConfig = rememberDeviceUiConfig()
    val contentListState = rememberLazyListState()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 12.dp
        DeviceSizeClass.Medium -> 18.dp
        DeviceSizeClass.Expanded -> 24.dp
    }

    val topBarTitle = stringResource(id = R.string.genui_demo_title)
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
            modifier = Modifier.fillMaxSize()
        ) { backgroundModifier ->
            val hasGeneratedJson = !generatedIrJson.isNullOrBlank()
            val shouldAutoScroll =
                when {
                    debugMode -> {
                        uiState is IrDemoRenderUiState.Loading ||
                            uiState is IrDemoRenderUiState.Success ||
                            hasGeneratedJson
                    }

                    else -> {
                        uiState is IrDemoRenderUiState.Success || hasGeneratedJson
                    }
                }
            LaunchedEffect(debugMode, shouldAutoScroll, logs.size, generatedIrJson, uiState::class) {
                if (!shouldAutoScroll) {
                    return@LaunchedEffect
                }
                repeat(3) {
                    snapshotFlow { contentListState.layoutInfo.totalItemsCount }
                        .filter { it > 0 }
                        .first()
                    val lastIndex = (contentListState.layoutInfo.totalItemsCount - 1).coerceAtLeast(0)
                    contentListState.scrollToItem(lastIndex)
                    delay(32)
                }
            }
            if (debugMode) {
                LazyColumn(
                    state = contentListState,
                    modifier = backgroundModifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .consumeWindowInsets(innerPadding)
                        .imePadding()
                        .padding(horizontal = horizontalPadding, vertical = 14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(bottom = 20.dp)
                ) {
                    item {
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            if (record != null) {
                                Text(
                                    text = stringResource(id = R.string.ir_demo_query_prefix),
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.primary
                                )
                                SelectionContainer {
                                    Text(
                                        text = decodeIrDemoQueryText(record.queryText),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                        }
                    }

                    if (logs.isNotEmpty()) {
                        item { IrDemoLogCard(logs = logs) }
                    }
                    if (generationDebugState.hasMetrics()) {
                        item {
                            IrDemoGenerationStatsCard(state = generationDebugState)
                        }
                    }
                    if (generationDebugState.streamText.isNotBlank()) {
                        item {
                            IrDemoDebugCard(
                                title = if (generationDebugState.complete) {
                                    "IR generation stream"
                                } else {
                                    "Live IR generation"
                                },
                                content = generationDebugState.streamText + if (generationDebugState.complete) "" else "\u258C",
                                monospace = true
                            )
                        }
                    }
                    item {
                        IrDemoDebugCard(
                            title = "Response",
                            content = record?.responseText.orEmpty(),
                            monospace = false
                        )
                    }
                    if (!generatedIrJson.isNullOrBlank()) {
                        item {
                            IrDemoDebugCard(
                                title = "Generated IR JSON",
                                content = formatIrDemoJsonForDebug(generatedIrJson),
                                monospace = true
                            )
                        }
                    }

                    when (uiState) {
                        is IrDemoRenderUiState.Loading -> {
                            item { IrDemoLoadingCard(message = uiState.message) }
                        }

                        is IrDemoRenderUiState.Failure -> {
                            item {
                                Text(
                                    text = uiState.message,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.error
                                )
                            }
                        }

                        is IrDemoRenderUiState.Success -> {
                            if (uiState.result.warnings.isNotEmpty()) {
                                item { IrDemoWarningCard(warnings = uiState.result.warnings) }
                            }
                            item {
                                GenUiNativeRenderer.RenderInline(
                                    result = uiState.result.renderResult,
                                    sourceDir = record?.savedSourceDir(),
                                    onOpenExternalUrl = onOpenExternalUrl,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        }
                    }
                }
            } else {
                LazyColumn(
                    state = contentListState,
                    modifier = backgroundModifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .consumeWindowInsets(innerPadding)
                        .imePadding()
                        .padding(horizontal = horizontalPadding, vertical = 14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(bottom = 20.dp)
                ) {
                    item {
                        if (record != null) {
                            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text(
                                    text = stringResource(id = R.string.ir_demo_query_prefix),
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.primary
                                )
                                SelectionContainer {
                                    Text(
                                        text = decodeIrDemoQueryText(record.queryText),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                        }
                    }

                    when (uiState) {
                        is IrDemoRenderUiState.Loading -> {
                            item { IrDemoLoadingCard(message = uiState.message) }
                        }

                        is IrDemoRenderUiState.Failure -> {
                            item {
                                Text(
                                    text = uiState.message,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.error
                                )
                            }
                        }

                        is IrDemoRenderUiState.Success -> {
                            if (uiState.result.warnings.isNotEmpty()) {
                                item { IrDemoWarningCard(warnings = uiState.result.warnings) }
                            }
                            item {
                                GenUiNativeRenderer.RenderInline(
                                    result = uiState.result.renderResult,
                                    sourceDir = record?.savedSourceDir(),
                                    onOpenExternalUrl = onOpenExternalUrl,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun IrDemoLoadingCard(message: String) {
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
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

@Composable
private fun IrDemoGenerationStatsCard(state: IrDemoGenerationDebugState) {
    val estimatePrefix = if (state.metricsAreEstimated) "~" else ""
    val tokenText = state.outputTokens?.let { "$estimatePrefix$it tokens" } ?: "Waiting for tokens"
    val speedText = state.outputTokensPerSecond?.let {
        String.format(java.util.Locale.US, "$estimatePrefix%.2f tokens/s", it)
    } ?: "Measuring speed"
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
        colors = genUiCardColors(GenUiCardTone.Primary),
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
                text = "IR inference",
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "Runtime: ${state.runtimeBackend ?: "Initializing"}",
                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "$tokenText  |  $speedText",
                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

private fun IrDemoGenerationDebugState.hasMetrics(): Boolean {
    return !runtimeBackend.isNullOrBlank() || outputTokens != null || outputTokensPerSecond != null
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
                modifier = Modifier.fillMaxWidth()
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
            logs.forEach { line ->
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
        .replace(Regex("""(?i)\bstage\s*[234]\b\s*[:\-]?\s*"""), "")
        .replace('\n', ' ')
        .replace(Regex("""\s{2,}"""), " ")
        .trim()
}

private fun previewIrDemoLog(value: String): String {
    return sanitizeIrDemoLogText(value).take(240)
}

private fun formatIrDemoJsonForDebug(value: String): String {
    return runCatching {
        IrDemoDebugJsonGson.toJson(JsonParser.parseString(value))
    }.getOrDefault(value)
}

private fun formatIrDemoDuration(durationMs: Long): String {
    if (durationMs < 60_000L) {
        return String.format(java.util.Locale.US, "%.1fs", durationMs / 1000.0)
    }
    val minutes = durationMs / 60_000L
    val seconds = (durationMs % 60_000L) / 1000L
    return "${minutes}m ${seconds}s"
}

private fun IrDemoRecord.savedSourceDir(): File? {
    val rawPath = sourceDirPath?.trim().orEmpty()
    if (rawPath.isBlank()) {
        return null
    }
    return File(rawPath)
}
