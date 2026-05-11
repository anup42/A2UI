package com.samsung.genuicraft

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.FileUpload
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.google.gson.GsonBuilder
import com.samsung.genuicraft.pipeline.PipelineJsonExtractor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class PipelineStepStatus {
    Pending,
    Running,
    Done,
    Failed
}

private data class StageStepUi(
    val stage: GenUiStagePipeline.Stage,
    val title: String,
    val status: PipelineStepStatus = PipelineStepStatus.Pending,
    val message: String = ""
)

private data class AssistantLogItem(
    val title: String,
    val content: String,
    val monospace: Boolean = false
)

private data class StarterPrompt(
    val label: String,
    val description: String,
    val prompt: String,
    val icon: ImageVector
)

private val GenUiStarterPrompts = listOf(
    StarterPrompt(
        label = "Weather in Bengaluru",
        description = "Forecast cards and weather metrics.",
        prompt = "Show weather in Bengaluru",
        icon = Icons.Filled.WbSunny
    ),
    StarterPrompt(
        label = "Flights BLR to LKO",
        description = "Flight cards with price and timing.",
        prompt = "Show flights from BLR to LKO on 15 June 2026",
        icon = Icons.Filled.FlightTakeoff
    )
)

private object GenUiAssistantSessionCache {
    var inputText: String = ""
    var debugMode: Boolean = false
    var currentStatus: String = ""
    var errorText: String? = null
    var stage2Text: String? = null
    var stage3Json: String? = null
    var currentQuery: String? = null
    var usedFallback: Boolean = false
    var warnings: List<String> = emptyList()
    var stage3InputTokens: Int? = null
    var stage3OutputTokens: Int? = null
    var renderResult: GenUiNativeRenderer.RenderResult? = null
    var logs: List<AssistantLogItem> = emptyList()
}

class GenUiAssistantActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestNotificationPermissionIfNeeded()
        applyOneUiWindowBlur()
        window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        setContent {
            GenUiCraftTheme {
                GenUiAssistantScreen(
                    onOpenExternalUrl = { openExternalUrl(it) }
                )
            }
        }
    }

    private fun openExternalUrl(url: String) {
        val uri = runCatching { Uri.parse(url) }.getOrNull() ?: return
        val scheme = uri.scheme?.lowercase()
        if (scheme == "http" || scheme == "https") {
            startActivity(Intent(Intent.ACTION_VIEW, uri))
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return
        }
        val granted = ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                2041
            )
        }
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun GenUiAssistantScreen(
    onOpenExternalUrl: (String) -> Unit
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val pipeline = remember { GenUiStagePipeline(context.applicationContext) }
    val coroutineScope = rememberCoroutineScope()
    val steps = remember {
        mutableStateListOf(
            StageStepUi(GenUiStagePipeline.Stage.STAGE2, "Rich response"),
            StageStepUi(GenUiStagePipeline.Stage.STAGE3, "GenUI JSON"),
            StageStepUi(GenUiStagePipeline.Stage.STAGE4, "Native render")
        )
    }

    var inputText by rememberSaveable { mutableStateOf(GenUiAssistantSessionCache.inputText) }
    var debugMode by rememberSaveable { mutableStateOf(GenUiAssistantSessionCache.debugMode) }
    var isRunning by remember { mutableStateOf(false) }
    var currentStatus by rememberSaveable { mutableStateOf(GenUiAssistantSessionCache.currentStatus) }
    var errorText by rememberSaveable { mutableStateOf<String?>(GenUiAssistantSessionCache.errorText) }
    var stage2Text by remember { mutableStateOf<String?>(GenUiAssistantSessionCache.stage2Text) }
    var stage3Json by remember { mutableStateOf<String?>(GenUiAssistantSessionCache.stage3Json) }
    var currentQuery by rememberSaveable { mutableStateOf<String?>(GenUiAssistantSessionCache.currentQuery) }
    var usedFallback by rememberSaveable { mutableStateOf(GenUiAssistantSessionCache.usedFallback) }
    var warnings by remember { mutableStateOf(ArrayList(GenUiAssistantSessionCache.warnings)) }
    var stage3InputTokens by rememberSaveable { mutableStateOf(GenUiAssistantSessionCache.stage3InputTokens) }
    var stage3OutputTokens by rememberSaveable { mutableStateOf(GenUiAssistantSessionCache.stage3OutputTokens) }
    var renderResult by remember {
        mutableStateOf<GenUiNativeRenderer.RenderResult?>(GenUiAssistantSessionCache.renderResult)
    }
    val logs = remember {
        mutableStateListOf<AssistantLogItem>().apply {
            addAll(GenUiAssistantSessionCache.logs)
        }
    }
    val restoredRenderResult = remember(stage3Json) {
        stage3Json
            ?.takeIf { it.isNotBlank() }
            ?.let { GenUiNativeRenderer.render(rawInput = it, sourceDir = null) }
            ?.takeIf { it.errorMessage == null }
    }
    val displayRenderResult = renderResult ?: restoredRenderResult
    LaunchedEffect(
        inputText,
        debugMode,
        currentStatus,
        errorText,
        stage2Text,
        stage3Json,
        currentQuery,
        usedFallback,
        warnings,
        stage3InputTokens,
        stage3OutputTokens,
        renderResult,
        logs.toList()
    ) {
        GenUiAssistantSessionCache.inputText = inputText
        GenUiAssistantSessionCache.debugMode = debugMode
        GenUiAssistantSessionCache.currentStatus = currentStatus
        GenUiAssistantSessionCache.errorText = errorText
        GenUiAssistantSessionCache.stage2Text = stage2Text
        GenUiAssistantSessionCache.stage3Json = stage3Json
        GenUiAssistantSessionCache.currentQuery = currentQuery
        GenUiAssistantSessionCache.usedFallback = usedFallback
        GenUiAssistantSessionCache.warnings = warnings.toList()
        GenUiAssistantSessionCache.stage3InputTokens = stage3InputTokens
        GenUiAssistantSessionCache.stage3OutputTokens = stage3OutputTokens
        GenUiAssistantSessionCache.renderResult = displayRenderResult
        GenUiAssistantSessionCache.logs = logs.toList()
    }

    fun resetSteps(initialStage: GenUiStagePipeline.Stage) {
        steps.indices.forEach { index ->
            val step = steps[index]
            val status = if (step.stage == initialStage) {
                PipelineStepStatus.Running
            } else {
                PipelineStepStatus.Pending
            }
            steps[index] = step.copy(status = status, message = "")
        }
    }

    fun updateSteps(update: GenUiStagePipeline.StageUpdate) {
        steps.indices.forEach { index ->
            val step = steps[index]
            val status = when {
                step.stage == update.stage -> PipelineStepStatus.Running
                step.stage.ordinal < update.stage.ordinal -> PipelineStepStatus.Done
                else -> PipelineStepStatus.Pending
            }
            val message = if (step.stage == update.stage) update.message else step.message
            steps[index] = step.copy(status = status, message = message)
        }
    }

    fun formatLogCardContent(title: String, content: String): String {
        if (!title.equals("GenUI JSON", ignoreCase = true)) {
            return content
        }
        val trimmed = content.trim()
        if (trimmed.isBlank()) {
            return content
        }
        return runCatching {
            val parsed = PipelineJsonExtractor.extractJsonElement(trimmed)
                ?: PipelineJsonExtractor.extractJsonElement(
                    PipelineJsonExtractor.extractFencedBlock(trimmed).orEmpty()
                )
                ?: return content
            GsonBuilder()
                .disableHtmlEscaping()
                .setPrettyPrinting()
                .create()
                .toJson(parsed)
        }.getOrElse { content }
    }

    fun upsertLogCard(
        title: String,
        content: String,
        monospace: Boolean = false,
        append: Boolean = false
    ) {
        if (content.isBlank()) return
        val normalizedContent = if (append) content else formatLogCardContent(title, content)
        val index = logs.indexOfFirst { it.title == title }
        if (index < 0) {
            val newItem = AssistantLogItem(title = title, content = normalizedContent, monospace = monospace)
            if (title == "Debug log") {
                logs.add(0, newItem)
            } else {
                logs.add(newItem)
            }
            return
        }
        val existing = logs[index]
        val mergedContent = if (append) {
            val normalizedLine = normalizedContent.trim()
            val hasLine = existing.content
                .lineSequence()
                .map { it.trim() }
                .any { it.equals(normalizedLine, ignoreCase = false) }
            if (hasLine) {
                existing.content
            } else {
                "${existing.content}\n$normalizedLine".trim()
            }
        } else {
            normalizedContent
        }
        logs[index] = existing.copy(
            content = mergedContent,
            monospace = existing.monospace || monospace
        )
    }

    fun appendDebugLogLine(line: String) {
        val sanitized = sanitizeUiLogText(line)
        if (sanitized.isBlank()) return
        upsertLogCard(
            title = "Debug log",
            content = sanitized,
            monospace = true,
            append = true
        )
    }

    fun formatDurationLabel(durationMs: Long): String {
        val totalSeconds = (durationMs / 1000.0)
        return if (totalSeconds < 60.0) {
            String.format(java.util.Locale.US, "%.1fs", totalSeconds)
        } else {
            val minutes = (durationMs / 60000L)
            val seconds = ((durationMs % 60000L) / 1000L)
            "${minutes}m ${seconds}s"
        }
    }

    fun applyStageDurations(
        durations: Map<GenUiStagePipeline.Stage, Long>,
        streamDurations: Map<GenUiStagePipeline.Stage, Long> = emptyMap(),
        failedStage: GenUiStagePipeline.Stage? = null
    ) {
        if (durations.isEmpty() && streamDurations.isEmpty()) return
        steps.indices.forEach { index ->
            val step = steps[index]
            val duration = durations[step.stage]
            val streamDuration = streamDurations[step.stage]
            val streamLabel = streamDuration?.let { formatDurationLabel(it) }
            val message = if (duration != null) {
                val label = formatDurationLabel(duration)
                val suffix = streamLabel?.let { " | stream $it" }.orEmpty()
                if (failedStage == step.stage) {
                    "Failed after $label$suffix"
                } else {
                    "Completed in $label$suffix"
                }
            } else if (streamLabel != null) {
                if (failedStage == step.stage) {
                    "Failed | stream $streamLabel"
                } else {
                    "Completed | stream $streamLabel"
                }
            } else {
                step.message
            }
            steps[index] = step.copy(message = message)
        }
    }

    fun resetToIdleSteps() {
        steps.indices.forEach { index ->
            steps[index] = steps[index].copy(
                status = PipelineStepStatus.Pending,
                message = ""
            )
        }
    }

    fun clearConversation() {
        inputText = ""
        currentStatus = ""
        errorText = null
        stage2Text = null
        stage3Json = null
        currentQuery = null
        usedFallback = false
        warnings = arrayListOf()
        stage3InputTokens = null
        stage3OutputTokens = null
        renderResult = null
        logs.clear()
        resetToIdleSteps()
    }

    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 14.dp
        DeviceSizeClass.Medium -> 20.dp
        DeviceSizeClass.Expanded -> 26.dp
    }
    val imeVisible = WindowInsets.ime.getBottom(androidx.compose.ui.platform.LocalDensity.current) > 0
    val listBottomPadding = if (imeVisible) 16.dp else 112.dp
    val canSaveToIrDemo = !isRunning &&
        !currentQuery.isNullOrBlank() &&
        !stage2Text.isNullOrBlank() &&
        !stage3Json.isNullOrBlank()
    val exportFolderLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { folderUri ->
        if (folderUri == null) {
            currentStatus = "Export cancelled"
            return@rememberLauncherForActivityResult
        }
        val exportQuery = currentQuery.orEmpty()
        val exportResponse = stage2Text.orEmpty()
        val exportIr = stage3Json.orEmpty()
        if (exportQuery.isBlank() || exportResponse.isBlank() || exportIr.isBlank()) {
            currentStatus = "Nothing to export yet"
            return@rememberLauncherForActivityResult
        }
        coroutineScope.launch {
            currentStatus = "Exporting scenario..."
            runCatching {
                withContext(Dispatchers.IO) {
                    IrDemoRecordRepository.exportDemoArtifactBundle(
                        context = context.applicationContext,
                        folderUri = folderUri,
                        queryText = exportQuery,
                        responseText = exportResponse,
                        genUiJson = exportIr
                    )
                }
            }.onSuccess { result ->
                val assetSummary = if (result.failedAssetCount > 0) {
                    "${result.embeddedAssetCount} assets embedded, ${result.failedAssetCount} skipped"
                } else {
                    "${result.embeddedAssetCount} assets embedded"
                }
                currentStatus = "Exported ${result.fileName}"
                logs += AssistantLogItem(
                    title = "Exported",
                    content = "Saved portable scenario bundle: ${result.fileName}\n$assetSummary",
                    monospace = false
                )
            }.onFailure { error ->
                val message = error.message ?: error.javaClass.simpleName
                currentStatus = "Export failed"
                errorText = "Export failed: $message"
                logs += AssistantLogItem(
                    title = "Export error",
                    content = errorText.orEmpty()
                )
            }
        }
    }
    val hasConversation = inputText.isNotBlank() ||
        currentStatus.isNotBlank() ||
        errorText != null ||
        stage2Text != null ||
        stage3Json != null ||
        currentQuery != null ||
        displayRenderResult != null ||
        logs.isNotEmpty() ||
        warnings.isNotEmpty() ||
        usedFallback
    val showLanding = !isRunning &&
        displayRenderResult == null &&
        currentQuery == null &&
        logs.isEmpty() &&
        errorText == null

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = stringResource(id = R.string.genui_assistant_title),
                        style = MaterialTheme.typography.headlineSmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                },
                actions = {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(end = 6.dp)
                    ) {
                        if (hasConversation) {
                            IconButton(
                                onClick = { clearConversation() },
                                enabled = !isRunning
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.Clear,
                                    contentDescription = stringResource(id = R.string.genui_assistant_clear_content_desc),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                        if (canSaveToIrDemo) {
                            IconButton(
                                onClick = { exportFolderLauncher.launch(null) }
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.FileUpload,
                                    contentDescription = stringResource(id = R.string.genui_assistant_export_content_desc),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                        if (canSaveToIrDemo) {
                        IconButton(
                            onClick = {
                                runCatching {
                                    val savedRecord = IrDemoRecordRepository.addSavedDemoArtifact(
                                        context = context.applicationContext,
                                        queryText = currentQuery.orEmpty(),
                                        responseText = stage2Text.orEmpty(),
                                        genUiJson = stage3Json.orEmpty()
                                    )
                                    if (savedRecord != null) {
                                        val refreshed = IrDemoRecordRepository.load(context.applicationContext).orEmpty()
                                        val sourceLabel = IrDemoSessionStore.current()?.sourceLabel
                                            ?: context.getString(R.string.ir_demo_source)
                                        IrDemoSessionStore.update(
                                            sourceLabel = sourceLabel,
                                            records = refreshed
                                        )
                                        currentStatus = "Saved Demo item to IR Demo"
                                        logs += AssistantLogItem(
                                            title = "Saved",
                                            content = "Added Demo item with query, response, GenUI IR, and asset references."
                                        )
                                    }
                                }.onFailure { error ->
                                    errorText = "Failed to save response: ${error.message ?: error.javaClass.simpleName}"
                                    logs += AssistantLogItem(
                                        title = "Save error",
                                        content = errorText.orEmpty()
                                    )
                                }
                            },
                            enabled = true
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Save,
                                contentDescription = stringResource(id = R.string.genui_assistant_save_ir_content_desc),
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                        }
                        Text(
                            text = "Debug",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Switch(
                            checked = debugMode,
                            onCheckedChange = { debugMode = it }
                        )
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = genUiTopBarContainerColor(),
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        },
        bottomBar = {
            val composerAnimatedDots = rememberAnimatedDots(isRunning)
            val composerBaseStatus = resolveCompactStatusText(
                statusText = currentStatus,
                running = isRunning,
                errorText = errorText,
                hasRenderedOutput = displayRenderResult != null
            )
            val composerStatus = if (isRunning) "$composerBaseStatus$composerAnimatedDots" else composerBaseStatus
            val showComposerStatus = isRunning ||
                currentStatus.isNotBlank() ||
                !errorText.isNullOrBlank() ||
                displayRenderResult != null
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(genUiScreenOverlayColor())
                    .background(genUiBackgroundBrush())
                    .imePadding()
                    .navigationBarsPadding()
            ) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = horizontalPadding, vertical = 6.dp),
                    shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                    colors = genUiCardColors(GenUiCardTone.Neutral),
                    border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
                    elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 10.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        if (showComposerStatus) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 4.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                if (isRunning) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(12.dp),
                                        color = MaterialTheme.colorScheme.primary,
                                        strokeWidth = 1.7.dp
                                    )
                                }
                                Text(
                                    text = composerStatus,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (!errorText.isNullOrBlank()) {
                                        MaterialTheme.colorScheme.error
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            OutlinedTextField(
                                value = inputText,
                                onValueChange = { inputText = it },
                                placeholder = {
                                    Text(
                                        text = stringResource(id = R.string.genui_assistant_placeholder),
                                        style = MaterialTheme.typography.bodySmall
                                    )
                                },
                                modifier = Modifier
                                    .weight(1f)
                                    .heightIn(min = 52.dp),
                                singleLine = true,
                                maxLines = 1,
                                enabled = !isRunning,
                                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                                textStyle = MaterialTheme.typography.bodySmall
                            )

                            IconButton(
                                onClick = {
                                    val query = inputText.trim()
                                    if (query.isBlank() || isRunning) {
                                        return@IconButton
                                    }

                                    currentQuery = query
                                    isRunning = true
                                    errorText = null
                                    stage2Text = null
                                    stage3Json = null
                                    renderResult = null
                                    warnings = arrayListOf()
                                    usedFallback = false
                                    stage3InputTokens = null
                                    stage3OutputTokens = null
                                    logs.clear()
                                    resetSteps(GenUiStagePipeline.Stage.STAGE2)
                                    currentStatus = "Starting pipeline"
                                    appendDebugLogLine("Pipeline started for query: $query")
                                    PipelineRunNotifier.showRunning(
                                        context.applicationContext,
                                        status = "Starting pipeline"
                                    )

                                    coroutineScope.launch {
                                        val outcome = pipeline.execute(query) { update ->
                                            currentStatus = sanitizeUiLogText(update.message)
                                            updateSteps(update)
                                            appendDebugLogLine(update.message)
                                            update.debugLog?.let(::appendDebugLogLine)
                                            PipelineRunNotifier.showRunning(
                                                context.applicationContext,
                                                status = currentStatus
                                            )

                                            if (!update.stage2Response.isNullOrBlank()) {
                                                stage2Text = update.stage2Response
                                                upsertLogCard(
                                                    title = "Response",
                                                    content = update.stage2Response
                                                )
                                            }
                                            if (!update.stage3Json.isNullOrBlank()) {
                                                stage3Json = update.stage3Json
                                                upsertLogCard(
                                                    title = "GenUI JSON",
                                                    content = update.stage3Json,
                                                    monospace = true
                                                )
                                            }
                                            if (update.stage == GenUiStagePipeline.Stage.STAGE3 &&
                                                (update.llmInputTokens != null || update.llmOutputTokens != null)
                                            ) {
                                                stage3InputTokens = update.llmInputTokens
                                                stage3OutputTokens = update.llmOutputTokens
                                                upsertLogCard(
                                                    title = "IR Tokens",
                                                    content = buildString {
                                                        append("Input: ")
                                                        append(update.llmInputTokens?.toString() ?: "n/a")
                                                        append('\n')
                                                        append("Output: ")
                                                        append(update.llmOutputTokens?.toString() ?: "n/a")
                                                    },
                                                    monospace = true
                                                )
                                            }
                                            update.renderResult?.let { partialRender ->
                                                if (partialRender.errorMessage == null) {
                                                    renderResult = partialRender
                                                }
                                            }
                                        }
                                        when (outcome) {
                                            is GenUiStagePipeline.Outcome.Success -> {
                                                val result = outcome.result
                                                stage2Text = result.stage2Response
                                                stage3Json = result.stage3Json
                                                stage3InputTokens = result.stage3InputTokens
                                                stage3OutputTokens = result.stage3OutputTokens
                                                renderResult = result.renderResult
                                                warnings = ArrayList(result.warnings.map(::sanitizeUiLogText))
                                                usedFallback = result.usedFallback
                                                upsertLogCard(
                                                    title = "Response",
                                                    content = result.stage2Response
                                                )
                                                upsertLogCard(
                                                    title = "GenUI JSON",
                                                    content = result.stage3Json,
                                                    monospace = true
                                                )
                                                if (result.stage3InputTokens != null || result.stage3OutputTokens != null) {
                                                    upsertLogCard(
                                                        title = "IR Tokens",
                                                        content = buildString {
                                                            append("Input: ")
                                                            append(result.stage3InputTokens?.toString() ?: "n/a")
                                                            append('\n')
                                                            append("Output: ")
                                                            append(result.stage3OutputTokens?.toString() ?: "n/a")
                                                        },
                                                        monospace = true
                                                    )
                                                }
                                                appendDebugLogLine("Pipeline completed successfully.")
                                                steps.indices.forEach { i ->
                                                    steps[i] = steps[i].copy(status = PipelineStepStatus.Done)
                                                }
                                                applyStageDurations(
                                                    durations = result.stageDurationsMs,
                                                    streamDurations = result.stageStreamDurationsMs
                                                )
                                                currentStatus = "Pipeline completed"
                                                PipelineRunNotifier.showCompleted(
                                                    context.applicationContext,
                                                    status = "Rendered output is ready."
                                                )
                                                inputText = ""
                                            }

                                            is GenUiStagePipeline.Outcome.Failure -> {
                                                errorText = sanitizeUiLogText(outcome.message)
                                                appendDebugLogLine("Pipeline failed: ${outcome.message}")
                                                if (!outcome.stage2Response.isNullOrBlank()) {
                                                    stage2Text = outcome.stage2Response
                                                    upsertLogCard(
                                                        title = "Response",
                                                        content = outcome.stage2Response
                                                    )
                                                }
                                                if (!outcome.stage3Json.isNullOrBlank()) {
                                                    stage3Json = outcome.stage3Json
                                                    upsertLogCard(
                                                        title = "GenUI JSON",
                                                        content = outcome.stage3Json,
                                                        monospace = true
                                                    )
                                                }
                                                steps.indices.forEach { i ->
                                                    val step = steps[i]
                                                    val status = when {
                                                        step.stage == outcome.stage -> PipelineStepStatus.Failed
                                                        step.stage.ordinal < outcome.stage.ordinal -> PipelineStepStatus.Done
                                                        else -> PipelineStepStatus.Pending
                                                    }
                                                    steps[i] = step.copy(status = status)
                                                }
                                                applyStageDurations(
                                                    durations = outcome.stageDurationsMs,
                                                    streamDurations = outcome.stageStreamDurationsMs,
                                                    failedStage = outcome.stage
                                                )
                                                currentStatus = "Pipeline failed"
                                                PipelineRunNotifier.showFailed(
                                                    context.applicationContext,
                                                    message = errorText.orEmpty()
                                                )
                                            }
                                        }
                                        isRunning = false
                                    }
                                },
                                enabled = !isRunning && inputText.trim().isNotEmpty()
                            ) {
                                Icon(
                                    imageVector = Icons.AutoMirrored.Filled.Send,
                                    contentDescription = stringResource(id = R.string.genui_assistant_send_content_desc),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        GenUiScreenBackground(
            modifier = Modifier.fillMaxSize()
        ) { backgroundModifier ->
            LazyColumn(
                modifier = backgroundModifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .consumeWindowInsets(innerPadding)
                    .padding(horizontal = horizontalPadding, vertical = 10.dp),
                contentPadding = PaddingValues(bottom = listBottomPadding),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (showLanding) {
                    item {
                        GenUiAssistantLandingCard(
                            prompts = GenUiStarterPrompts,
                            enabled = !isRunning,
                            onPromptSelected = { selectedPrompt ->
                                inputText = selectedPrompt
                            }
                        )
                    }
                }

                if (!currentQuery.isNullOrBlank()) {
                    item {
                        PromptSummaryCard(prompt = currentQuery.orEmpty())
                    }
                }

                if (debugMode) {
                    item {
                        StageProgressCard(
                            steps = steps,
                            statusText = currentStatus,
                            running = isRunning
                        )
                    }

                    if (errorText != null) {
                        item {
                            StageCard(
                                title = "Error",
                                content = errorText.orEmpty(),
                                monospace = false,
                                tone = GenUiCardTone.Error
                            )
                        }
                    }

                    items(logs) { log ->
                        StageCard(
                            title = log.title,
                            content = log.content,
                            monospace = log.monospace,
                            tone = GenUiCardTone.Neutral
                        )
                    }

                    if (warnings.isNotEmpty()) {
                        item {
                            StageCard(
                                title = "Warnings",
                                content = warnings.joinToString("\n") { "- $it" },
                                monospace = false,
                                tone = GenUiCardTone.Warning
                            )
                        }
                    }

                    if (usedFallback) {
                        item {
                            Text(
                                text = "Fallback GenUI was used to keep rendering stable.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.tertiary
                            )
                        }
                    }
                }

                if (displayRenderResult != null) {
                    item {
                        displayRenderResult?.let { resolvedRender ->
                            GenUiNativeRenderer.RenderInline(
                                result = resolvedRender,
                                sourceDir = null,
                                onOpenExternalUrl = onOpenExternalUrl,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }
                }

                if (debugMode && !isRunning && logs.isEmpty() && errorText == null && !showLanding) {
                    item {
                        StageCard(
                            title = "How It Works",
                            content = "1) Generate a rich response from your prompt.\n2) Convert that response into GenUI JSON.\n3) Render it with native Compose components.",
                            tone = GenUiCardTone.Neutral
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PromptSummaryCard(prompt: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.72f)
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.28f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "User prompt",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = prompt,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun GenUiAssistantLandingCard(
    prompts: List<StarterPrompt>,
    enabled: Boolean,
    onPromptSelected: (String) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = genUiCardColors(GenUiCardTone.Neutral),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.72f),
                            MaterialTheme.colorScheme.surfaceContainerLow.copy(alpha = 0.68f),
                            MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.64f)
                        )
                    )
                )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 18.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        text = "Ask once. Get a rendered mobile UI.",
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = "GenUI Demo turns live answers into compact flat-spec IR and renders the result as native Compose cards, tables, and actions.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    prompts.forEach { prompt ->
                        StarterPromptChip(
                            prompt = prompt,
                            enabled = enabled,
                            onPromptSelected = onPromptSelected
                        )
                    }
                }

                DemoFeatureTile(
                    title = "Native render",
                    body = "The generated IR is shown immediately in the same mobile surface.",
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

@Composable
private fun StarterPromptChip(
    prompt: StarterPrompt,
    enabled: Boolean,
    onPromptSelected: (String) -> Unit
) {
    Card(
        modifier = Modifier.clickable(enabled = enabled) {
            onPromptSelected(prompt.prompt)
        },
        shape = RoundedCornerShape(GenUiTokens.RadiusPill),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.78f)
        ),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = prompt.icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(17.dp)
            )
            Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Text(
                    text = prompt.label,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = prompt.description,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
private fun DemoFeatureTile(
    title: String,
    body: String,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(GenUiTokens.RadiusLg),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.78f)
        ),
        border = BorderStroke(GenUiTokens.BorderSm, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 11.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = body,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

private fun sanitizeUiLogText(value: String): String {
    if (value.isBlank()) {
        return value
    }
    val withoutPrefix = value.replace(
        Regex("""(?i)\bstage\s*\d+\s*:\s*"""),
        ""
    )
    val normalizedWarnings = withoutPrefix
        .replace(
            Regex("""(?i)\bdropped inline media\b.*"""),
            "Adjusted media content for compatibility."
        )
        .replace(
            Regex("""(?i)\bdropped quick action urls?\b.*"""),
            "Adjusted quick actions for compatibility."
        )
    return normalizedWarnings
        .replace(Regex("""(?i)\bstage\s*\d+\b"""), "step")
        .replace(Regex("""\s{2,}"""), " ")
        .trim()
}

@Composable
private fun CompactStatusLine(
    statusText: String,
    running: Boolean,
    errorText: String?,
    hasRenderedOutput: Boolean
) {
    val animatedDots = rememberAnimatedDots(running)
    val oneLineError = errorText?.lineSequence()?.firstOrNull()?.trim().orEmpty()
    val baseStatus = resolveCompactStatusText(
        statusText = statusText,
        running = running,
        errorText = errorText,
        hasRenderedOutput = hasRenderedOutput
    )

    val status = if (running) "$baseStatus$animatedDots" else baseStatus
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = genUiCardColors(if (oneLineError.isNotBlank()) GenUiCardTone.Error else GenUiCardTone.Neutral),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (running) {
                CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    color = MaterialTheme.colorScheme.primary,
                    strokeWidth = 1.8.dp
                )
            }
            Text(
                text = status,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
        }
    }
}

private fun resolveCompactStatusText(
    statusText: String,
    running: Boolean,
    errorText: String?,
    hasRenderedOutput: Boolean
): String {
    val oneLineError = errorText?.lineSequence()?.firstOrNull()?.trim().orEmpty()
    return when {
        running && statusText.isNotBlank() -> statusText
        running -> "Working"
        oneLineError.isNotBlank() -> "Failed: $oneLineError"
        hasRenderedOutput -> "Completed. Rendered output is ready."
        statusText.isNotBlank() -> statusText
        else -> "Ready"
    }.replace('\n', ' ').trim()
}

@Composable
private fun rememberAnimatedDots(active: Boolean): String {
    var dotCount by remember(active) { mutableStateOf(0) }
    LaunchedEffect(active) {
        if (!active) {
            dotCount = 0
            return@LaunchedEffect
        }
        while (true) {
            delay(420)
            dotCount = (dotCount + 1) % 4
        }
    }
    return ".".repeat(dotCount)
}

@Composable
private fun StageProgressCard(
    steps: List<StageStepUi>,
    statusText: String,
    running: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = genUiCardColors(GenUiCardTone.Neutral),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Pipeline Status",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            steps.forEach { step ->
                val statusLabel = when (step.status) {
                    PipelineStepStatus.Pending -> "Pending"
                    PipelineStepStatus.Running -> "Running"
                    PipelineStepStatus.Done -> "Done"
                    PipelineStepStatus.Failed -> "Failed"
                }
                val color = when (step.status) {
                    PipelineStepStatus.Pending -> MaterialTheme.colorScheme.onSurfaceVariant
                    PipelineStepStatus.Running -> MaterialTheme.colorScheme.primary
                    PipelineStepStatus.Done -> MaterialTheme.colorScheme.secondary
                    PipelineStepStatus.Failed -> MaterialTheme.colorScheme.error
                }
                Text(
                    text = "${step.title} - $statusLabel",
                    style = MaterialTheme.typography.bodySmall,
                    color = color,
                    maxLines = 2,
                    overflow = TextOverflow.Clip
                )
                if (step.message.isNotBlank()) {
                    Text(
                        text = sanitizeUiLogText(step.message),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
            if (running || statusText.isNotBlank()) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (running) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(14.dp),
                            color = MaterialTheme.colorScheme.primary,
                            strokeWidth = 1.8.dp
                        )
                    }
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun StageCard(
    title: String,
    content: String,
    monospace: Boolean = false,
    tone: GenUiCardTone = GenUiCardTone.Neutral
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = genUiCardColors(tone),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor()),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = content,
                style = if (monospace) {
                    MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
                } else {
                    MaterialTheme.typography.bodySmall
                },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}
