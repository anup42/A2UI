package com.samsung.genuicraft

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.inference.SdkGemmaModelDownload
import com.samsung.genuicraft.sdk.*
import com.samsung.genuicraft.sdk.provider.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

private const val PREFERENCE_TOKEN_METRICS_ENABLED = "token_metrics_enabled"
private const val PREFERENCE_USE_GEMMA = "use_gemma_provider"
private const val PREFERENCE_GEMMA_MODEL_SOURCE = "gemma_model_source"

/** The demo uses the published SDK AAR; it does not call the app's legacy pipeline. */
class GenUiSdkDemoActivity : ComponentActivity() {
    private var document by mutableStateOf<GenUiDocument?>(null)
    private var status by mutableStateOf("Choose a Bixby response or enter text.")
    private var working by mutableStateOf(false)
    private var editorVisible by mutableStateOf(true)
    private var tokenMetricsEnabled by mutableStateOf(true)
    private var generationMetrics by mutableStateOf<GenerationMetricsUiState?>(null)
    private var managedModelState by mutableStateOf<SdkGemmaModelDownload.State>(
        SdkGemmaModelDownload.State.Idle
    )
    private lateinit var managedModelDownload: SdkGemmaModelDownload
    private var activeProvider: GenUiProvider? = null
    private var activeProviderKey: String? = null
    private var activeJob: Job? = null
    private var lastRuntime by mutableStateOf<String?>(null)
    private var generationTrace by mutableStateOf(SdkGenerationTrace())
    private val streamHandler = Handler(Looper.getMainLooper())
    private var generationRunId = 0L

    internal fun generationTraceForTest(): SdkGenerationTrace = generationTrace
    internal fun runtimeForTest(): String? = lastRuntime
    internal fun renderedDocumentForTest(): GenUiDocument? = document
    internal fun statusForTest(): String = status

    private fun postGenerationUpdate(runId: Long, update: () -> Unit) {
        streamHandler.post {
            if (generationRunId == runId && working &&
                generationTrace.phase != SdkGenerationPhase.CANCELLED
            ) update()
        }
    }

    private fun updateGenerationAttempt(number: Int, rawText: String, complete: Boolean) {
        val attempts = generationTrace.attempts.toMutableList()
        val item = SdkGenerationAttempt(number, rawText, complete)
        val index = attempts.indexOfFirst { it.number == number }
        if (index >= 0) attempts[index] = item else attempts += item
        generationTrace = generationTrace.copy(
            attempts = attempts,
            phase = if (complete) SdkGenerationPhase.REPAIRING else SdkGenerationPhase.GENERATING,
        )
    }
    /** Optional host override used by integration tests; default opens approved web links. */
    var onSdkAction: ((GenUiAction) -> Unit)? = null

    fun showDocument(value: GenUiDocument, message: String = "A2UI rendered by GenUICraft AAR") {
        generationMetrics = null
        lastRuntime = null
        generationTrace = SdkGenerationTrace()
        document = value
        status = message
        editorVisible = false
    }

    /** Test/demo hook for rendering the same actual SDK metrics captured by another harness. */
    fun showGenerationMetrics(outputs: List<GenUiModelOutput>, elapsedMs: Long) {
        require(elapsedMs >= 0L) { "elapsedMs must not be negative." }
        generationMetrics = GenerationMetricsUiState(
            attempts = outputs.map { it.toUiMetrics() },
            reportedAttempts = outputs.size,
            conversionElapsedMs = elapsedMs,
        )
    }

    @OptIn(ExperimentalLayoutApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val cases = assets.open("genuicraft_bixby50.jsonl").bufferedReader().useLines { lines ->
            lines.filter(String::isNotBlank).map { JsonParser.parseString(it).asJsonObject }.toList()
        }
        val preferences = getSharedPreferences("genuicraft_sdk_demo", MODE_PRIVATE)
        tokenMetricsEnabled = preferences.getBoolean(PREFERENCE_TOKEN_METRICS_ENABLED, true)
        managedModelDownload = SdkGemmaModelDownload(this)
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { managedModelDownload.state.collect { managedModelState = it } }
                while (isActive) {
                    managedModelState = managedModelDownload.inspect()
                    delay(1_000)
                }
            }
        }
        setContent {
            GenUiCraftTheme {
                var settingsVisible by remember { mutableStateOf(false) }
                var mtpEnabled by remember {
                    mutableStateOf(preferences.getBoolean(PREFERENCE_E2B_MTP_ENABLED, true))
                }
                var caseIndex by remember { mutableIntStateOf(0) }
                var source by remember { mutableStateOf(cases.first().get("text").asString) }
                var useGemma by remember {
                    mutableStateOf(preferences.getBoolean(PREFERENCE_USE_GEMMA, false))
                }
                var gemmaModelSource by remember {
                    mutableStateOf(
                        GemmaModelSource.fromPreference(
                            preferences.getString(PREFERENCE_GEMMA_MODEL_SOURCE, null)
                        )
                    )
                }
                var e2bModelChoice by remember {
                    mutableStateOf(
                        E2bModelChoice.fromPreference(
                            preferences.getString(PREFERENCE_E2B_MODEL_CHOICE, null)
                        )
                    )
                }
                var officialModelPath by remember {
                    mutableStateOf(
                        preferences.getString(
                            PREFERENCE_OFFICIAL_E2B_MODEL_PATH,
                            java.io.File(
                                getExternalFilesDir(null),
                                "sdk_models/gemma-4-E2B-it.litertlm",
                            ).absolutePath,
                        ).orEmpty()
                    )
                }
                var trainedModelPath by remember {
                    mutableStateOf(
                        preferences.getString(
                            PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
                            OnDeviceModelCatalog.entries.first { it.usesTrainedSdkConverter }
                                .localFile(this@GenUiSdkDemoActivity).absolutePath,
                        ).orEmpty()
                    )
                }
                var trainedModelCheckRevision by remember { mutableIntStateOf(0) }
                var modelSetupVisible by remember { mutableStateOf(false) }
                val selectedCase = cases[caseIndex]
                val customInput = source != selectedCase.get("text").asString
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(
                        Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text("GenUICraft SDK", style = MaterialTheme.typography.headlineSmall)
                                Text(
                                    "Markdown to an interactive A2UI experience",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            TextButton(
                                onClick = { settingsVisible = true },
                                modifier = Modifier.testTag("sdk_settings_button"),
                            ) { Text("Settings") }
                        }
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = MaterialTheme.shapes.medium,
                            color = when (generationTrace.phase) {
                                SdkGenerationPhase.FAILED -> MaterialTheme.colorScheme.errorContainer
                                SdkGenerationPhase.CANCELLED -> MaterialTheme.colorScheme.tertiaryContainer
                                SdkGenerationPhase.COMPLETE -> MaterialTheme.colorScheme.primaryContainer
                                else -> MaterialTheme.colorScheme.surfaceVariant
                            },
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                            ) {
                                Column(Modifier.weight(1f)) {
                                    Text(
                                        status,
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = when (generationTrace.phase) {
                                            SdkGenerationPhase.FAILED -> MaterialTheme.colorScheme.onErrorContainer
                                            SdkGenerationPhase.CANCELLED -> MaterialTheme.colorScheme.onTertiaryContainer
                                            SdkGenerationPhase.COMPLETE -> MaterialTheme.colorScheme.onPrimaryContainer
                                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                                        },
                                    )
                                    lastRuntime?.let {
                                        Text(
                                            it,
                                            style = MaterialTheme.typography.labelSmall,
                                            modifier = Modifier.testTag("sdk_last_runtime"),
                                        )
                                    }
                                    if (tokenMetricsEnabled) generationMetrics?.let { metrics ->
                                        val decodeRate = metrics.weightedNativeDecodeTokensPerSecond
                                        Text(
                                            "${formatTokenCount(metrics.totalInputTokens)} in · " +
                                                "${formatTokenCount(metrics.totalOutputTokens)} out · " +
                                                if (decodeRate != null) {
                                                    "${formatTokensPerSecond(decodeRate)} decode"
                                                } else {
                                                    "${formatTokensPerSecond(metrics.requestAverageTokensPerSecond)} request average"
                                                },
                                            style = MaterialTheme.typography.labelSmall,
                                            modifier = Modifier.testTag("sdk_metrics_summary"),
                                        )
                                    }
                                }
                                when {
                                    working -> OutlinedButton(
                                        onClick = {
                                            activeJob?.cancel()
                                            generationMetrics = null
                                            generationTrace = generationTrace.copy(
                                                phase = SdkGenerationPhase.CANCELLED
                                            )
                                            status = "Cancelled · generated text retained"
                                        },
                                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                                    ) { Text("Cancel") }
                                    !editorVisible -> TextButton(
                                        onClick = { editorVisible = true },
                                    ) { Text("Edit input") }
                                }
                            }
                        }
                        if (working) LinearProgressIndicator(Modifier.fillMaxWidth())
                        if (!editorVisible && generationTrace.phase != SdkGenerationPhase.IDLE) {
                            Text(
                                if (customInput) {
                                    "Custom input"
                                } else {
                                    "${selectedCase.get("id").asString} · " +
                                        selectedCase.get("query").asString
                                },
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1,
                                overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                            )
                        }
                        val managedReadyFile =
                            (managedModelState as? SdkGemmaModelDownload.State.Ready)?.file
                        val trainedModelReadiness = remember(
                            trainedModelPath,
                            trainedModelCheckRevision,
                        ) {
                            trainedE2bW4Readiness(trainedModelPath)
                        }
                        val actionAvailability = sdkDemoActionAvailability(
                            working = working,
                            useGemma = useGemma,
                            e2bModelChoice = e2bModelChoice,
                            officialModelSource = gemmaModelSource,
                            managedOfficialModelReady = managedReadyFile != null,
                            trainedModelReady = trainedModelReadiness.usable,
                        )
                        if (editorVisible) {
                            OutlinedCard(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 470.dp),
                            ) {
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .verticalScroll(rememberScrollState())
                                        .padding(14.dp),
                                    verticalArrangement = Arrangement.spacedBy(10.dp),
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    ) {
                                        OutlinedButton(
                                            enabled = !working,
                                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 7.dp),
                                            onClick = {
                                                caseIndex = (caseIndex + cases.size - 1) % cases.size
                                                source = cases[caseIndex].get("text").asString
                                            },
                                        ) { Text("Previous") }
                                        Text(
                                            if (customInput) "Custom input" else "${caseIndex + 1} / ${cases.size}",
                                            modifier = Modifier.weight(1f),
                                            style = MaterialTheme.typography.labelLarge,
                                            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                                        )
                                        OutlinedButton(
                                            enabled = !working,
                                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 7.dp),
                                            onClick = {
                                                caseIndex = (caseIndex + 1) % cases.size
                                                source = cases[caseIndex].get("text").asString
                                            },
                                        ) { Text("Next") }
                                    }
                                    Text(
                                        if (customInput) {
                                            "Edited text · use Previous or Next to load a Bixby50 sample."
                                        } else {
                                            "${selectedCase.get("id").asString} · " +
                                                selectedCase.get("query").asString
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        maxLines = 2,
                                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                                    )
                                    OutlinedTextField(
                                        source,
                                        { source = it },
                                        modifier = Modifier.fillMaxWidth(),
                                        label = { Text("Markdown, text or A2UI Express") },
                                        supportingText = {
                                            Text("Paste an answer, or use a Bixby50 example above.")
                                        },
                                        minLines = 3,
                                        maxLines = 5,
                                    )
                                    Text("Generation model", style = MaterialTheme.typography.titleSmall)
                                    FlowRow(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    ) {
                                        FilterChip(
                                            selected = !useGemma,
                                            onClick = {
                                                useGemma = false
                                                modelSetupVisible = false
                                                preferences.edit()
                                                    .putBoolean(PREFERENCE_USE_GEMMA, false)
                                                    .apply()
                                            },
                                            enabled = !working,
                                            label = { Text("Gauss 30B") },
                                        )
                                        FilterChip(
                                            selected = useGemma &&
                                                e2bModelChoice == E2bModelChoice.OFFICIAL_E2B,
                                            onClick = {
                                                useGemma = true
                                                e2bModelChoice = E2bModelChoice.OFFICIAL_E2B
                                                preferences.edit()
                                                    .putBoolean(PREFERENCE_USE_GEMMA, true)
                                                    .putString(
                                                        PREFERENCE_E2B_MODEL_CHOICE,
                                                        E2bModelChoice.OFFICIAL_E2B.preferenceValue,
                                                    ).apply()
                                            },
                                            enabled = !working,
                                            modifier = Modifier.testTag("e2b_model_official"),
                                            label = { Text("Official E2B") },
                                        )
                                        FilterChip(
                                            selected = useGemma &&
                                                e2bModelChoice == E2bModelChoice.TRAINED_E2B_V10_W4,
                                            onClick = {
                                                useGemma = true
                                                e2bModelChoice = E2bModelChoice.TRAINED_E2B_V10_W4
                                                preferences.edit()
                                                    .putBoolean(PREFERENCE_USE_GEMMA, true)
                                                    .putString(
                                                        PREFERENCE_E2B_MODEL_CHOICE,
                                                        E2bModelChoice.TRAINED_E2B_V10_W4.preferenceValue,
                                                    ).apply()
                                            },
                                            enabled = !working,
                                            modifier = Modifier.testTag("e2b_model_trained_v10_w4"),
                                            label = { Text("Trained E2B") },
                                        )
                                    }
                                    Text(
                                        when {
                                            !useGemma -> "Server model · no local setup required"
                                            e2bModelChoice == E2bModelChoice.OFFICIAL_E2B &&
                                                managedReadyFile != null -> "On-device GPU · ready offline"
                                            e2bModelChoice == E2bModelChoice.OFFICIAL_E2B ->
                                                "On-device GPU · model setup required"
                                            trainedModelReadiness.usable -> "Trained on-device GPU · model ready"
                                            else -> "Trained on-device GPU · model setup required"
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = if (
                                            useGemma &&
                                            e2bModelChoice == E2bModelChoice.TRAINED_E2B_V10_W4 &&
                                            !trainedModelReadiness.usable
                                        ) MaterialTheme.colorScheme.error
                                        else MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                    if (useGemma) {
                                        TextButton(
                                            onClick = { modelSetupVisible = !modelSetupVisible },
                                            contentPadding = PaddingValues(horizontal = 0.dp, vertical = 4.dp),
                                        ) {
                                            Text(if (modelSetupVisible) "Hide model setup" else "Model setup")
                                        }
                                    }
                                    if (useGemma && modelSetupVisible) {
                                        HorizontalDivider()
                                        if (e2bModelChoice == E2bModelChoice.OFFICIAL_E2B) {
                                            Text("Official model source", style = MaterialTheme.typography.titleSmall)
                                            Row(
                                                modifier = Modifier.horizontalScroll(rememberScrollState()),
                                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            ) {
                                                FilterChip(
                                                    selected = gemmaModelSource == GemmaModelSource.MANAGED_DOWNLOAD,
                                                    onClick = {
                                                        gemmaModelSource = GemmaModelSource.MANAGED_DOWNLOAD
                                                        preferences.edit().putString(
                                                            PREFERENCE_GEMMA_MODEL_SOURCE,
                                                            GemmaModelSource.MANAGED_DOWNLOAD.preferenceValue,
                                                        ).apply()
                                                    },
                                                    enabled = !working,
                                                    modifier = Modifier.testTag("gemma_model_source_managed"),
                                                    label = { Text("Managed download") },
                                                )
                                                FilterChip(
                                                    selected = gemmaModelSource == GemmaModelSource.LOCAL_FILE,
                                                    onClick = {
                                                        gemmaModelSource = GemmaModelSource.LOCAL_FILE
                                                        preferences.edit().putString(
                                                            PREFERENCE_GEMMA_MODEL_SOURCE,
                                                            GemmaModelSource.LOCAL_FILE.preferenceValue,
                                                        ).apply()
                                                    },
                                                    enabled = !working,
                                                    modifier = Modifier.testTag("gemma_model_source_local"),
                                                    label = { Text("Local file") },
                                                )
                                            }
                                            if (gemmaModelSource == GemmaModelSource.MANAGED_DOWNLOAD) {
                                                ManagedModelDownloadPanel(
                                                    state = managedModelState,
                                                    modelFile = managedModelDownload.modelFile,
                                                    enabled = !working,
                                                    onStart = {
                                                        lifecycleScope.launch {
                                                            managedModelState = managedModelDownload.start()
                                                        }
                                                    },
                                                    onCancel = {
                                                        lifecycleScope.launch {
                                                            managedModelState = managedModelDownload.cancel()
                                                        }
                                                    },
                                                )
                                            } else {
                                                OutlinedTextField(
                                                    officialModelPath,
                                                    {
                                                        officialModelPath = it
                                                        preferences.edit().putString(
                                                            PREFERENCE_OFFICIAL_E2B_MODEL_PATH,
                                                            it,
                                                        ).apply()
                                                    },
                                                    modifier = Modifier.fillMaxWidth(),
                                                    label = { Text("Official .litertlm model path") },
                                                    singleLine = true,
                                                )
                                            }
                                        } else {
                                            Text(
                                                "GPU · MTP follows Settings and requires a bundled drafter.",
                                                style = MaterialTheme.typography.bodySmall,
                                            )
                                            OutlinedTextField(
                                                trainedModelPath,
                                                {
                                                    trainedModelPath = it
                                                    preferences.edit().putString(
                                                        PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
                                                        it,
                                                    ).apply()
                                                },
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .testTag("trained_e2b_w4_model_path"),
                                                label = { Text("Trained .litertlm model path") },
                                                singleLine = true,
                                            )
                                            Text(
                                                trainedModelReadiness.message,
                                                style = MaterialTheme.typography.bodySmall,
                                                color = if (trainedModelReadiness.usable) {
                                                    MaterialTheme.colorScheme.onSurface
                                                } else {
                                                    MaterialTheme.colorScheme.error
                                                },
                                                modifier = Modifier.testTag("trained_e2b_w4_model_readiness"),
                                            )
                                            Text(
                                                "Place the exported model at the path above. No download URL is configured.",
                                                style = MaterialTheme.typography.bodySmall,
                                            )
                                            OutlinedButton(
                                                onClick = { trainedModelCheckRevision += 1 },
                                                enabled = !working,
                                                modifier = Modifier.testTag("trained_e2b_w4_check_model"),
                                            ) { Text("Check model") }
                                        }
                                    }
                                }
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                Button(
                                    enabled = actionAvailability.convertEnabled,
                                    modifier = Modifier
                                        .weight(1f)
                                        .testTag("convert_render_button"),
                                    onClick = {
                                        // Compose can deliver a second tap before the disabled state recomposes.
                                        if (working) return@Button
                                        val useGemmaForRun = useGemma
                                        val e2bModelChoiceForRun = e2bModelChoice
                                        val modelPathForRun = if (
                                            useGemmaForRun &&
                                            e2bModelChoiceForRun == E2bModelChoice.OFFICIAL_E2B &&
                                            gemmaModelSource == GemmaModelSource.MANAGED_DOWNLOAD
                                        ) {
                                            managedReadyFile?.absolutePath ?: run {
                                                status = "Download and verify the Gemma model first."
                                                return@Button
                                            }
                                        } else if (
                                            useGemmaForRun &&
                                            e2bModelChoiceForRun ==
                                                E2bModelChoice.TRAINED_E2B_V10_W4
                                        ) {
                                            val readiness = trainedE2bW4Readiness(trainedModelPath)
                                            if (!readiness.usable) {
                                                status = readiness.message
                                                return@Button
                                            }
                                            trainedModelPath.trim()
                                        } else {
                                            officialModelPath
                                        }
                                        document = null
                                        generationMetrics = null
                                        lastRuntime = null
                                        working = true
                                        editorVisible = false
                                        generationTrace = SdkGenerationTrace(phase = SdkGenerationPhase.GENERATING)
                                        val runId = ++generationRunId
                                        status = "Generating IR…"
                                        val sourceForRun = source
                                        val metricsForRun = tokenMetricsEnabled
                                        val mtpForRun = mtpEnabled
                                        activeJob = lifecycleScope.launch {
                                            var captureForRun: GenUiSession? = null
                                            try {
                                                val providerKey = sdkDemoProviderKey(
                                                    useGemma = useGemmaForRun,
                                                    e2bModelChoice = e2bModelChoiceForRun,
                                                    modelPath = modelPathForRun,
                                                    enableMetrics = metricsForRun,
                                                    enableMtp = mtpForRun,
                                                )
                                                val provider = if (
                                                    providerKey == activeProviderKey &&
                                                    activeProvider != null
                                                ) {
                                                    activeProvider!!
                                                } else {
                                                    val providerToClose = activeProvider
                                                    activeProvider = null
                                                    activeProviderKey = null
                                                    closeProviderBeforeReplacement(providerToClose)
                                                    (if (useGemmaForRun) {
                                                        Gemma4Provider(
                                                            if (
                                                                e2bModelChoiceForRun ==
                                                                    E2bModelChoice.TRAINED_E2B_V10_W4
                                                            ) {
                                                                trainedE2bW4Config(
                                                                    modelPath = modelPathForRun,
                                                                    enableMetrics = metricsForRun,
                                                                    enableMtp = mtpForRun,
                                                                )
                                                            } else {
                                                                Gemma4Config(
                                                                    modelPath = modelPathForRun,
                                                                    enableMetrics = metricsForRun,
                                                                    enableSpeculativeDecoding = mtpForRun,
                                                                )
                                                            }
                                                        )
                                                    } else {
                                                        Gauss30bProvider()
                                                    }).also {
                                                        activeProvider = it
                                                        activeProviderKey = providerKey
                                                    }
                                                }
                                                // Runtime identity comes from the initialized engine, even with metrics off.
                                                var runtimeForRun: String? = null
                                                var lastPartialAtMs = 0L
                                                val session = GenUiSession(
                                                    this@GenUiSdkDemoActivity,
                                                    provider,
                                                    if (useGemmaForRun && e2bModelChoiceForRun == E2bModelChoice.TRAINED_E2B_V10_W4)
                                                        GenUiConversionProfile.TRAINED_E2B_V10_W4
                                                    else GenUiConversionProfile.SOURCE_BOUND,
                                                ).also { captureForRun = it }
                                                val observer = GenUiGenerationObserver(
                                                    onAttemptStarted = { number ->
                                                        lastPartialAtMs = 0L
                                                        postGenerationUpdate(runId) {
                                                            updateGenerationAttempt(number, "", complete = false)
                                                            status = "Generating IR · attempt $number"
                                                        }
                                                    },
                                                    onPartialText = { number, rawText ->
                                                        val now = SystemClock.elapsedRealtime()
                                                        if (lastPartialAtMs == 0L || now - lastPartialAtMs >= 75L) {
                                                            lastPartialAtMs = now
                                                            postGenerationUpdate(runId) {
                                                                updateGenerationAttempt(number, rawText, complete = false)
                                                            }
                                                        }
                                                    },
                                                    onAttemptCompleted = { number, output ->
                                                        runtimeForRun = output.runtime
                                                        postGenerationUpdate(runId) {
                                                            updateGenerationAttempt(number, output.text, complete = true)
                                                            lastRuntime = output.runtime
                                                            status = "Validating and repairing IR…"
                                                        }
                                                    },
                                                )
                                                val result = session.convert(GenUiRequest(sourceForRun), observer)
                                                generationTrace = generationTrace.copy(
                                                    attempts = session.attemptSnapshots.map {
                                                        SdkGenerationAttempt(it.number, it.rawText, it.complete)
                                                    },
                                                )
                                                when (result) {
                                                    is GenUiConversionResult.Success -> {
                                                        document = result.document
                                                        editorVisible = false
                                                        val stateLabel = if (result.repairKind == GenUiRepairKind.GENERATED_DSL_REPAIR) {
                                                            "Recovered"
                                                        } else {
                                                            "Ready"
                                                        }
                                                        status = "$stateLabel · ${result.elapsedMs} ms · ${result.attempts} attempt(s)"
                                                        generationTrace = generationTrace.copy(
                                                            phase = SdkGenerationPhase.COMPLETE,
                                                            finalIr = result.document.express,
                                                            repairKind = result.repairKind,
                                                            warnings = result.warnings,
                                                        )
                                                        generationMetrics = if (metricsForRun) session.toUiState(
                                                            reportedAttempts = result.attempts,
                                                            conversionElapsedMs = result.elapsedMs,
                                                        ) else null
                                                    }
                                                    is GenUiConversionResult.Failure -> {
                                                        status = "Conversion failed"
                                                        generationTrace = generationTrace.copy(
                                                            phase = SdkGenerationPhase.FAILED,
                                                            error = result.message,
                                                        )
                                                        generationMetrics = if (metricsForRun) session.toUiState(
                                                            reportedAttempts = result.attempts,
                                                            conversionElapsedMs = result.elapsedMs,
                                                        ) else null
                                                    }
                                                }
                                                lastRuntime = runtimeForRun
                                            } catch (cancel: kotlinx.coroutines.CancellationException) {
                                                generationMetrics = null
                                                generationTrace = generationTrace.copy(phase = SdkGenerationPhase.CANCELLED)
                                                status = "Cancelled · generated text retained"
                                                throw cancel
                                            } catch (error: Exception) {
                                                generationMetrics = null
                                                generationTrace = generationTrace.copy(
                                                    phase = SdkGenerationPhase.FAILED,
                                                    error = error.message ?: error.javaClass.simpleName,
                                                )
                                                status = "Generation failed"
                                            } finally {
                                                captureForRun?.let { capture ->
                                                    generationTrace = generationTrace.copy(
                                                        attempts = capture.attemptSnapshots.map {
                                                            SdkGenerationAttempt(it.number, it.rawText, it.complete)
                                                        },
                                                    )
                                                }
                                                working = false
                                            }
                                        }
                                    },
                                ) { Text("Generate UI") }
                                OutlinedButton(
                                    enabled = actionAvailability.renderEnabled,
                                    modifier = Modifier
                                        .weight(1f)
                                        .testTag("render_a2ui_button"),
                                    onClick = {
                                    generationMetrics = null
                                    runCatching { GenUiCompiler.compile(source) }
                                        .onSuccess { showDocument(it, "Renderer only · no model call") }
                                        .onFailure { status = "Invalid A2UI: ${it.message}" }
                                    },
                                ) { Text("Render A2UI") }
                            }
                        }
                        SdkGenerationWorkspace(
                            trace = generationTrace,
                            document = document,
                            modifier = Modifier.weight(1f).fillMaxWidth(),
                            metrics = {
                                if (tokenMetricsEnabled) {
                                    generationMetrics?.let { metrics -> GenerationMetricsPanel(metrics) }
                                }
                            },
                            onAction = { action ->
                                val handler = onSdkAction
                                if (handler != null) handler(action)
                                else if (action.name == "openUrl") {
                                    val uri = Uri.parse(action.parameters["url"].orEmpty())
                                    if (uri.scheme in listOf("http", "https")) runCatching { startActivity(Intent(Intent.ACTION_VIEW, uri)) }
                                }
                            },
                        )
                    }
                }
                if (settingsVisible) {
                    AlertDialog(
                        onDismissRequest = { settingsVisible = false },
                        title = { Text("Demo settings") },
                        text = {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 520.dp)
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Column(Modifier.weight(1f)) {
                                        Text("Performance metrics", style = MaterialTheme.typography.titleSmall)
                                        Text(
                                            "Show token counts and timing after each run.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    Switch(
                                        checked = tokenMetricsEnabled,
                                        onCheckedChange = { enabled ->
                                            tokenMetricsEnabled = enabled
                                            generationMetrics = null
                                            preferences.edit()
                                                .putBoolean(PREFERENCE_TOKEN_METRICS_ENABLED, enabled)
                                                .apply()
                                        },
                                        enabled = !working,
                                        modifier = Modifier.testTag(TOKEN_METRICS_SWITCH_TEST_TAG)
                                            .semantics { contentDescription = "Performance metrics" },
                                    )
                                }
                                HorizontalDivider()
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Column(Modifier.weight(1f)) {
                                        Text("MTP acceleration", style = MaterialTheme.typography.titleSmall)
                                        Text(
                                            "Uses a bundled drafter on supported Gemma models.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    Switch(
                                        checked = mtpEnabled,
                                        onCheckedChange = { enabled ->
                                            mtpEnabled = enabled
                                            preferences.edit().putBoolean(PREFERENCE_E2B_MTP_ENABLED, enabled).apply()
                                        },
                                        enabled = !working,
                                        modifier = Modifier.testTag("e2b_mtp_switch")
                                            .semantics { contentDescription = "MTP drafter" },
                                    )
                                }
                                Text(
                                    "Applies on the next run. Turn it off for older packages without a drafter. Gauss is unaffected.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                lastRuntime?.let {
                                    HorizontalDivider()
                                    Text("Last runtime", style = MaterialTheme.typography.labelLarge)
                                    Text(it, style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        },
                        confirmButton = {
                            TextButton(onClick = { settingsVisible = false }) { Text("Done") }
                        },
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        ++generationRunId
        activeJob?.cancel()
        streamHandler.removeCallbacksAndMessages(null)
        activeProvider?.close()
        super.onDestroy()
    }
}

@Composable
private fun ManagedModelDownloadPanel(
    state: SdkGemmaModelDownload.State,
    modelFile: java.io.File,
    enabled: Boolean,
    onStart: () -> Unit,
    onCancel: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text("Official Gemma 4 E2B · about 2.59 GB", style = MaterialTheme.typography.bodyMedium)
        when (state) {
            SdkGemmaModelDownload.State.Idle -> {
                Text("Not downloaded", style = MaterialTheme.typography.bodySmall)
                Button(
                    onClick = onStart,
                    enabled = enabled,
                    modifier = Modifier.testTag("gemma_download_action"),
                ) { Text("Download model") }
            }
            is SdkGemmaModelDownload.State.Downloading -> {
                val hasTotal = state.totalBytes > 0L
                val progress = if (hasTotal) {
                    (state.downloadedBytes.toDouble() / state.totalBytes.toDouble())
                        .coerceIn(0.0, 1.0)
                } else {
                    null
                }
                if (progress != null) {
                    LinearProgressIndicator(
                        progress = { progress.toFloat() },
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                val progressText = if (hasTotal) {
                    val percent = (progress!! * 100.0).toInt()
                    "$percent% · ${formatDownloadBytes(state.downloadedBytes)} / " +
                        formatDownloadBytes(state.totalBytes)
                } else {
                    formatDownloadBytes(state.downloadedBytes)
                }
                Text(
                    if (state.paused) "Paused · $progressText" else "Downloading · $progressText",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedButton(
                    onClick = onCancel,
                    enabled = enabled,
                    modifier = Modifier.testTag("gemma_download_cancel"),
                ) { Text("Cancel download") }
            }
            SdkGemmaModelDownload.State.Verifying -> {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Text("Verifying downloaded model…", style = MaterialTheme.typography.bodySmall)
                OutlinedButton(
                    onClick = onCancel,
                    enabled = enabled,
                    modifier = Modifier.testTag("gemma_download_cancel"),
                ) { Text("Cancel") }
            }
            is SdkGemmaModelDownload.State.Ready -> {
                Text("Ready · verified", style = MaterialTheme.typography.bodyMedium)
                Text("Saved on this device · available offline", style = MaterialTheme.typography.bodySmall)
            }
            is SdkGemmaModelDownload.State.Error -> {
                Text(
                    "Download failed: ${state.message}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
                Button(
                    onClick = onStart,
                    enabled = enabled,
                    modifier = Modifier.testTag("gemma_download_action"),
                ) { Text("Retry download") }
            }
        }
        if (state !is SdkGemmaModelDownload.State.Ready) {
            Text("Download once to use the model offline.", style = MaterialTheme.typography.bodySmall)
        }
    }
}

/** Formats SDK-owned native measurements for this view. */
private fun GenUiSession.toUiState(reportedAttempts: Int, conversionElapsedMs: Long) =
    GenerationMetricsUiState(
        attempts = attemptSnapshots.map { attempt ->
            attempt.output?.toUiMetrics(attempt.elapsedNanos) ?: GenerationAttemptUiMetrics(
                inputTokens = null, outputTokens = null,
                nativeDecodeTokensPerSecond = null, providerCallElapsedNanos = attempt.elapsedNanos,
            )
        },
        reportedAttempts = reportedAttempts,
        conversionElapsedMs = conversionElapsedMs,
    )

@Composable
private fun GenerationMetricsPanel(metrics: GenerationMetricsUiState) {
    var detailsExpanded by remember(metrics) { mutableStateOf(false) }
    val nativeDecodeRate = metrics.weightedNativeDecodeTokensPerSecond
    val summaryRate = if (nativeDecodeRate != null) {
        "Native decode: ${formatTokensPerSecond(nativeDecodeRate)}"
    } else {
        "Request average: ${formatTokensPerSecond(metrics.requestAverageTokensPerSecond)}"
    }
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .testTag(GENERATION_METRICS_PANEL_TEST_TAG),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Performance", style = MaterialTheme.typography.titleSmall)
                    Text(
                        "${formatElapsedMillis(metrics.conversionElapsedMs)} · $summaryRate",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                TextButton(onClick = { detailsExpanded = !detailsExpanded }) {
                    Text(if (detailsExpanded) "Hide details" else "Details")
                }
            }
            Text(
                "${formatTokenCount(metrics.totalInputTokens)} in · " +
                    "${formatTokenCount(metrics.totalOutputTokens)} out · " +
                    "${metrics.reportedAttempts} attempt(s)",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (detailsExpanded) {
                Text(
                    "Token counts include thinking when reported. Native decode excludes startup and prefill.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 220.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    if (metrics.attempts.size != metrics.reportedAttempts) {
                        Text(
                            "Metrics returned for ${metrics.attempts.size} of " +
                                "${metrics.reportedAttempts} attempts.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    metrics.attempts.forEachIndexed { index, attempt ->
                        if (index > 0) HorizontalDivider(Modifier.padding(vertical = 3.dp))
                        Column(
                            modifier = Modifier.testTag(
                                "$GENERATION_METRICS_ATTEMPT_TEST_TAG_PREFIX${index + 1}"
                            ),
                            verticalArrangement = Arrangement.spacedBy(2.dp),
                        ) {
                            Text("Attempt ${index + 1}", style = MaterialTheme.typography.labelLarge)
                            Text(
                                "Actual tokens · input: ${formatTokenCount(attempt.inputTokens)} " +
                                    "· output: ${formatTokenCount(attempt.outputTokens)}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Text(
                                "Native decode: " +
                                    "${formatTokensPerSecond(attempt.nativeDecodeTokensPerSecond)} " +
                                    "· Provider-call wall: " +
                                    formatElapsedNanos(attempt.providerCallElapsedNanos),
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Text(
                                "Request average: " +
                                    formatTokensPerSecond(attempt.requestAverageTokensPerSecond),
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                    Text(
                        "Native decode is reported by the runtime and excludes startup and " +
                            "validation. Conversion total covers provider calls, generation, " +
                            "validation, and repair attempts.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        "Request average uses actual output tokens divided by provider-call wall " +
                            "time; it includes any network/startup, prefill, reasoning, and decode.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}
