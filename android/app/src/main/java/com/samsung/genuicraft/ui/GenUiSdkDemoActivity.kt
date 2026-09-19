package com.samsung.genuicraft

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.gson.JsonParser
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
    /** Optional host override used by integration tests; default opens approved web links. */
    var onSdkAction: ((GenUiAction) -> Unit)? = null

    fun showDocument(value: GenUiDocument, message: String = "A2UI rendered by GenUICraft AAR") {
        generationMetrics = null
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
                            java.io.File(
                                getExternalFilesDir(null),
                                "sdk_models/e2b_v10_w4.litertlm",
                            ).absolutePath,
                        ).orEmpty()
                    )
                }
                var trainedModelCheckRevision by remember { mutableIntStateOf(0) }
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(Modifier.fillMaxSize().safeDrawingPadding().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("GenUICraft SDK · Bixby50", style = MaterialTheme.typography.titleLarge)
                        Text(status, style = MaterialTheme.typography.bodySmall)
                        if (working) LinearProgressIndicator(Modifier.fillMaxWidth())
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = { editorVisible = !editorVisible }) { Text(if (editorVisible) "Hide input" else "Edit input") }
                            if (working) OutlinedButton(onClick = {
                                activeJob?.cancel()
                                generationMetrics = null
                                status = "Cancelled"
                            }) { Text("Cancel") }
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
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
                                modifier = Modifier.testTag(TOKEN_METRICS_SWITCH_TEST_TAG),
                            )
                            Text("Token metrics", modifier = Modifier.padding(horizontal = 12.dp))
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
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 300.dp)
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    OutlinedButton(enabled = !working, onClick = {
                                        caseIndex = (caseIndex + cases.size - 1) % cases.size
                                        source = cases[caseIndex].get("text").asString
                                    }) { Text("Previous") }
                                    Text(cases[caseIndex].get("id").asString, modifier = Modifier.padding(top = 12.dp))
                                    OutlinedButton(enabled = !working, onClick = {
                                        caseIndex = (caseIndex + 1) % cases.size
                                        source = cases[caseIndex].get("text").asString
                                    }) { Text("Next") }
                                }
                                OutlinedTextField(
                                    source,
                                    { source = it },
                                    modifier = Modifier.fillMaxWidth(),
                                    label = { Text("Markdown, text or A2UI") },
                                    minLines = 3,
                                    maxLines = 5,
                                )
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Switch(
                                        checked = useGemma,
                                        onCheckedChange = { enabled ->
                                            useGemma = enabled
                                            preferences.edit()
                                                .putBoolean(PREFERENCE_USE_GEMMA, enabled)
                                                .apply()
                                        },
                                        enabled = !working,
                                    )
                                    Text(
                                        if (useGemma) "Gemma 4 E2B · on device" else "Gauss 30B · server",
                                        modifier = Modifier.padding(horizontal = 12.dp),
                                    )
                                }
                                if (useGemma) {
                                    Text("E2B model", style = MaterialTheme.typography.titleSmall)
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        RadioButton(
                                            selected = e2bModelChoice == E2bModelChoice.OFFICIAL_E2B,
                                            onClick = {
                                                e2bModelChoice = E2bModelChoice.OFFICIAL_E2B
                                                preferences.edit().putString(
                                                    PREFERENCE_E2B_MODEL_CHOICE,
                                                    E2bModelChoice.OFFICIAL_E2B.preferenceValue,
                                                ).apply()
                                            },
                                            enabled = !working,
                                            modifier = Modifier.testTag("e2b_model_official"),
                                        )
                                        Text("Official E2B · GPU")
                                    }
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        RadioButton(
                                            selected =
                                                e2bModelChoice == E2bModelChoice.TRAINED_E2B_V10_W4,
                                            onClick = {
                                                e2bModelChoice = E2bModelChoice.TRAINED_E2B_V10_W4
                                                preferences.edit().putString(
                                                    PREFERENCE_E2B_MODEL_CHOICE,
                                                    E2bModelChoice.TRAINED_E2B_V10_W4.preferenceValue,
                                                ).apply()
                                            },
                                            enabled = !working,
                                            modifier = Modifier.testTag("e2b_model_trained_v10_w4"),
                                        )
                                        Text("Trained E2B v10 · W4 · GPU")
                                    }
                                    if (e2bModelChoice == E2bModelChoice.OFFICIAL_E2B) {
                                        Text(
                                            "Official model source",
                                            style = MaterialTheme.typography.titleSmall,
                                        )
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            RadioButton(
                                                selected =
                                                    gemmaModelSource == GemmaModelSource.MANAGED_DOWNLOAD,
                                                onClick = {
                                                    gemmaModelSource = GemmaModelSource.MANAGED_DOWNLOAD
                                                    preferences.edit().putString(
                                                        PREFERENCE_GEMMA_MODEL_SOURCE,
                                                        GemmaModelSource.MANAGED_DOWNLOAD.preferenceValue,
                                                    ).apply()
                                                },
                                                enabled = !working,
                                                modifier = Modifier.testTag(
                                                    "gemma_model_source_managed"
                                                ),
                                            )
                                            Text("Download model")
                                        }
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            RadioButton(
                                                selected =
                                                    gemmaModelSource == GemmaModelSource.LOCAL_FILE,
                                                onClick = {
                                                    gemmaModelSource = GemmaModelSource.LOCAL_FILE
                                                    preferences.edit().putString(
                                                        PREFERENCE_GEMMA_MODEL_SOURCE,
                                                        GemmaModelSource.LOCAL_FILE.preferenceValue,
                                                    ).apply()
                                                },
                                                enabled = !working,
                                                modifier = Modifier.testTag(
                                                    "gemma_model_source_local"
                                                ),
                                            )
                                            Text("Use local file · advanced")
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
                                            "GPU · MTP unavailable in this target-only export.",
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
                                            modifier = Modifier.testTag(
                                                "trained_e2b_w4_model_readiness"
                                            ),
                                        )
                                        Text(
                                            "Place the exported model at the path above. " +
                                                "No download URL is configured.",
                                            style = MaterialTheme.typography.bodySmall,
                                        )
                                        OutlinedButton(
                                            onClick = { trainedModelCheckRevision += 1 },
                                            enabled = !working,
                                            modifier = Modifier.testTag(
                                                "trained_e2b_w4_check_model"
                                            ),
                                        ) {
                                            Text("Check model")
                                        }
                                    }
                                }
                            }
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Button(
                                    enabled = actionAvailability.convertEnabled,
                                    modifier = Modifier.testTag("convert_render_button"),
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
                                        working = true
                                        status = "Generating A2UI…"
                                        val sourceForRun = source
                                        val metricsForRun = tokenMetricsEnabled
                                        activeJob = lifecycleScope.launch {
                                            try {
                                                val providerKey = sdkDemoProviderKey(
                                                    useGemma = useGemmaForRun,
                                                    e2bModelChoice = e2bModelChoiceForRun,
                                                    modelPath = modelPathForRun,
                                                    enableMetrics = metricsForRun,
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
                                                                )
                                                            } else {
                                                                Gemma4Config(
                                                                    modelPath = modelPathForRun,
                                                                    enableMetrics = metricsForRun,
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
                                                val metricsProvider = if (metricsForRun) {
                                                    MetricsRecordingProvider(provider)
                                                } else {
                                                    null
                                                }
                                                val converterProvider = metricsProvider ?: provider
                                                val result = if (
                                                    useGemmaForRun &&
                                                    e2bModelChoiceForRun ==
                                                        E2bModelChoice.TRAINED_E2B_V10_W4
                                                ) {
                                                    GenUiTrainedConverter(
                                                        this@GenUiSdkDemoActivity,
                                                        converterProvider,
                                                    ).convert(GenUiRequest(sourceForRun))
                                                } else {
                                                    GenUiConverter(
                                                        this@GenUiSdkDemoActivity,
                                                        converterProvider,
                                                    ).convert(GenUiRequest(sourceForRun))
                                                }
                                                when (result) {
                                                    is GenUiConversionResult.Success -> {
                                                        showDocument(
                                                            result.document,
                                                            "${result.provider} · " +
                                                                "${result.elapsedMs} ms · " +
                                                                "${result.attempts} attempt(s)"
                                                        )
                                                        generationMetrics = metricsProvider?.toUiState(
                                                            reportedAttempts = result.attempts,
                                                            conversionElapsedMs = result.elapsedMs,
                                                        )
                                                    }
                                                    is GenUiConversionResult.Failure -> {
                                                        status = "Conversion failed: ${result.message}"
                                                        generationMetrics = metricsProvider?.toUiState(
                                                            reportedAttempts = result.attempts,
                                                            conversionElapsedMs = result.elapsedMs,
                                                        )
                                                    }
                                                }
                                            } catch (cancel: kotlinx.coroutines.CancellationException) {
                                                generationMetrics = null
                                                throw cancel
                                            } catch (error: Exception) {
                                                generationMetrics = null
                                                status = "Error: ${error.message}"
                                            } finally {
                                                working = false
                                            }
                                        }
                                    },
                                ) { Text("Convert + render") }
                                OutlinedButton(
                                    enabled = actionAvailability.renderEnabled,
                                    modifier = Modifier.testTag("render_a2ui_button"),
                                    onClick = {
                                    generationMetrics = null
                                    runCatching { GenUiCompiler.compile(source) }
                                        .onSuccess { showDocument(it, "Renderer only · no model call") }
                                        .onFailure { status = "Invalid A2UI: ${it.message}" }
                                    },
                                ) { Text("Render A2UI") }
                            }
                        }
                        if (tokenMetricsEnabled) {
                            generationMetrics?.let { metrics -> GenerationMetricsPanel(metrics) }
                        }
                        document?.let { output ->
                          key(output) {
                            GenUiContent(output, Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState())) { action ->
                                val handler = onSdkAction
                                if (handler != null) handler(action)
                                else if (action.name == "openUrl") {
                                    val uri = Uri.parse(action.parameters["url"].orEmpty())
                                    if (uri.scheme in listOf("http", "https")) runCatching { startActivity(Intent(Intent.ACTION_VIEW, uri)) }
                                }
                            }
                          }
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        activeJob?.cancel()
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

/** Records provider-returned metrics without changing routing through [GenUiProvider.id]. */
private class MetricsRecordingProvider(
    private val delegate: GenUiProvider,
) : GenUiProvider {
    override val id: String
        get() = delegate.id

    private val attempts = mutableListOf<GenerationAttemptUiMetrics>()

    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
        val started = System.nanoTime()
        var output: GenUiModelOutput? = null
        try {
            return delegate.generate(prompt).also { output = it }
        } finally {
            val elapsedNanos = (System.nanoTime() - started).coerceAtLeast(0L)
            val attempt = output?.toUiMetrics(elapsedNanos) ?: GenerationAttemptUiMetrics(
                inputTokens = null,
                outputTokens = null,
                nativeDecodeTokensPerSecond = null,
                providerCallElapsedNanos = elapsedNanos,
            )
            synchronized(attempts) { attempts += attempt }
        }
    }

    fun toUiState(reportedAttempts: Int, conversionElapsedMs: Long): GenerationMetricsUiState =
        GenerationMetricsUiState(
            attempts = synchronized(attempts) { attempts.toList() },
            reportedAttempts = reportedAttempts,
            conversionElapsedMs = conversionElapsedMs,
        )

    override fun close() = delegate.close()

    override suspend fun closeAndAwait() = delegate.closeAndAwait()
}

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
            verticalArrangement = Arrangement.spacedBy(3.dp),
        ) {
            Text("Token metrics", style = MaterialTheme.typography.titleMedium)
            Text(
                "Input: ${formatTokenCount(metrics.totalInputTokens)} · " +
                    "Output: ${formatTokenCount(metrics.totalOutputTokens)}",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(summaryRate, style = MaterialTheme.typography.bodyMedium)
            Text(
                "${metrics.reportedAttempts} attempt(s) · Total: " +
                    formatElapsedMillis(metrics.conversionElapsedMs),
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                "Includes thinking when reported; native decode excludes startup/prefill.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(onClick = { detailsExpanded = !detailsExpanded }) {
                Text(if (detailsExpanded) "Hide details" else "Details")
            }
            if (detailsExpanded) {
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
