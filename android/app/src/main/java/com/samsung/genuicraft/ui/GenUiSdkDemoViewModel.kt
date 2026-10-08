package com.samsung.genuicraft

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.samsung.genuicraft.sdk.*
import com.samsung.genuicraft.sdk.provider.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/** Owns the demo's session, never an Activity, across rotation and UI-mode recreation. */
internal class GenUiSdkDemoViewModel(application: Application) : AndroidViewModel(application) {
    var document by mutableStateOf<GenUiDocument?>(null)
    var renderSnapshot by mutableStateOf<GenUiRenderSnapshot?>(null)
        private set
    var generationSourceText by mutableStateOf<String?>(null)
        private set
    var status by mutableStateOf("Choose a Bixby response or enter text.")
    var working by mutableStateOf(false)
    var editorVisible by mutableStateOf(true)
    var generationMetrics by mutableStateOf<GenerationMetricsUiState?>(null)
    var lastRuntime by mutableStateOf<String?>(null)
    var lastPrecision by mutableStateOf<String?>(null)
    var generationTrace by mutableStateOf(SdkGenerationTrace())
    var workspaceRevision by mutableStateOf(0L)
        private set
    private var activeProvider: GenUiProvider? = null
    private var activeProviderKey: String? = null
    private var activeJob: Job? = null
    private val streamHandler = Handler(Looper.getMainLooper())
    private var generationRunId = 0L
    private var conversionStartedAtMs = 0L

    private fun postGenerationUpdate(runId: Long, update: () -> Unit) {
        val guardedUpdate = {
            if (generationRunId == runId && working &&
                generationTrace.phase != SdkGenerationPhase.CANCELLED
            ) update()
        }
        if (Looper.myLooper() == Looper.getMainLooper()) guardedUpdate()
        else streamHandler.post { guardedUpdate() }
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

    fun showDocument(value: GenUiDocument, message: String = "A2UI rendered by GenUICraft AAR") {
        workspaceRevision++
        generationMetrics = null
        lastRuntime = null
        lastPrecision = null
        generationTrace = SdkGenerationTrace()
        renderSnapshot = null
        generationSourceText = null
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

    fun cancelGeneration() {
        activeJob?.cancel()
        renderSnapshot = null
        generationMetrics = null
        generationTrace = generationTrace.copy(phase = SdkGenerationPhase.CANCELLED)
        status = "Cancelled · generated text retained"
    }

    fun generate(
        source: String,
        e2bModelChoiceForRun: E2bModelChoice,
        modelPathForRun: String,
        metricsForRun: Boolean,
        mtpForRun: Boolean,
        gpuPrecisionForRun: Gemma4GpuPrecision = Gemma4GpuPrecision.FP32,
        streamingRenderingForRun: Boolean = true,
        // Injectable provider keeps lifecycle tests independent of network or native model timing.
        providerFactory: () -> GenUiProvider = {
            createSdkDemoProvider(
                e2bModelChoiceForRun, modelPathForRun, metricsForRun, mtpForRun, gpuPrecisionForRun,
            )
        },
    ) {
        if (working) return
        workspaceRevision++
        document = null
        renderSnapshot = null
        generationSourceText = source
        generationMetrics = null
        lastRuntime = null
        lastPrecision = if (e2bModelChoiceForRun == E2bModelChoice.TRAINED_E2B_V10_W4) {
            when (gpuPrecisionForRun) {
                Gemma4GpuPrecision.MODEL_DEFAULT -> "Requested GPU model default"
                Gemma4GpuPrecision.FP32 -> "Requested GPU FP32"
                Gemma4GpuPrecision.FP16_CORRECTED -> "Requested GPU FP16 (corrected)"
            }
        } else null
        working = true
        editorVisible = false
        generationTrace = SdkGenerationTrace(
            phase = SdkGenerationPhase.GENERATING,
            streamingRenderingEnabled = streamingRenderingForRun,
        )
        val runId = ++generationRunId
        status = "Generating IR…" + lastPrecision?.let { " · $it" }.orEmpty()
        val sourceForRun = source
        activeJob = viewModelScope.launch {
            var captureForRun: GenUiSession? = null
            try {
                val providerKey = sdkDemoProviderKey(
                    e2bModelChoice = e2bModelChoiceForRun,
                    modelPath = modelPathForRun,
                    enableMetrics = metricsForRun,
                    enableMtp = mtpForRun,
                    gpuPrecision = gpuPrecisionForRun,
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
                    providerFactory().also {
                        activeProvider = it
                        activeProviderKey = providerKey
                    }
                }
                // Runtime identity comes from the initialized engine, even with metrics off.
                var runtimeForRun: String? = null
                var finalSnapshotForRun: GenUiRenderSnapshot? = null
                var lastPartialAtMs = 0L
                val session = GenUiSession(
                    getApplication<Application>(),
                    provider,
                    if (e2bModelChoiceForRun == E2bModelChoice.TRAINED_E2B_V10_W4)
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
                    onRenderSnapshot = { snapshot ->
                        if (snapshot.isFinal) finalSnapshotForRun = snapshot
                        postGenerationUpdate(runId) {
                            val activeAttempt = generationTrace.attempts.lastOrNull()?.number ?: 1
                            val previous = renderSnapshot
                            if (snapshot.attempt == activeAttempt &&
                                (previous == null || previous.surfaceKey != snapshot.surfaceKey ||
                                    snapshot.revision > previous.revision)
                            ) {
                                renderSnapshot = snapshot
                                generationTrace = generationTrace.copy(
                                    previewSnapshotCount = generationTrace.previewSnapshotCount +
                                        if (snapshot.isFinal) 0 else 1,
                                    firstPreviewElapsedMs = generationTrace.firstPreviewElapsedMs ?:
                                        snapshot.elapsedMs.takeIf { !snapshot.isFinal },
                                    latestReadyComponentCount = snapshot.readyComponentCount,
                                )
                            }
                        }
                    },
                )
                conversionStartedAtMs = SystemClock.elapsedRealtime()
                val result = session.convert(
                    GenUiRequest(sourceForRun), observer,
                    enableStreamingRendering = streamingRenderingForRun,
                )
                generationTrace = generationTrace.copy(
                    attempts = session.attemptSnapshots.map {
                        SdkGenerationAttempt(it.number, it.rawText, it.complete)
                    },
                )
                when (result) {
                    is GenUiConversionResult.Success -> {
                        document = result.document
                        renderSnapshot = (finalSnapshotForRun ?: renderSnapshot)?.takeIf {
                            it.isFinal && it.document.a2uiJson == result.document.a2uiJson &&
                                it.document.express == result.document.express
                        }
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
                            latestReadyComponentCount = renderSnapshot?.readyComponentCount
                                ?: generationTrace.latestReadyComponentCount,
                        )
                        generationMetrics = if (metricsForRun) session.toUiState(
                            reportedAttempts = result.attempts,
                            conversionElapsedMs = result.elapsedMs,
                        ) else null
                    }
                    is GenUiConversionResult.Failure -> {
                        renderSnapshot = null
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
                renderSnapshot = null
                generationMetrics = null
                generationTrace = generationTrace.copy(phase = SdkGenerationPhase.CANCELLED)
                status = "Cancelled · generated text retained"
                throw cancel
            } catch (error: Exception) {
                renderSnapshot = null
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
    }

    /** First native preview frame, measured against the same conversion start as SDK snapshots. */
    fun onSnapshotPresented(snapshot: GenUiRenderSnapshot) {
        if (working && !snapshot.isFinal && renderSnapshot?.surfaceKey == snapshot.surfaceKey &&
            generationTrace.firstPreviewFrameElapsedMs == null
        ) {
            generationTrace = generationTrace.copy(
                firstPreviewFrameElapsedMs =
                    (SystemClock.elapsedRealtime() - conversionStartedAtMs).coerceAtLeast(0L),
            )
        }
    }

    override fun onCleared() {
        ++generationRunId
        activeJob?.cancel()
        streamHandler.removeCallbacksAndMessages(null)
        activeProvider?.close()
        super.onCleared()
    }
}

private fun createSdkDemoProvider(
    e2bModelChoiceForRun: E2bModelChoice,
    modelPathForRun: String,
    metricsForRun: Boolean,
    mtpForRun: Boolean,
    gpuPrecisionForRun: Gemma4GpuPrecision,
): GenUiProvider =
    Gemma4Provider(
        if (e2bModelChoiceForRun == E2bModelChoice.TRAINED_E2B_V10_W4) {
            trainedE2bW4Config(
                modelPath = modelPathForRun,
                enableMetrics = metricsForRun,
                enableMtp = mtpForRun,
                gpuPrecision = gpuPrecisionForRun,
            )
        } else {
            Gemma4Config(
                modelPath = modelPathForRun,
                enableMetrics = metricsForRun,
                enableSpeculativeDecoding = mtpForRun,
            )
        },
    )

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
        speculativeDecodingEnabled = generationSessionMetrics?.speculativeDecodingEnabled,
        drafterAcceptanceRate = generationSessionMetrics?.drafterAcceptanceRate,
    )
