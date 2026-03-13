package com.samsung.genuicraft

import android.content.Context
import android.util.Base64
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.withContext
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URLEncoder
import java.net.URL
import java.net.UnknownHostException
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.Instant
import java.util.Locale
import kotlin.math.min

class GenUiStagePipeline(private val appContext: Context) {
    enum class Stage {
        STAGE2,
        STAGE3,
        STAGE4
    }

    data class StageUpdate(
        val stage: Stage,
        val message: String
    )

    data class PipelineResult(
        val queryText: String,
        val stage2Prompt: String,
        val stage2Response: String,
        val stage3Prompt: String,
        val stage3SystemPrompt: String?,
        val stage3Json: String,
        val stageDurationsMs: Map<Stage, Long>,
        val stageStreamDurationsMs: Map<Stage, Long>,
        val usedFallback: Boolean,
        val warnings: List<String>,
        val renderResult: GenUiNativeRenderer.RenderResult
    )

    sealed interface Outcome {
        data class Success(val result: PipelineResult) : Outcome
        data class Failure(
            val stage: Stage,
            val message: String,
            val stage2Response: String? = null,
            val stage3Json: String? = null,
            val stageDurationsMs: Map<Stage, Long> = emptyMap(),
            val stageStreamDurationsMs: Map<Stage, Long> = emptyMap()
        ) : Outcome
    }

    suspend fun execute(
        queryText: String,
        onStageUpdate: (StageUpdate) -> Unit
    ): Outcome = withContext(Dispatchers.IO) {
        val stageDurationsMs = linkedMapOf<Stage, Long>()
        val stageStreamDurationsMs = linkedMapOf<Stage, Long>()
        fun markDuration(stage: Stage, startMs: Long) {
            if (startMs <= 0L) return
            stageDurationsMs[stage] = (System.currentTimeMillis() - startMs).coerceAtLeast(0L)
        }
        fun markStreamDuration(stage: Stage, durationMs: Long?) {
            val duration = durationMs ?: return
            stageStreamDurationsMs[stage] = (stageStreamDurationsMs[stage] ?: 0L) + duration.coerceAtLeast(0L)
        }

        val normalizedQuery = queryText.trim()
        if (normalizedQuery.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Query is empty."
            )
        }

        val provider = InferenceBackendSettings.getProvider(appContext)
        val selectedModel = GeminiModelSettings.getSelectedModel(appContext)
        val localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(appContext)
        val localModelPath = InferenceBackendSettings.getLocalModelPath(appContext)
        val isLocalServer = provider == InferenceBackendSettings.Provider.LOCAL_SERVER
        val stage2MaxOutputTokens = if (isLocalServer) LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS else STAGE2_MAX_OUTPUT_TOKENS
        val stage3MaxOutputTokens = if (isLocalServer) LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS else STAGE3_MAX_OUTPUT_TOKENS
        val stage3RepairMaxOutputTokens = stage3MaxOutputTokens

        val apiKey = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            BuildConfig.GEMINI_API_KEY.trim()
        } else {
            ""
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI && apiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Gemini API key is missing. Set GEMINI_API_KEY before building the app."
            )
        }
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER && localServerBaseUrl.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Local server URL is missing. Open Settings and configure Local Server."
            )
        }
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            postUpdate(onStageUpdate, Stage.STAGE2, "Checking local server connectivity")
            val health = checkLocalServerHealth(localServerBaseUrl)
            if (health != null) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE2,
                    message = health
                )
            }
        }

        val responseTemplate = runCatching { loadPromptAsset(STAGE2_PROMPT_ASSET) }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE2,
                    message = "Could not load stage 2 prompt: ${it.message ?: it.javaClass.simpleName}"
                )
            }

        val stage2Prompt = renderPrompt(
            responseTemplate,
            "query_text" to normalizedQuery,
            "intent" to "",
            "tags" to ""
        )

        val genUiTemplate = runCatching { loadPromptAsset(STAGE3_PROMPT_ASSET) }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}"
                )
            }
        val promptContext = prepareStage3PromptContext(genUiTemplate)
        val stage3CacheDeferred = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            async(Dispatchers.IO) {
                ensureStage3InstructionCache(
                    apiKey = apiKey,
                    model = selectedModel,
                    systemPrompt = promptContext.systemPrompt
                )
            }
        } else {
            null
        }

        postUpdate(onStageUpdate, Stage.STAGE2, "Fetching response")
        val stage2StartedAtMs = System.currentTimeMillis()
        val stage2Call = generateWithRetry(
            provider = provider,
            apiKey = apiKey,
            model = selectedModel,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath,
            prompt = stage2Prompt,
            systemPrompt = null,
            temperature = 0.3,
            maxOutputTokens = stage2MaxOutputTokens,
            jsonMode = false,
            enableGoogleSearch = provider == InferenceBackendSettings.Provider.GEMINI,
            allowCachedContent = false,
            structuredOutput = false
        )
        markStreamDuration(Stage.STAGE2, stage2Call.streamDurationMs)
        if (stage2Call.error != null) {
            markDuration(Stage.STAGE2, stage2StartedAtMs)
            stage3CacheDeferred?.cancel()
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = stage2Call.error,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        val stage2ResponseRaw = stage2Call.text.trim()
        if (stage2ResponseRaw.isBlank()) {
            markDuration(Stage.STAGE2, stage2StartedAtMs)
            stage3CacheDeferred?.cancel()
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Stage 2 returned empty output.",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        val stage2WithActions = ensureFlightQuickActions(
            responseText = stage2ResponseRaw,
            queryText = normalizedQuery
        )
        val stage2WithTravelMedia = ensureTravelInlineMedia(
            responseText = stage2WithActions,
            queryText = normalizedQuery
        )
        val stage2Response = normalizeUrlTokensForDisplay(stage2WithTravelMedia)
        val normalizedBareDomains = stage2Response != stage2WithTravelMedia
        val injectedTravelMedia = stage2WithTravelMedia != stage2WithActions
        markDuration(Stage.STAGE2, stage2StartedAtMs)
        val stage3Prompt = buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = stage2Response,
            assets = emptyList()
        )
        val localStage3SystemPromptCacheKey = if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            buildLocalSystemPromptCacheKey(
                systemPrompt = promptContext.systemPrompt
            )
        } else {
            null
        }
        val warnings = mutableListOf<String>()
        if (provider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Using Gemini model: $selectedModel"
        } else {
            warnings += "Using local server: $localServerBaseUrl"
            warnings += "Local model path: $localModelPath"
            warnings += "Local token caps: stage2=$stage2MaxOutputTokens, stage3=$stage3MaxOutputTokens"
            if (!localStage3SystemPromptCacheKey.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache key: ${localStage3SystemPromptCacheKey.take(16)}..."
            }
        }
        if (normalizedBareDomains) {
            warnings += "Normalized bare source/action domains to https URLs."
        }
        if (injectedTravelMedia) {
            warnings += "Added fallback inline media URLs for travel content."
        }
        val stage3Cache = if (stage3CacheDeferred != null) {
            runCatching { stage3CacheDeferred.await() }
                .getOrElse {
                    CacheSetupResult(
                        name = null,
                        created = false,
                        error = it.message ?: it.javaClass.simpleName
                    )
                }
        } else {
            CacheSetupResult(name = null, created = false, error = null)
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI) {
            if (stage3Cache.name != null && !stage3Cache.created) {
                warnings += "Stage 3 instruction cache hit."
            }
            if (stage3Cache.created) {
                warnings += "Stage 3 instruction cache created."
            }
            stage3Cache.error?.let {
                warnings += "Stage 3 instruction cache unavailable ($it). Using direct prompt."
            }
        }
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            val primeError = ensureLocalSystemPromptCache(
                localServerBaseUrl = localServerBaseUrl,
                localModelPath = localModelPath,
                cacheKey = localStage3SystemPromptCacheKey,
                systemPrompt = promptContext.systemPrompt
            )
            if (primeError == null && !localStage3SystemPromptCacheKey.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache primed."
            } else if (!primeError.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache prime failed ($primeError)."
            }
        }
        val localSendStage3SystemPrompt = shouldSendLocalSystemPrompt(localStage3SystemPromptCacheKey)

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            provider = provider,
            apiKey = apiKey,
            model = selectedModel,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath,
            prompt = stage3Prompt,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            temperature = 0.2,
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            structuredOutput = provider == InferenceBackendSettings.Provider.GEMINI,
            localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
            localSendSystemPrompt = localSendStage3SystemPrompt
        )
        markStreamDuration(Stage.STAGE3, stage3Call.streamDurationMs)

        if (stage3Call.error != null) {
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = stage3Call.error,
                stage2Response = stage2Response,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        var stage3JsonElement = extractJsonElement(stage3Call.text)
        var usedFallback = false

        if (stage3JsonElement == null) {
            warnings += "Stage 3 JSON parse failed; running repair pass."
            val repairCall = generateWithRetry(
                provider = provider,
                apiKey = apiKey,
                model = selectedModel,
                localServerBaseUrl = localServerBaseUrl,
                localModelPath = localModelPath,
                prompt = buildRepairPrompt(stage3Call.text),
                systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
                temperature = 0.2,
                maxOutputTokens = stage3RepairMaxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = stage3Cache.name,
                allowCachedContent = true,
                structuredOutput = provider == InferenceBackendSettings.Provider.GEMINI,
                localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
                localSendSystemPrompt = localSendStage3SystemPrompt
            )
            markStreamDuration(Stage.STAGE3, repairCall.streamDurationMs)
            if (repairCall.error == null) {
                stage3JsonElement = extractJsonElement(repairCall.text)
            }
        }

        if (stage3JsonElement == null) {
            warnings += "Stage 3 fallback JSON was used."
            stage3JsonElement = buildFallbackGenUi(stage2Response)
            usedFallback = true
        }

        val normalizedGenUi = normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        val stage2HasInlineImage = hasInlineImageUrl(stage2Response)
        val stage2HasInlineIcon = hasInlineIconUrl(stage2Response)
        val stage3HasInlineImage = genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if ((missingInlineImage || missingInlineIcon) && !usedFallback) {
            warnings += "Media content was adjusted for compatibility."
            stage3Json = gson.toJson(buildFallbackGenUi(stage2Response))
            usedFallback = true
        }
        if (responseContainsActionButtons(stage2Response) && !genUiPreservesActionButtons(stage3Json) && !usedFallback) {
            warnings += "Quick actions were adjusted for compatibility."
            stage3Json = gson.toJson(buildFallbackGenUi(stage2Response))
            usedFallback = true
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        var renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

        if (renderResult.errorMessage != null && !usedFallback) {
            warnings += "Native rendering failed for stage 3 output; using fallback UI."
            val fallback = buildFallbackGenUi(stage2Response)
            stage3Json = gson.toJson(fallback)
            renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)
            usedFallback = true
        }

        if (renderResult.errorMessage != null) {
            markDuration(Stage.STAGE4, stage4StartedAtMs)
            return@withContext Outcome.Failure(
                stage = Stage.STAGE4,
                message = renderResult.errorMessage,
                stage2Response = stage2Response,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        markDuration(Stage.STAGE4, stage4StartedAtMs)

        return@withContext Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = stage2Prompt,
                stage2Response = stage2Response,
                stage3Prompt = stage3Prompt,
                stage3SystemPrompt = promptContext.systemPrompt,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap(),
                usedFallback = usedFallback,
                warnings = warnings,
                renderResult = renderResult
            )
        )
    }

    suspend fun executeStage3FromResponse(
        queryText: String,
        stage2ResponseText: String,
        onStageUpdate: (StageUpdate) -> Unit
    ): Outcome = withContext(Dispatchers.IO) {
        val stageDurationsMs = linkedMapOf<Stage, Long>()
        val stageStreamDurationsMs = linkedMapOf<Stage, Long>()
        fun markDuration(stage: Stage, startMs: Long) {
            if (startMs <= 0L) return
            stageDurationsMs[stage] = (System.currentTimeMillis() - startMs).coerceAtLeast(0L)
        }
        fun markStreamDuration(stage: Stage, durationMs: Long?) {
            val duration = durationMs ?: return
            stageStreamDurationsMs[stage] = (stageStreamDurationsMs[stage] ?: 0L) + duration.coerceAtLeast(0L)
        }

        val normalizedQuery = queryText.trim()
            .takeIf { it.isNotBlank() }
            ?: "IR demo query"
        val normalizedResponseRaw = stage2ResponseText.trim()
        if (normalizedResponseRaw.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Response text is empty.",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        val provider = InferenceBackendSettings.getProvider(appContext)
        val selectedModel = GeminiModelSettings.getSelectedModel(appContext)
        val localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(appContext)
        val localModelPath = InferenceBackendSettings.getLocalModelPath(appContext)
        val isLocalServer = provider == InferenceBackendSettings.Provider.LOCAL_SERVER
        val stage3MaxOutputTokens = if (isLocalServer) LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS else STAGE3_MAX_OUTPUT_TOKENS
        val stage3RepairMaxOutputTokens = stage3MaxOutputTokens

        val apiKey = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            BuildConfig.GEMINI_API_KEY.trim()
        } else {
            ""
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI && apiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Gemini API key is missing. Set GEMINI_API_KEY before building the app.",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER && localServerBaseUrl.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Local server URL is missing. Open Settings and configure Local Server.",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            postUpdate(onStageUpdate, Stage.STAGE3, "Checking local server connectivity")
            val health = checkLocalServerHealth(localServerBaseUrl)
            if (health != null) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = health,
                    stageDurationsMs = stageDurationsMs.toMap(),
                    stageStreamDurationsMs = stageStreamDurationsMs.toMap()
                )
            }
        }

        val genUiTemplate = runCatching { loadPromptAsset(STAGE3_PROMPT_ASSET) }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}",
                    stageDurationsMs = stageDurationsMs.toMap(),
                    stageStreamDurationsMs = stageStreamDurationsMs.toMap()
                )
            }
        val promptContext = prepareStage3PromptContext(genUiTemplate)
        val stage3CacheDeferred = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            async(Dispatchers.IO) {
                ensureStage3InstructionCache(
                    apiKey = apiKey,
                    model = selectedModel,
                    systemPrompt = promptContext.systemPrompt
                )
            }
        } else {
            null
        }

        val stage2WithActions = ensureFlightQuickActions(
            responseText = normalizedResponseRaw,
            queryText = normalizedQuery
        )
        val stage2WithTravelMedia = ensureTravelInlineMedia(
            responseText = stage2WithActions,
            queryText = normalizedQuery
        )
        val stage2Response = normalizeUrlTokensForDisplay(stage2WithTravelMedia)
        val normalizedBareDomains = stage2Response != stage2WithTravelMedia
        val injectedTravelMedia = stage2WithTravelMedia != stage2WithActions

        val stage3Prompt = buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = stage2Response,
            assets = emptyList()
        )
        val localStage3SystemPromptCacheKey = if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            buildLocalSystemPromptCacheKey(
                systemPrompt = promptContext.systemPrompt
            )
        } else {
            null
        }
        val warnings = mutableListOf<String>()
        warnings += "Using preloaded IR demo response (stage 2 skipped)."
        if (provider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Using Gemini model: $selectedModel"
        } else {
            warnings += "Using local server: $localServerBaseUrl"
            warnings += "Local model path: $localModelPath"
            warnings += "Local token cap: stage3=$stage3MaxOutputTokens"
            if (!localStage3SystemPromptCacheKey.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache key: ${localStage3SystemPromptCacheKey.take(16)}..."
            }
        }
        if (normalizedBareDomains) {
            warnings += "Normalized bare source/action domains to https URLs."
        }
        if (injectedTravelMedia) {
            warnings += "Added fallback inline media URLs for travel content."
        }
        val stage3Cache = if (stage3CacheDeferred != null) {
            runCatching { stage3CacheDeferred.await() }
                .getOrElse {
                    CacheSetupResult(
                        name = null,
                        created = false,
                        error = it.message ?: it.javaClass.simpleName
                    )
                }
        } else {
            CacheSetupResult(name = null, created = false, error = null)
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI) {
            if (stage3Cache.name != null && !stage3Cache.created) {
                warnings += "Stage 3 instruction cache hit."
            }
            if (stage3Cache.created) {
                warnings += "Stage 3 instruction cache created."
            }
            stage3Cache.error?.let {
                warnings += "Stage 3 instruction cache unavailable ($it). Using direct prompt."
            }
        }
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            val primeError = ensureLocalSystemPromptCache(
                localServerBaseUrl = localServerBaseUrl,
                localModelPath = localModelPath,
                cacheKey = localStage3SystemPromptCacheKey,
                systemPrompt = promptContext.systemPrompt
            )
            if (primeError == null && !localStage3SystemPromptCacheKey.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache primed."
            } else if (!primeError.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache prime failed ($primeError)."
            }
        }
        val localSendStage3SystemPrompt = shouldSendLocalSystemPrompt(localStage3SystemPromptCacheKey)

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            provider = provider,
            apiKey = apiKey,
            model = selectedModel,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath,
            prompt = stage3Prompt,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            temperature = 0.2,
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            structuredOutput = provider == InferenceBackendSettings.Provider.GEMINI,
            localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
            localSendSystemPrompt = localSendStage3SystemPrompt
        )
        markStreamDuration(Stage.STAGE3, stage3Call.streamDurationMs)

        if (stage3Call.error != null) {
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = stage3Call.error,
                stage2Response = stage2Response,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        var stage3JsonElement = extractJsonElement(stage3Call.text)
        var usedFallback = false

        if (stage3JsonElement == null) {
            warnings += "Stage 3 JSON parse failed; running repair pass."
            val repairCall = generateWithRetry(
                provider = provider,
                apiKey = apiKey,
                model = selectedModel,
                localServerBaseUrl = localServerBaseUrl,
                localModelPath = localModelPath,
                prompt = buildRepairPrompt(stage3Call.text),
                systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
                temperature = 0.2,
                maxOutputTokens = stage3RepairMaxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = stage3Cache.name,
                allowCachedContent = true,
                structuredOutput = provider == InferenceBackendSettings.Provider.GEMINI,
                localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
                localSendSystemPrompt = localSendStage3SystemPrompt
            )
            markStreamDuration(Stage.STAGE3, repairCall.streamDurationMs)
            if (repairCall.error == null) {
                stage3JsonElement = extractJsonElement(repairCall.text)
            }
        }

        if (stage3JsonElement == null) {
            warnings += "Stage 3 fallback JSON was used."
            stage3JsonElement = buildFallbackGenUi(stage2Response)
            usedFallback = true
        }

        val normalizedGenUi = normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        val stage2HasInlineImage = hasInlineImageUrl(stage2Response)
        val stage2HasInlineIcon = hasInlineIconUrl(stage2Response)
        val stage3HasInlineImage = genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if ((missingInlineImage || missingInlineIcon) && !usedFallback) {
            warnings += "Media content was adjusted for compatibility."
            stage3Json = gson.toJson(buildFallbackGenUi(stage2Response))
            usedFallback = true
        }
        if (responseContainsActionButtons(stage2Response) && !genUiPreservesActionButtons(stage3Json) && !usedFallback) {
            warnings += "Quick actions were adjusted for compatibility."
            stage3Json = gson.toJson(buildFallbackGenUi(stage2Response))
            usedFallback = true
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        var renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

        if (renderResult.errorMessage != null && !usedFallback) {
            warnings += "Native rendering failed for stage 3 output; using fallback UI."
            val fallback = buildFallbackGenUi(stage2Response)
            stage3Json = gson.toJson(fallback)
            renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)
            usedFallback = true
        }

        if (renderResult.errorMessage != null) {
            markDuration(Stage.STAGE4, stage4StartedAtMs)
            return@withContext Outcome.Failure(
                stage = Stage.STAGE4,
                message = renderResult.errorMessage,
                stage2Response = stage2Response,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        markDuration(Stage.STAGE4, stage4StartedAtMs)

        return@withContext Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = "IR demo preloaded response (stage 2 skipped).",
                stage2Response = stage2Response,
                stage3Prompt = stage3Prompt,
                stage3SystemPrompt = promptContext.systemPrompt,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap(),
                usedFallback = usedFallback,
                warnings = warnings,
                renderResult = renderResult
            )
        )
    }

    private suspend fun postUpdate(
        callback: (StageUpdate) -> Unit,
        stage: Stage,
        message: String
    ) {
        withContext(Dispatchers.Main) {
            callback(StageUpdate(stage = stage, message = message))
        }
    }

    private fun buildRepairPrompt(rawText: String): String {
        return (
            "The previous output was not valid JSON or failed schema validation. " +
                "Fix the output to be valid JSON that satisfies the schema requirements. " +
                "Return ONLY the corrected JSON.\n\nOriginal:\n${rawText.trim()}"
            )
    }

    private fun ensureFlightQuickActions(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeFlightQuery(queryText) && !looksLikeFlightContent(responseText)) {
            return responseText
        }

        val hasActionWithUrl = Regex(
            """(?im)^\s*Action:\s*\[Button:\s*.+?\]\s*(?:https?://|//|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,24})\S*"""
        )
            .containsMatchIn(responseText)
        val hasQuickActionsWithUrl = Regex(
            """(?is)Quick\s*Actions.*(?:https?://|//|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,24})"""
        )
            .containsMatchIn(responseText)
        if (hasActionWithUrl || hasQuickActionsWithUrl) {
            return responseText
        }

        val sourceUrls = extractUrlsForQuickActions(responseText)
        val actionLines = if (sourceUrls.isNotEmpty()) {
            sourceUrls.take(3).mapIndexed { index, url ->
                "Action: [Button: ${quickActionLabelForUrl(url, index)}] $url"
            }
        } else {
            listOf(
                "Action: [Button: Search on MakeMyTrip] https://www.makemytrip.com/flights/",
                "Action: [Button: Search on Skyscanner] https://www.skyscanner.co.in/",
                "Action: [Button: Search on Goibibo] https://www.goibibo.com/flights/"
            )
        }

        return buildString {
            append(responseText.trimEnd())
            append("\n\nQuick Actions\n")
            append(actionLines.joinToString(separator = "\n"))
        }
    }

    private fun ensureTravelInlineMedia(
        responseText: String,
        queryText: String
    ): String {
        if (!looksLikeTravelQuery(queryText) && !looksLikeTravelContent(responseText)) {
            return responseText
        }
        if (hasTravelMediaCoverage(responseText)) {
            return responseText
        }

        val normalized = responseText.replace("\r\n", "\n").trim()
        if (normalized.isBlank()) {
            return responseText
        }

        val lines = normalized.split('\n')
        val output = mutableListOf<String>()
        val locationKeyword = extractTravelLocationKeyword(queryText)
        var inserted = 0
        val maxInsertions = 4

        lines.forEachIndexed { index, rawLine ->
            val line = rawLine.trimEnd()
            output += rawLine

            if (inserted >= maxInsertions) {
                return@forEachIndexed
            }

            val trimmed = line.trim()
            if (!shouldAttachTravelMediaAfterLine(trimmed)) {
                return@forEachIndexed
            }
            if (hasNearbyMediaLine(lines, index)) {
                return@forEachIndexed
            }

            val mediaLine = buildTravelMediaLine(
                line = trimmed,
                locationKeyword = locationKeyword,
                baseText = normalized
            )
            output += mediaLine
            inserted += 1
        }

        if (inserted == 0) {
            val mediaLine = buildTravelMediaLine(
                line = queryText,
                locationKeyword = locationKeyword,
                baseText = normalized
            )
            return buildString {
                append(normalized)
                append("\n\n")
                append(mediaLine)
            }
        }

        return output.joinToString(separator = "\n").trimEnd()
    }

    private fun looksLikeTravelQuery(queryText: String): Boolean {
        val normalized = queryText.lowercase(Locale.US)
        return normalized.contains("travel") ||
            normalized.contains("trip") ||
            normalized.contains("itinerary") ||
            normalized.contains("places to visit") ||
            normalized.contains("things to do") ||
            normalized.contains("attractions") ||
            normalized.contains("visit")
    }

    private fun looksLikeTravelContent(text: String): Boolean {
        val normalized = text.lowercase(Locale.US)
        return normalized.contains("day 1") ||
            normalized.contains("itinerary") ||
            normalized.contains("places to visit") ||
            normalized.contains("things to do") ||
            normalized.contains("attraction") ||
            normalized.contains("must-visit")
    }

    private fun hasNearbyMediaLine(lines: List<String>, index: Int): Boolean {
        val start = maxOf(0, index)
        val end = minOf(lines.lastIndex, index + 2)
        for (cursor in start..end) {
            val candidate = lines[cursor].trim()
            if (candidate.isBlank()) {
                continue
            }
            if (candidate.startsWith("Media:", ignoreCase = true)) {
                return true
            }
            if (candidate.contains("Image=", ignoreCase = true) || candidate.contains("Icon=", ignoreCase = true)) {
                return true
            }
        }
        return false
    }

    private fun shouldAttachTravelMediaAfterLine(line: String): Boolean {
        if (line.isBlank()) {
            return false
        }
        val normalized = line
            .trim()
            .replace(Regex("""^#+\s*"""), "")
        if (containsUrlLikeToken(normalized)) {
            return false
        }
        if (normalized.startsWith("Media:", ignoreCase = true) ||
            normalized.startsWith("Action:", ignoreCase = true) ||
            normalized.startsWith("Source", ignoreCase = true) ||
            normalized.startsWith("Sources", ignoreCase = true) ||
            normalized.startsWith("Quick Actions", ignoreCase = true)
        ) {
            return false
        }
        if (Regex("""(?i)^(option\s*\d+|day\s*\d+|place\s*\d+|stop\s*\d+|attraction\s*\d+)\s*[:\-]""").containsMatchIn(normalized)) {
            return true
        }
        if (normalized.contains('|')) {
            return false
        }
        if (normalized.startsWith("-") || normalized.startsWith("\u2022")) {
            return false
        }

        val wordCount = normalized.split(Regex("""\s+""")).count { it.isNotBlank() }
        if (wordCount in 2..12 && normalized.length <= 96 && !normalized.endsWith(".")) {
            return true
        }
        return Regex("""(?i)^(day\s*\d+|place\s*\d+|stop\s*\d+|attraction\s*\d+)\b""")
            .containsMatchIn(normalized)
    }

    private fun containsUrlLikeToken(value: String): Boolean {
        val normalized = value.lowercase(Locale.US)
        return normalized.contains("http://") ||
            normalized.contains("https://") ||
            Regex("""(?i)\b(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b""")
                .containsMatchIn(value)
    }

    private fun buildTravelMediaLine(
        line: String,
        locationKeyword: String,
        baseText: String
    ): String {
        val imageKeyword = buildTravelImageKeyword(line, locationKeyword)
        val iconName = pickTravelIconName(line)
        val hasImage = hasInlineImageUrl(baseText)
        val hasIcon = hasInlineIconUrl(baseText)
        return when {
            !hasImage && !hasIcon ->
                "Media: Image=https://loremflickr.com/1200/800/$imageKeyword Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
            !hasImage ->
                "Media: Image=https://loremflickr.com/1200/800/$imageKeyword"
            !hasIcon ->
                "Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
            else ->
                "Media: Image=https://loremflickr.com/1200/800/$imageKeyword Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/$iconName.svg"
        }
    }

    private fun buildTravelImageKeyword(line: String, locationKeyword: String): String {
        val stopwords = setOf(
            "the", "and", "for", "with", "from", "into", "your", "this", "that", "day",
            "place", "visit", "best", "top", "must", "to", "in", "of", "at", "on", "a", "an"
        )
        val words = line.lowercase(Locale.US)
            .replace(Regex("""[^a-z0-9\s-]"""), " ")
            .split(Regex("""\s+"""))
            .filter { it.length >= 3 && it !in stopwords }
            .take(3)
        val merged = buildList {
            add(locationKeyword)
            addAll(words)
        }
            .distinct()
            .joinToString(",")
            .ifBlank { "$locationKeyword,travel" }
        return URLEncoder.encode(merged, StandardCharsets.UTF_8.name())
            .replace("+", "%20")
    }

    private fun pickTravelIconName(line: String): String {
        val normalized = line.lowercase(Locale.US)
        return when {
            normalized.contains("beach") || normalized.contains("island") || normalized.contains("sea") || normalized.contains("bay") -> "water"
            normalized.contains("temple") || normalized.contains("shrine") || normalized.contains("buddha") || normalized.contains("old town") -> "building"
            normalized.contains("market") || normalized.contains("food") || normalized.contains("street") -> "shop"
            normalized.contains("night") || normalized.contains("sunset") -> "moon-stars"
            normalized.contains("view") || normalized.contains("hike") || normalized.contains("trail") -> "signpost-split"
            normalized.contains("boat") || normalized.contains("pier") -> "geo-alt"
            else -> "geo-alt"
        }
    }

    private fun hasTravelMediaCoverage(text: String): Boolean =
        hasInlineImageUrl(text) && hasInlineIconUrl(text)

    private fun hasInlineImageUrl(text: String): Boolean {
        val imageAssignment = Regex(
            """(?im)\bImage\s*=\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        if (imageAssignment) {
            return true
        }
        val imageColon = Regex(
            """(?im)^\s*Image\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        return imageColon
    }

    private fun hasInlineIconUrl(text: String): Boolean {
        val iconAssignment = Regex(
            """(?im)\bIcon\s*=\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        if (iconAssignment) {
            return true
        }
        val iconColon = Regex(
            """(?im)^\s*Icon\s*:\s*(https?://\S+|/assets/\S+|assets/\S+)"""
        )
            .findAll(text)
            .map { sanitizeMediaUrlToken(it.groupValues[1]) }
            .any { looksLikeUsableInlineMediaUrl(it) }
        return iconColon
    }

    private fun extractTravelLocationKeyword(queryText: String): String {
        val prepositionMatch = Regex("""(?i)\b(?:in|at|to|from|for)\s+([a-z][a-z0-9-]{2,30})\b""")
            .findAll(queryText)
            .lastOrNull()
            ?.groupValues
            ?.getOrNull(1)
            ?.lowercase(Locale.US)
        if (!prepositionMatch.isNullOrBlank()) {
            return prepositionMatch
        }

        val blocked = setOf(
            "show", "best", "top", "places", "visit", "travel", "trip", "itinerary",
            "things", "todo", "to", "in", "for", "with"
        )
        val token = queryText.lowercase(Locale.US)
            .split(Regex("""[^a-z0-9-]+"""))
            .firstOrNull { it.length >= 3 && it !in blocked }
        return token ?: "travel"
    }

    private fun looksLikeFlightQuery(queryText: String): Boolean {
        val normalized = queryText.lowercase(Locale.US)
        return normalized.contains("flight") ||
            normalized.contains("airline") ||
            normalized.contains("departure") ||
            normalized.contains("arrival")
    }

    private fun looksLikeFlightContent(text: String): Boolean {
        val normalized = text.lowercase(Locale.US)
        return normalized.contains("airline") &&
            (normalized.contains("departure") || normalized.contains("arrival")) &&
            normalized.contains("fare")
    }

    private fun extractUrlsForQuickActions(text: String): List<String> {
        return URL_TOKEN_REGEX.findAll(text)
            .mapNotNull { normalizeExternalUrlCandidate(it.value) }
            .distinct()
            .toList()
    }

    private fun normalizeUrlTokensForDisplay(text: String): String {
        if (text.isBlank()) {
            return text
        }
        return URL_TOKEN_REGEX.replace(text) { match ->
            normalizeExternalUrlCandidate(match.value) ?: match.value
        }
    }

    private fun normalizeExternalUrlCandidate(value: String): String? {
        val token = value.trim().trim('"', '\'').trimEnd('.', ',', ';', ')', ']', '}')
        if (token.isBlank()) {
            return null
        }
        if (token.startsWith("http://", ignoreCase = true) || token.startsWith("https://", ignoreCase = true)) {
            return token
        }
        if (token.startsWith("//")) {
            return "https:$token"
        }
        if (!token.startsWith("www.", ignoreCase = true) &&
            !Regex("""(?i)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#].*)?""").matches(token)
        ) {
            return null
        }

        val host = token
            .removePrefix("www.")
            .substringBefore('/')
            .substringBefore('?')
            .substringBefore('#')
            .lowercase(Locale.US)
        if (!isLikelyPublicDomainHost(host)) {
            return null
        }
        return "https://$token"
    }

    private fun isLikelyPublicDomainHost(host: String): Boolean {
        if (host.isBlank() || host.contains('_')) {
            return false
        }
        val labels = host.split('.').filter { it.isNotBlank() }
        if (labels.size < 2 || labels.any { !HOST_LABEL_REGEX.matches(it) }) {
            return false
        }
        val tld = labels.last().lowercase(Locale.US)
        if (!tld.all { it in 'a'..'z' } || tld.length !in 2..24) {
            return false
        }
        if (
            tld in setOf(
                "png", "jpg", "jpeg", "svg", "webp", "gif", "bmp", "ico",
                "json", "xml", "txt", "csv", "md", "pdf", "zip", "apk"
            )
        ) {
            return false
        }
        return true
    }

    private fun quickActionLabelForUrl(url: String, index: Int): String {
        val host = runCatching { URL(url).host.lowercase(Locale.US) }.getOrDefault("")
        return when {
            host.contains("makemytrip") -> "Search on MakeMyTrip"
            host.contains("skyscanner") -> "Search on Skyscanner"
            host.contains("goibibo") -> "Search on Goibibo"
            host.contains("goindigo") -> "Open IndiGo"
            host.contains("airindia") -> "Open Air India"
            host.contains("akasaair") -> "Open Akasa Air"
            host.contains("flightsfrom") -> "Open FlightsFrom"
            else -> "Open Source ${index + 1}"
        }
    }

    private fun responseContainsInlineMedia(text: String): Boolean {
        val assignmentRegex = Regex(
            """(?im)\b(?:Media:\s*)?(?:Image|Icon)\s*=\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)"""
        )
        val colonRegex = Regex(
            """(?im)^\s*(?:Image|Icon)\s*:\s*(https?://\S+|/assets/\S+|assets/\S+|\S+)"""
        )
        val candidates = mutableListOf<String>()
        assignmentRegex.findAll(text).forEach { match ->
            candidates += sanitizeMediaUrlToken(match.groupValues[1])
        }
        colonRegex.findAll(text).forEach { match ->
            candidates += sanitizeMediaUrlToken(match.groupValues[1])
        }
        return candidates.any(::looksLikeUsableInlineMediaUrl)
    }

    private fun genUiPreservesInlineMedia(jsonText: String): Boolean {
        return genUiPreservesInlineImages(jsonText) || genUiPreservesInlineIcons(jsonText)
    }

    private fun genUiPreservesInlineImages(jsonText: String): Boolean {
        return Regex(
            """"component"\s*:\s*"Image"[\s\S]{0,320}"(?:url|src|source|image)"\s*:\s*"(?:https?://|/assets/|assets/)"""",
            setOf(RegexOption.IGNORE_CASE)
        ).containsMatchIn(jsonText) ||
            Regex(
                """(?i)Media:\s*Image=|(?:^|\\n)Image:\s*(?:https?://|/assets/|assets/)"""
            ).containsMatchIn(jsonText)
    }

    private fun genUiPreservesInlineIcons(jsonText: String): Boolean {
        return Regex(
            """"component"\s*:\s*"Icon"[\s\S]{0,240}"(?:url|icon|name|glyph|asset)"\s*:\s*"[^"]+"""",
            setOf(RegexOption.IGNORE_CASE)
        ).containsMatchIn(jsonText) ||
            Regex(
                """(?i)Media:\s*Icon=|(?:^|\\n)Icon:\s*(?:https?://|/assets/|assets/)"""
            ).containsMatchIn(jsonText)
    }

    private fun responseContainsActionButtons(text: String): Boolean {
        return Regex(
            """(?im)^\s*Action:\s*\[Button:\s*.+?\]\s*(?:https?://|//|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,24})\S*"""
        )
            .containsMatchIn(text)
    }

    private fun genUiPreservesActionButtons(jsonText: String): Boolean {
        return Regex("""(?i)"call"\s*:\s*"openUrl"""").containsMatchIn(jsonText) ||
            (
                Regex("""(?i)"component"\s*:\s*"Button"""").containsMatchIn(jsonText) &&
                    Regex("""(?i)"url"\s*:\s*"https?://""").containsMatchIn(jsonText)
                )
    }

    private fun sanitizeMediaUrlToken(value: String): String =
        value.trim().trim('\'', '"').trimEnd('.', ',', ';', ')', ']')

    private fun looksLikeUsableInlineMediaUrl(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isBlank()) {
            return false
        }
        val lower = normalized.lowercase(Locale.US)
        if (
            lower in setOf(
                "<image_url>",
                "<icon_url>",
                "<url>",
                "image_url",
                "icon_url",
                "url",
                "n/a",
                "na",
                "none",
                "null",
                "--"
            ) ||
            lower.contains("placeholder") ||
            lower.contains("<") ||
            lower.contains(">")
        ) {
            return false
        }
        if (lower.startsWith("/assets/") || lower.startsWith("assets/")) {
            return true
        }

        val pathWithoutQuery = lower.substringBefore('?').substringBefore('#')
        if (
            pathWithoutQuery.endsWith(".png") ||
            pathWithoutQuery.endsWith(".jpg") ||
            pathWithoutQuery.endsWith(".jpeg") ||
            pathWithoutQuery.endsWith(".svg") ||
            pathWithoutQuery.endsWith(".webp")
        ) {
            return true
        }

        val uri = runCatching { URI(normalized) }.getOrNull() ?: return false
        val scheme = uri.scheme?.lowercase(Locale.US) ?: return false
        if (scheme != "http" && scheme != "https") {
            return false
        }
        val host = uri.host?.lowercase(Locale.US).orEmpty()
        val path = uri.path?.lowercase(Locale.US).orEmpty()
        if (
            host.contains("cdn.jsdelivr.net") ||
            host.contains("raw.githubusercontent.com") ||
            host.contains("upload.wikimedia.org") ||
            host.contains("imgur.com") ||
            host.contains("gstatic.com") ||
            host.contains("twimg.com") ||
            host.contains("loremflickr.com") ||
            host.contains("picsum.photos")
        ) {
            return true
        }
        return path.contains("/icon") || path.contains("/icons/") || path.contains("/image") || path.contains("/images/")
    }

    private fun normalizeGenUiPayload(json: JsonElement): JsonElement {
        if (json.isJsonArray) {
            return json
        }
        if (!json.isJsonObject) {
            return json
        }

        val obj = json.asJsonObject
        val directArray = obj.get("genui_json")
        if (directArray != null && directArray.isJsonArray) {
            return directArray
        }
        val messages = obj.get("messages")
        if (messages != null && messages.isJsonArray) {
            return messages
        }
        val payload = obj.get("payload")
        if (payload != null) {
            if (payload.isJsonArray) {
                return payload
            }
            if (payload.isJsonObject) {
                val payloadObj = payload.asJsonObject
                val payloadMessages = payloadObj.get("messages")
                if (payloadMessages != null && payloadMessages.isJsonArray) {
                    return payloadMessages
                }
            }
        }
        return json
    }

    private fun buildFallbackGenUi(stage2Response: String): JsonArray {
        val textValue = stage2Response.trim().ifBlank { "No content generated." }
        val surfaceId = "surface_live"
        return JsonArray().apply {
            add(
                JsonObject().apply {
                    addProperty("version", "v0.9")
                    add("createSurface", JsonObject().apply {
                        addProperty("surfaceId", surfaceId)
                        addProperty("catalogId", "https://genui.local/specification/v0_9/standard_catalog.json")
                    })
                }
            )
            add(
                JsonObject().apply {
                    addProperty("version", "v0.9")
                    add("updateComponents", JsonObject().apply {
                        addProperty("surfaceId", surfaceId)
                        add("components", JsonArray().apply {
                            add(JsonObject().apply {
                                addProperty("id", "root")
                                addProperty("component", "Column")
                                add("children", JsonArray().apply {
                                    add("text_1")
                                })
                            })
                            add(JsonObject().apply {
                                addProperty("id", "text_1")
                                addProperty("component", "Text")
                                addProperty("variant", "body")
                                addProperty("text", textValue)
                            })
                        })
                    })
                }
            )
        }
    }

    private fun prepareStage3PromptContext(template: String): Stage3PromptContext {
        val placeholder = "{response_text}"
        if (!template.contains(placeholder)) {
            return Stage3PromptContext(
                systemPrompt = template.trim(),
                userTemplate =
                    "Convert the response text into valid GenUICraft JSON.\n" +
                        "Return ONLY the JSON message array.\n\n" +
                        "Response:\n{response_text}"
            )
        }

        val split = template.split(placeholder, limit = 2)
        val systemPrompt = "${split[0]}[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]${split[1]}".trim()
        val userTemplate = (
            "Convert the response text into valid GenUICraft JSON.\n" +
                "Return ONLY the JSON message array.\n\n" +
                "Response:\n{response_text}"
            )
        return Stage3PromptContext(systemPrompt = systemPrompt, userTemplate = userTemplate)
    }

    private fun buildStage3UserPrompt(
        userTemplate: String,
        stage2Response: String,
        assets: List<AssetMapping>
    ): String {
        val assetPolicy = if (assets.isEmpty()) {
            "Asset URL policy for this request:\n" +
                "- No local asset mapping is provided.\n" +
                "- Preserve media URLs from the response exactly as written.\n" +
                "- Do not invent local placeholder paths such as /image.jpg or /asset/foo.png."
        } else {
            "Asset URL policy for this request:\n" +
                "- Use only local media paths from the provided Assets mapping.\n" +
                "- Do not emit remote media URLs for images/icons.\n" +
                "- Do not invent local placeholder paths not present in the mapping."
        }

        val assetContext = if (assets.isEmpty()) {
            ""
        } else {
            val rows = assets.joinToString("\n") { "- ${it.url} -> ${it.localPath}" }
            "Assets (local copies of any URLs in the response; use ONLY these local paths):\n$rows"
        }

        val responseWithPolicy = "${stage2Response.trim()}\n\n$assetPolicy"
        val responseText = if (assetContext.isBlank()) {
            responseWithPolicy
        } else {
            "$responseWithPolicy\n\n$assetContext"
        }

        return renderPrompt(userTemplate, "response_text" to responseText)
    }

    private fun renderPrompt(template: String, vararg args: Pair<String, String>): String {
        var rendered = template
        args.forEach { (key, value) ->
            rendered = rendered.replace("{$key}", value)
        }
        return rendered
    }

    private fun loadPromptAsset(path: String): String {
        return appContext.assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    private fun extractJsonElement(text: String): JsonElement? {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) {
            return null
        }

        extractFencedBlock(cleaned)?.let { fenced ->
            try {
                return JsonParser.parseString(fenced)
            } catch (_: Exception) {
            }
        }

        if (cleaned.startsWith("[") || cleaned.startsWith("{")) {
            try {
                return JsonParser.parseString(cleaned)
            } catch (_: Exception) {
            }
        }

        findFirstBalanced(cleaned, '[', ']')?.let { arrayCandidate ->
            try {
                return JsonParser.parseString(arrayCandidate)
            } catch (_: Exception) {
            }
        }

        findFirstBalanced(cleaned, '{', '}')?.let { objectCandidate ->
            try {
                return JsonParser.parseString(objectCandidate)
            } catch (_: Exception) {
            }
        }

        return null
    }

    private fun extractFencedBlock(text: String): String? {
        val start = text.indexOf("```")
        if (start < 0) {
            return null
        }
        val end = text.indexOf("```", startIndex = start + 3)
        if (end <= start) {
            return null
        }
        val block = text.substring(start + 3, end).trim()
        return if (block.startsWith("json", ignoreCase = true)) {
            block.removePrefix("json").trim()
        } else {
            block
        }
    }

    private fun findFirstBalanced(text: String, open: Char, close: Char): String? {
        var index = text.indexOf(open)
        while (index >= 0) {
            val candidate = balancedSubstring(text, index, open, close)
            if (candidate != null) {
                return candidate
            }
            index = text.indexOf(open, startIndex = index + 1)
        }
        return null
    }

    private fun balancedSubstring(text: String, start: Int, open: Char, close: Char): String? {
        var depth = 0
        var inString = false
        var escape = false

        for (i in start until text.length) {
            val ch = text[i]
            if (escape) {
                escape = false
                continue
            }
            if (ch == '\\' && inString) {
                escape = true
                continue
            }
            if (ch == '"') {
                inString = !inString
                continue
            }
            if (inString) {
                continue
            }
            if (ch == open) {
                depth += 1
            } else if (ch == close) {
                depth -= 1
                if (depth == 0) {
                    return text.substring(start, i + 1)
                }
            }
        }
        return null
    }

    private fun generateWithRetry(
        provider: InferenceBackendSettings.Provider,
        apiKey: String,
        model: String,
        localServerBaseUrl: String,
        localModelPath: String,
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean = false,
        cachedContentName: String? = null,
        allowCachedContent: Boolean = true,
        structuredOutput: Boolean = false,
        localSystemPromptCacheKey: String? = null,
        localSendSystemPrompt: Boolean = true
    ): GeminiResponse {
        var attempt = 0
        var accumulatedStreamMs = 0L
        var hasStreamSample = false
        var last: GeminiResponse = GeminiResponse(
            text = "",
            rawResponse = null,
            error = "Unknown generation error",
            streamDurationMs = null
        )
        val effectiveCachedContentName = if (
            allowCachedContent && provider == InferenceBackendSettings.Provider.GEMINI
        ) {
            cachedContentName
        } else {
            null
        }
        while (attempt < 3) {
            attempt += 1
            last = generateOnce(
                provider = provider,
                apiKey = apiKey,
                model = model,
                localServerBaseUrl = localServerBaseUrl,
                localModelPath = localModelPath,
                prompt = prompt,
                systemPrompt = systemPrompt,
                temperature = temperature,
                maxOutputTokens = maxOutputTokens,
                jsonMode = jsonMode,
                enableGoogleSearch = enableGoogleSearch,
                cachedContentName = effectiveCachedContentName,
                structuredOutput = structuredOutput,
                localSystemPromptCacheKey = localSystemPromptCacheKey,
                localSendSystemPrompt = localSendSystemPrompt
            )
            last.streamDurationMs?.let {
                accumulatedStreamMs += it
                hasStreamSample = true
            }
            if (last.error == null) {
                if (
                    provider == InferenceBackendSettings.Provider.LOCAL_SERVER &&
                    localSendSystemPrompt &&
                    !localSystemPromptCacheKey.isNullOrBlank()
                ) {
                    markLocalSystemPromptCacheKeyReady(localSystemPromptCacheKey)
                }
                return last.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            if (
                provider == InferenceBackendSettings.Provider.GEMINI &&
                enableGoogleSearch &&
                isSearchToolConfigError(last.error)
            ) {
                val fallback = generateOnce(
                    provider = provider,
                    apiKey = apiKey,
                    model = model,
                    localServerBaseUrl = localServerBaseUrl,
                    localModelPath = localModelPath,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    temperature = temperature,
                    maxOutputTokens = maxOutputTokens,
                    jsonMode = jsonMode,
                    enableGoogleSearch = false,
                    cachedContentName = effectiveCachedContentName,
                    structuredOutput = structuredOutput,
                    localSystemPromptCacheKey = localSystemPromptCacheKey,
                    localSendSystemPrompt = localSendSystemPrompt
                )
                fallback.streamDurationMs?.let {
                    accumulatedStreamMs += it
                    hasStreamSample = true
                }
                return fallback.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            if (
                provider == InferenceBackendSettings.Provider.GEMINI &&
                structuredOutput &&
                isStructuredOutputConfigError(last.error)
            ) {
                val fallback = generateOnce(
                    provider = provider,
                    apiKey = apiKey,
                    model = model,
                    localServerBaseUrl = localServerBaseUrl,
                    localModelPath = localModelPath,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    temperature = temperature,
                    maxOutputTokens = maxOutputTokens,
                    jsonMode = jsonMode,
                    enableGoogleSearch = enableGoogleSearch,
                    cachedContentName = effectiveCachedContentName,
                    structuredOutput = false,
                    localSystemPromptCacheKey = localSystemPromptCacheKey,
                    localSendSystemPrompt = localSendSystemPrompt
                )
                fallback.streamDurationMs?.let {
                    accumulatedStreamMs += it
                    hasStreamSample = true
                }
                return fallback.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            if (
                provider == InferenceBackendSettings.Provider.LOCAL_SERVER &&
                !localSendSystemPrompt &&
                !localSystemPromptCacheKey.isNullOrBlank() &&
                isLocalSystemPromptCacheMiss(last.error)
            ) {
                val cacheRecovery = generateOnce(
                    provider = provider,
                    apiKey = apiKey,
                    model = model,
                    localServerBaseUrl = localServerBaseUrl,
                    localModelPath = localModelPath,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    temperature = temperature,
                    maxOutputTokens = maxOutputTokens,
                    jsonMode = jsonMode,
                    enableGoogleSearch = enableGoogleSearch,
                    cachedContentName = effectiveCachedContentName,
                    structuredOutput = structuredOutput,
                    localSystemPromptCacheKey = localSystemPromptCacheKey,
                    localSendSystemPrompt = true
                )
                cacheRecovery.streamDurationMs?.let {
                    accumulatedStreamMs += it
                    hasStreamSample = true
                }
                if (cacheRecovery.error == null) {
                    markLocalSystemPromptCacheKeyReady(localSystemPromptCacheKey)
                }
                return cacheRecovery.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            val lower = last.error.lowercase(Locale.US)
            val retryable = lower.contains("timed out") ||
                lower.contains("timeout") ||
                lower.contains("http 429") ||
                lower.contains("http 503") ||
                lower.contains("candidate text")
            if (!retryable || attempt >= 3) {
                return last.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            Thread.sleep(1000L * attempt)
        }
        return last.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
    }

    private fun generateOnce(
        provider: InferenceBackendSettings.Provider,
        apiKey: String,
        model: String,
        localServerBaseUrl: String,
        localModelPath: String,
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean = false,
        cachedContentName: String? = null,
        structuredOutput: Boolean = false,
        localSystemPromptCacheKey: String? = null,
        localSendSystemPrompt: Boolean = true
    ): GeminiResponse {
        return when (provider) {
            InferenceBackendSettings.Provider.GEMINI -> generateOnceGemini(
                apiKey = apiKey,
                model = model,
                prompt = prompt,
                systemPrompt = systemPrompt,
                temperature = temperature,
                maxOutputTokens = maxOutputTokens,
                jsonMode = jsonMode,
                enableGoogleSearch = enableGoogleSearch,
                cachedContentName = cachedContentName,
                structuredOutput = structuredOutput
            )

            InferenceBackendSettings.Provider.LOCAL_SERVER -> generateOnceLocalServer(
                localServerBaseUrl = localServerBaseUrl,
                localModelPath = localModelPath,
                prompt = prompt,
                systemPrompt = systemPrompt,
                localSystemPromptCacheKey = localSystemPromptCacheKey,
                localSendSystemPrompt = localSendSystemPrompt,
                temperature = temperature,
                maxOutputTokens = maxOutputTokens,
                jsonMode = jsonMode
            )
        }
    }

    private fun generateOnceGemini(
        apiKey: String,
        model: String,
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean = false,
        cachedContentName: String? = null,
        structuredOutput: Boolean = false
    ): GeminiResponse {
        val encodedKey = URLEncoder.encode(apiKey, StandardCharsets.UTF_8.name())
        val endpoint = URL("https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$encodedKey")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 20000
            readTimeout = 180000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = buildRequestPayload(
            prompt = prompt,
            systemPrompt = systemPrompt,
            temperature = temperature,
            maxOutputTokens = maxOutputTokens,
            jsonMode = jsonMode,
            enableGoogleSearch = enableGoogleSearch,
            cachedContentName = cachedContentName,
            structuredOutput = structuredOutput
        )

        return try {
            connection.outputStream.use { out ->
                out.write(body.toByteArray(StandardCharsets.UTF_8))
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val streamRead = readStreamWithTiming(stream)
            val raw = streamRead.text

            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return GeminiResponse(
                    text = "",
                    rawResponse = raw,
                    error = "HTTP $code: ${short.take(320)}",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            val extraction = extractGeminiText(raw)
            if (extraction.text.isNullOrBlank()) {
                val details = extraction.diagnostics?.let { " $it" }.orEmpty()
                return GeminiResponse(
                    text = "",
                    rawResponse = raw,
                    error = "Gemini response did not include candidate text.$details",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            GeminiResponse(
                text = extraction.text,
                rawResponse = raw,
                error = null,
                streamDurationMs = streamRead.streamDurationMs
            )
        } catch (io: IOException) {
            GeminiResponse(
                text = "",
                rawResponse = null,
                error = io.message ?: io.javaClass.simpleName,
                streamDurationMs = null
            )
        } finally {
            connection.disconnect()
        }
    }

    private fun generateOnceLocalServer(
        localServerBaseUrl: String,
        localModelPath: String,
        prompt: String,
        systemPrompt: String?,
        localSystemPromptCacheKey: String?,
        localSendSystemPrompt: Boolean,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean
    ): GeminiResponse {
        val baseUrl = localServerBaseUrl.trim().trimEnd('/')
        if (baseUrl.isBlank()) {
            return GeminiResponse(
                text = "",
                rawResponse = null,
                error = "Local server URL is empty.",
                streamDurationMs = null
            )
        }
        val endpoint = URL("$baseUrl/v1/generate")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 7000
            readTimeout = 300000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = JsonObject().apply {
            addProperty("prompt", prompt)
            if (localSendSystemPrompt && !systemPrompt.isNullOrBlank()) {
                addProperty("system_prompt", systemPrompt)
            }
            if (!localSystemPromptCacheKey.isNullOrBlank()) {
                addProperty("system_prompt_cache_key", localSystemPromptCacheKey)
            }
            addProperty("temperature", temperature)
            addProperty("max_output_tokens", min(maxOutputTokens, 8192))
            addProperty("json_mode", jsonMode)
            if (localModelPath.isNotBlank()) {
                addProperty("model_path", localModelPath)
            }
        }

        return try {
            connection.outputStream.use { out ->
                out.write(gson.toJson(body).toByteArray(StandardCharsets.UTF_8))
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val streamRead = readStreamWithTiming(stream)
            val raw = streamRead.text

            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return GeminiResponse(
                    text = "",
                    rawResponse = raw,
                    error = "HTTP $code: ${short.take(320)}",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            val text = extractLocalServerText(raw)
            if (text.isNullOrBlank()) {
                return GeminiResponse(
                    text = "",
                    rawResponse = raw,
                    error = "Local server response did not include text output.",
                    streamDurationMs = streamRead.streamDurationMs
                )
            }

            GeminiResponse(
                text = text,
                rawResponse = raw,
                error = null,
                streamDurationMs = streamRead.streamDurationMs
            )
        } catch (unknownHost: UnknownHostException) {
            GeminiResponse(
                text = "",
                rawResponse = null,
                error = "Local server host is unreachable ($baseUrl): ${unknownHost.message ?: "unknown host"}",
                streamDurationMs = null
            )
        } catch (io: IOException) {
            GeminiResponse(
                text = "",
                rawResponse = null,
                error = "Local server request failed at $baseUrl: ${io.message ?: io.javaClass.simpleName}",
                streamDurationMs = null
            )
        } finally {
            connection.disconnect()
        }
    }

    private fun extractLocalServerText(raw: String): String? {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull() ?: return null
        val directText = root.get("text")
            ?.takeIf { it.isJsonPrimitive }
            ?.asString
            ?.trim()
        if (!directText.isNullOrBlank()) {
            return directText
        }

        val outputText = root.get("output_text")
            ?.takeIf { it.isJsonPrimitive }
            ?.asString
            ?.trim()
        if (!outputText.isNullOrBlank()) {
            return outputText
        }

        val choices = root.getAsJsonArray("choices")
        if (choices != null && choices.size() > 0) {
            val first = runCatching { choices[0].asJsonObject }.getOrNull()
            val message = first?.getAsJsonObject("message")
            val content = message
                ?.get("content")
                ?.takeIf { it.isJsonPrimitive }
                ?.asString
                ?.trim()
            if (!content.isNullOrBlank()) {
                return content
            }
        }
        return null
    }

    private fun checkLocalServerHealth(localServerBaseUrl: String): String? {
        val baseUrl = localServerBaseUrl.trim().trimEnd('/')
        if (baseUrl.isBlank()) {
            return "Local server URL is empty."
        }
        val endpoint = URL("$baseUrl/health")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 5000
            readTimeout = 7000
        }
        return try {
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                "Local server health check failed: HTTP $code: ${short.take(200)}"
            } else {
                null
            }
        } catch (unknownHost: UnknownHostException) {
            "Local server host is unreachable: ${unknownHost.message ?: "unknown host"}"
        } catch (io: IOException) {
            "Could not connect to local server at $baseUrl (${io.message ?: io.javaClass.simpleName})"
        } finally {
            connection.disconnect()
        }
    }

    private fun readStreamWithTiming(stream: java.io.InputStream?): StreamReadResult {
        if (stream == null) {
            return StreamReadResult(text = "", streamDurationMs = null)
        }
        var firstChunkAtMs: Long? = null
        val builder = StringBuilder()
        stream.bufferedReader(Charsets.UTF_8).use { reader ->
            val buffer = CharArray(4096)
            while (true) {
                val read = reader.read(buffer)
                if (read <= 0) {
                    break
                }
                if (firstChunkAtMs == null) {
                    firstChunkAtMs = System.currentTimeMillis()
                }
                builder.append(buffer, 0, read)
            }
        }
        val streamDurationMs = firstChunkAtMs?.let { start ->
            (System.currentTimeMillis() - start).coerceAtLeast(0L)
        }
        return StreamReadResult(
            text = builder.toString(),
            streamDurationMs = streamDurationMs
        )
    }

    private fun extractGeminiText(raw: String): GeminiTextExtraction {
        val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
            ?: return GeminiTextExtraction(
                text = null,
                diagnostics = "Response was not valid JSON."
            )

        val candidates = root.getAsJsonArray("candidates")
        if (candidates == null || candidates.size() == 0) {
            return GeminiTextExtraction(
                text = null,
                diagnostics = buildGeminiDiagnostics(root, null)
            )
        }

        val first = candidates.firstOrNull()?.asJsonObject
        if (first == null) {
            return GeminiTextExtraction(
                text = null,
                diagnostics = buildGeminiDiagnostics(root, null)
            )
        }

        val content = first.getAsJsonObject("content")
        val parts = content?.getAsJsonArray("parts")
        if (parts == null || parts.size() == 0) {
            return GeminiTextExtraction(
                text = null,
                diagnostics = buildGeminiDiagnostics(root, first)
            )
        }

        val builder = StringBuilder()
        for (part in parts) {
            val partObj = runCatching { part.asJsonObject }.getOrNull() ?: continue
            val textPart = partObj.get("text")?.takeIf { it.isJsonPrimitive }?.asString
            if (!textPart.isNullOrBlank()) {
                if (builder.isNotEmpty()) {
                    builder.append('\n')
                }
                builder.append(textPart)
                continue
            }

            val inlineText = decodeInlineDataText(partObj)
            if (!inlineText.isNullOrBlank()) {
                if (builder.isNotEmpty()) {
                    builder.append('\n')
                }
                builder.append(inlineText)
            }
        }

        val text = builder.toString().trim().ifBlank { null }
        return GeminiTextExtraction(
            text = text,
            diagnostics = if (text == null) buildGeminiDiagnostics(root, first) else null
        )
    }

    private fun decodeInlineDataText(part: JsonObject): String? {
        val inlineData = part.getAsJsonObject("inlineData") ?: return null
        val mimeType = inlineData.get("mimeType")?.takeIf { it.isJsonPrimitive }?.asString
            ?.lowercase(Locale.US)
            .orEmpty()
        if (mimeType.isNotBlank() && !mimeType.startsWith("text/") && !mimeType.contains("json")) {
            return null
        }
        val data = inlineData.get("data")?.takeIf { it.isJsonPrimitive }?.asString ?: return null
        return runCatching {
            val decoded = Base64.decode(data, Base64.DEFAULT)
            String(decoded, Charsets.UTF_8).trim()
        }.getOrNull()?.ifBlank { null }
    }

    private fun buildGeminiDiagnostics(root: JsonObject, candidate: JsonObject?): String {
        val parts = mutableListOf<String>()
        root.getAsJsonObject("promptFeedback")?.let { feedback ->
            feedback.get("blockReason")?.takeIf { it.isJsonPrimitive }?.asString?.takeIf { it.isNotBlank() }?.let {
                parts += "blockReason=$it"
            }
            feedback.get("finishReason")?.takeIf { it.isJsonPrimitive }?.asString?.takeIf { it.isNotBlank() }?.let {
                parts += "finishReason=$it"
            }
        }
        candidate?.get("finishReason")?.takeIf { it.isJsonPrimitive }?.asString?.takeIf { it.isNotBlank() }?.let {
            parts += "finishReason=$it"
        }
        candidate?.get("safetyRatings")?.let { ratings ->
            val ratingSummary = ratings.toString().take(160)
            if (ratingSummary.isNotBlank()) {
                parts += "safetyRatings=$ratingSummary"
            }
        }
        if (parts.isEmpty()) {
            parts += "Raw response shape did not contain text parts."
        }
        return parts.joinToString("; ").take(280)
    }

    private fun buildRequestPayload(
        prompt: String,
        systemPrompt: String?,
        temperature: Double,
        maxOutputTokens: Int,
        jsonMode: Boolean,
        enableGoogleSearch: Boolean,
        cachedContentName: String?,
        structuredOutput: Boolean
    ): String {
        val body = JsonObject().apply {
            if (!cachedContentName.isNullOrBlank()) {
                addProperty("cachedContent", cachedContentName)
            }
            add("contents", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("role", "user")
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", prompt)
                        })
                    })
                })
            })

            add("generationConfig", JsonObject().apply {
                addProperty("temperature", temperature)
                addProperty("maxOutputTokens", min(maxOutputTokens, 8192))
                if (jsonMode) {
                    addProperty("responseMimeType", "application/json")
                    if (structuredOutput) {
                        add("responseSchema", buildStage3ResponseSchema())
                    }
                }
            })

            if (!systemPrompt.isNullOrBlank()) {
                add("systemInstruction", JsonObject().apply {
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", systemPrompt)
                        })
                    })
                })
            }

            if (enableGoogleSearch) {
                add("tools", JsonArray().apply {
                    add(JsonObject().apply {
                        add("google_search", JsonObject())
                    })
                })
            }
        }
        return gson.toJson(body)
    }

    private fun isSearchToolConfigError(error: String): Boolean {
        val normalized = error.lowercase(Locale.US)
        return normalized.contains("google_search") ||
            normalized.contains("unknown name \"tools\"") ||
            normalized.contains("unknown field \"tools\"") ||
            (normalized.contains("invalid_argument") && normalized.contains("tool")) ||
            (normalized.contains("http 400") && normalized.contains("tool"))
    }

    private fun isLocalSystemPromptCacheMiss(error: String): Boolean {
        val normalized = error.lowercase(Locale.US)
        return normalized.contains("system prompt cache miss for key")
    }

    private fun buildLocalSystemPromptCacheKey(systemPrompt: String?): String? {
        if (systemPrompt.isNullOrBlank()) {
            return null
        }
        return LOCAL_STAGE3_SYSTEM_PROMPT_CACHE_KEY
    }

    private fun shouldSendLocalSystemPrompt(cacheKey: String?): Boolean {
        if (cacheKey.isNullOrBlank()) {
            return true
        }
        synchronized(localSystemPromptCacheLock) {
            return !localReadySystemPromptCacheKeys.contains(cacheKey)
        }
    }

    private fun markLocalSystemPromptCacheKeyReady(cacheKey: String?) {
        if (cacheKey.isNullOrBlank()) {
            return
        }
        synchronized(localSystemPromptCacheLock) {
            localReadySystemPromptCacheKeys += cacheKey
        }
    }

    private fun ensureLocalSystemPromptCache(
        localServerBaseUrl: String,
        localModelPath: String,
        cacheKey: String?,
        systemPrompt: String?
    ): String? {
        val normalizedKey = cacheKey?.trim().orEmpty()
        val normalizedPrompt = systemPrompt?.trim().orEmpty()
        if (normalizedKey.isBlank() || normalizedPrompt.isBlank()) {
            return null
        }
        if (!shouldSendLocalSystemPrompt(normalizedKey)) {
            return null
        }

        val baseUrl = localServerBaseUrl.trim().trimEnd('/')
        if (baseUrl.isBlank()) {
            return "Local server URL is empty."
        }

        val endpoint = URL("$baseUrl/v1/cache/system_prompt")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 5000
            readTimeout = 30000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = JsonObject().apply {
            addProperty("cache_key", normalizedKey)
            addProperty("system_prompt", normalizedPrompt)
            if (localModelPath.isNotBlank()) {
                addProperty("model_path", localModelPath)
            }
        }

        return try {
            connection.outputStream.use { out ->
                out.write(gson.toJson(body).toByteArray(StandardCharsets.UTF_8))
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                "HTTP $code: ${short.take(220)}"
            } else {
                markLocalSystemPromptCacheKeyReady(normalizedKey)
                null
            }
        } catch (unknownHost: UnknownHostException) {
            "Local server host is unreachable: ${unknownHost.message ?: "unknown host"}"
        } catch (io: IOException) {
            io.message ?: io.javaClass.simpleName
        } finally {
            connection.disconnect()
        }
    }

    private fun isStructuredOutputConfigError(error: String): Boolean {
        val normalized = error.lowercase(Locale.US)
        return normalized.contains("responseschema") ||
            normalized.contains("response schema") ||
            normalized.contains("unknown name \"responseschema\"") ||
            normalized.contains("unknown field \"responseschema\"") ||
            (normalized.contains("invalid_argument") && normalized.contains("schema")) ||
            (normalized.contains("http 400") && normalized.contains("schema"))
    }

    private fun buildStage3ResponseSchema(): JsonObject {
        // Keep schema permissive to reduce rejection risk while still forcing structured JSON output.
        return JsonObject().apply {
            addProperty("type", "ARRAY")
            add("items", JsonObject().apply {
                addProperty("type", "OBJECT")
                add("properties", JsonObject().apply {
                    add("version", JsonObject().apply { addProperty("type", "STRING") })
                    add("createSurface", JsonObject().apply { addProperty("type", "OBJECT") })
                    add("updateComponents", JsonObject().apply { addProperty("type", "OBJECT") })
                    add("clearSurface", JsonObject().apply { addProperty("type", "OBJECT") })
                })
            })
        }
    }

    private fun ensureStage3InstructionCache(
        apiKey: String,
        model: String,
        systemPrompt: String?
    ): CacheSetupResult {
        if (systemPrompt.isNullOrBlank()) {
            return CacheSetupResult(name = null, created = false, error = null)
        }

        val promptHash = sha256Hex("$model\n$systemPrompt")
        val now = System.currentTimeMillis()

        synchronized(stage3CacheLock) {
            val inMemory = stage3InstructionCache
            if (inMemory != null && inMemory.hash == promptHash && inMemory.expiresAtMs > now + CACHE_EXPIRY_SAFETY_MS) {
                return CacheSetupResult(name = inMemory.name, created = false, error = null)
            }
        }

        readPersistedStage3Cache()?.let { persisted ->
            if (persisted.hash == promptHash && persisted.expiresAtMs > now + CACHE_EXPIRY_SAFETY_MS) {
                synchronized(stage3CacheLock) {
                    stage3InstructionCache = persisted
                }
                return CacheSetupResult(name = persisted.name, created = false, error = null)
            }
        }

        val created = createCachedInstruction(
            apiKey = apiKey,
            model = model,
            promptHash = promptHash,
            systemPrompt = systemPrompt
        )
        if (created.name != null) {
            val expiresAtMs = created.expiresAtMs ?: now + STAGE3_CACHE_TTL_SECONDS * 1000L
            val entry = CachedInstructionEntry(
                hash = promptHash,
                name = created.name,
                expiresAtMs = expiresAtMs
            )
            synchronized(stage3CacheLock) {
                stage3InstructionCache = entry
            }
            persistStage3Cache(entry)
            return CacheSetupResult(name = entry.name, created = true, error = null)
        }
        return CacheSetupResult(name = null, created = false, error = created.error)
    }

    private fun createCachedInstruction(
        apiKey: String,
        model: String,
        promptHash: String,
        systemPrompt: String
    ): CachedInstructionCreateResult {
        val encodedKey = URLEncoder.encode(apiKey, StandardCharsets.UTF_8.name())
        val endpoint = URL("https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$encodedKey")
        val connection = (endpoint.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 20000
            readTimeout = 60000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        val body = JsonObject().apply {
            addProperty("model", "models/$model")
            addProperty("displayName", "genuicraft_stage3_${promptHash.take(12)}")
            addProperty("ttl", "${STAGE3_CACHE_TTL_SECONDS}s")
            add("systemInstruction", JsonObject().apply {
                add("parts", JsonArray().apply {
                    add(JsonObject().apply {
                        addProperty("text", systemPrompt)
                    })
                })
            })
            // Keep one tiny content part so cache creation stays valid across API variants.
            add("contents", JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("role", "user")
                    add("parts", JsonArray().apply {
                        add(JsonObject().apply {
                            addProperty("text", "Use the cached GenUICraft conversion instructions.")
                        })
                    })
                })
            })
        }

        return try {
            connection.outputStream.use { out ->
                out.write(gson.toJson(body).toByteArray(StandardCharsets.UTF_8))
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val short = raw.trim().ifBlank { "HTTP $code" }
                return CachedInstructionCreateResult(
                    name = null,
                    expiresAtMs = null,
                    error = "HTTP $code: ${short.take(180)}"
                )
            }

            val root = runCatching { JsonParser.parseString(raw).asJsonObject }.getOrNull()
                ?: return CachedInstructionCreateResult(
                    name = null,
                    expiresAtMs = null,
                    error = "Cache create response was not valid JSON."
                )
            val name = root.get("name")?.takeIf { it.isJsonPrimitive }?.asString?.trim()
            if (name.isNullOrBlank()) {
                return CachedInstructionCreateResult(
                    name = null,
                    expiresAtMs = null,
                    error = "Cache create response did not include name."
                )
            }
            val expireTime = root.get("expireTime")?.takeIf { it.isJsonPrimitive }?.asString
            CachedInstructionCreateResult(
                name = name,
                expiresAtMs = parseExpireTimeMillis(expireTime),
                error = null
            )
        } catch (io: IOException) {
            CachedInstructionCreateResult(
                name = null,
                expiresAtMs = null,
                error = io.message ?: io.javaClass.simpleName
            )
        } finally {
            connection.disconnect()
        }
    }

    private fun parseExpireTimeMillis(value: String?): Long? {
        if (value.isNullOrBlank()) {
            return null
        }
        return runCatching { Instant.parse(value).toEpochMilli() }.getOrNull()
    }

    private fun sha256Hex(value: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(value.toByteArray(StandardCharsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        digest.forEach { byte ->
            val v = byte.toInt() and 0xff
            if (v < 16) hex.append('0')
            hex.append(v.toString(16))
        }
        return hex.toString()
    }

    private fun readPersistedStage3Cache(): CachedInstructionEntry? {
        val prefs = appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
        val hash = prefs.getString(CACHE_KEY_HASH, null)?.trim().orEmpty()
        val name = prefs.getString(CACHE_KEY_NAME, null)?.trim().orEmpty()
        val expiresAt = prefs.getLong(CACHE_KEY_EXPIRES_AT_MS, 0L)
        if (hash.isBlank() || name.isBlank() || expiresAt <= 0L) {
            return null
        }
        return CachedInstructionEntry(hash = hash, name = name, expiresAtMs = expiresAt)
    }

    private fun persistStage3Cache(entry: CachedInstructionEntry) {
        appContext.getSharedPreferences(CACHE_PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(CACHE_KEY_HASH, entry.hash)
            .putString(CACHE_KEY_NAME, entry.name)
            .putLong(CACHE_KEY_EXPIRES_AT_MS, entry.expiresAtMs)
            .apply()
    }

    private data class GeminiResponse(
        val text: String,
        val rawResponse: String?,
        val error: String?,
        val streamDurationMs: Long?
    )

    private data class StreamReadResult(
        val text: String,
        val streamDurationMs: Long?
    )

    private data class CacheSetupResult(
        val name: String?,
        val created: Boolean,
        val error: String?
    )

    private data class CachedInstructionCreateResult(
        val name: String?,
        val expiresAtMs: Long?,
        val error: String?
    )

    private data class CachedInstructionEntry(
        val hash: String,
        val name: String,
        val expiresAtMs: Long
    )

    private data class GeminiTextExtraction(
        val text: String?,
        val diagnostics: String?
    )

    private data class Stage3PromptContext(
        val systemPrompt: String?,
        val userTemplate: String
    )

    private data class AssetMapping(
        val url: String,
        val localPath: String
    )

    private companion object {
        const val STAGE2_PROMPT_ASSET = "pipeline_prompts/response_gen.md"
        const val STAGE3_PROMPT_ASSET = "pipeline_prompts/genui_gen.md"
        const val MODEL_GEMINI_2_5_PRO = "gemini-2.5-pro"
        const val STAGE2_MAX_OUTPUT_TOKENS = 4096
        const val STAGE3_MAX_OUTPUT_TOKENS = 8192
        const val LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS = 1024
        const val LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS = 15000
        const val STAGE3_CACHE_TTL_SECONDS = 21600
        const val CACHE_PREFS_NAME = "genui_stage_pipeline_cache"
        const val CACHE_KEY_HASH = "stage3_cache_hash"
        const val CACHE_KEY_NAME = "stage3_cache_name"
        const val CACHE_KEY_EXPIRES_AT_MS = "stage3_cache_expires_at_ms"
        const val CACHE_EXPIRY_SAFETY_MS = 60_000L
        const val LOCAL_STAGE3_SYSTEM_PROMPT_CACHE_KEY = "stage3_ir_system_prompt_v1"
        val HOST_LABEL_REGEX = Regex("""(?i)^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$""")
        val URL_TOKEN_REGEX = Regex(
            """(?i)(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?"""
        )

        val stage3CacheLock = Any()
        val localSystemPromptCacheLock = Any()
        @Volatile
        var stage3InstructionCache: CachedInstructionEntry? = null
        val localReadySystemPromptCacheKeys = mutableSetOf<String>()
        val gson = com.google.gson.GsonBuilder().disableHtmlEscaping().create()
    }
}
