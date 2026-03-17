package com.samsung.genuicraft

import android.content.Context
import android.util.Log
import com.google.gson.GsonBuilder
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.InferenceBackendFactory
import com.samsung.genuicraft.inference.LocalServerBackend
import com.samsung.genuicraft.mcp.McpClient
import com.samsung.genuicraft.mcp.McpLlmRouter
import com.samsung.genuicraft.mcp.McpResponseFormatter
import com.samsung.genuicraft.mcp.McpSettings
import com.samsung.genuicraft.pipeline.PipelineCacheManager
import com.samsung.genuicraft.pipeline.PipelineJsonExtractor
import com.samsung.genuicraft.pipeline.PipelineMediaSanitizer
import com.samsung.genuicraft.pipeline.PipelinePromptBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.withContext

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

    private val cacheManager = PipelineCacheManager(appContext)

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

        GeminiApiKeyProvider.refresh(appContext)

        val responseProvider = InferenceBackendSettings.getResponseProvider(appContext)
        val irProvider = InferenceBackendSettings.getIrProvider(appContext)
        val responseModel = GeminiModelSettings.getResponseModel(appContext)
        val irModel = GeminiModelSettings.getIrModel(appContext)
        val localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(appContext)
        val localModelPath = InferenceBackendSettings.getLocalModelPath(appContext)
        val stage2MaxOutputTokens = if (responseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS
        } else {
            STAGE2_MAX_OUTPUT_TOKENS
        }
        val stage3MaxOutputTokens = if (irProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS
        } else {
            STAGE3_MAX_OUTPUT_TOKENS
        }
        val stage3RepairMaxOutputTokens = stage3MaxOutputTokens

        val responseApiKey = if (responseProvider == InferenceBackendSettings.Provider.GEMINI) {
            GeminiApiKeyProvider.stage2ApiKey(appContext).trim()
        } else {
            ""
        }
        val irApiKey = if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
            GeminiApiKeyProvider.stage3ApiKey(appContext).trim()
        } else {
            ""
        }
        if (responseProvider == InferenceBackendSettings.Provider.GEMINI && responseApiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Gemini stage-2 key is missing. Add GEMINI_STAGE2_API_KEY (or GEMINI_RESPONSE_API_KEY / GEMINI_API_KEY) at ${GeminiApiKeyProvider.setupHintPath(appContext)}"
            )
        }
        if (irProvider == InferenceBackendSettings.Provider.GEMINI && irApiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Gemini stage-3 key is missing. Add GEMINI_IR_API_KEY (or GEMINI_API_KEY_2) at ${GeminiApiKeyProvider.setupHintPath(appContext)}"
            )
        }
        if ((responseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER ||
                irProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) &&
            localServerBaseUrl.isBlank()
        ) {
            val failureStage = if (responseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
                Stage.STAGE2
            } else {
                Stage.STAGE3
            }
            return@withContext Outcome.Failure(
                stage = failureStage,
                message = "Local server URL is missing. Open Settings and configure Local Server."
            )
        }
        if (responseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            postUpdate(onStageUpdate, Stage.STAGE2, "Checking local server connectivity")
            val healthResult = LocalServerBackend(localServerBaseUrl, localModelPath).checkHealth()
            if (!healthResult.healthy) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE2,
                    message = healthResult.errorMessage ?: "Local server health check failed."
                )
            }
        }

        val responseBackend = InferenceBackendFactory.create(
            provider = responseProvider,
            apiKey = responseApiKey,
            model = responseModel,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath
        )
        val irBackend = InferenceBackendFactory.create(
            provider = irProvider,
            apiKey = irApiKey,
            model = irModel,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath
        )

        // ── MCP path: LLM routes query → optional live data fetch ──────────
        // When MCP is enabled, Stage 2 uses a special routing prompt.
        // The LLM decides which domain (if any) to call and provides:
        //   - intro text (2-3 sentences) to show before live data, OR
        //   - full_response when no MCP domain is needed (treated as normal Stage 2 output).
        val mcpEnabled = McpSettings.isEnabled(appContext)
        if (mcpEnabled) {
            postUpdate(onStageUpdate, Stage.STAGE2, "Fetching data")
            val stage2StartedAtMs = System.currentTimeMillis()
            val routerResult = McpLlmRouter.route(
                query = normalizedQuery,
                backend = responseBackend,
                assets = appContext.assets,
                maxOutputTokens = if (responseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
                    LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS
                } else {
                    STAGE2_MAX_OUTPUT_TOKENS
                }
            )
            markStreamDuration(Stage.STAGE2, routerResult.streamDurationMs)

            when {
                routerResult.error != null -> {
                    // Router call failed entirely — fall through to normal Stage 2
                    Log.w(LOG_TAG, "MCP router failed: ${routerResult.error}; falling back to normal Stage 2")
                    markDuration(Stage.STAGE2, stage2StartedAtMs)
                }

                routerResult.domain != null && McpSettings.isDomainReady(appContext, routerResult.domain) -> {
                    // LLM identified a live-data domain → fetch MCP data
                    Log.i(LOG_TAG, "MCP router: domain=${routerResult.domain.key} entities=${routerResult.entities}")
                    postUpdate(onStageUpdate, Stage.STAGE2, "Fetching live ${routerResult.domain.displayName} data")
                    val mcpApiKey = McpSettings.getApiKey(appContext, routerResult.domain)
                    val mcpResult = McpClient.fetch(
                        domain = routerResult.domain,
                        entities = routerResult.entities,
                        apiKey = mcpApiKey,
                        queryText = normalizedQuery
                    )
                    val mcpDataSection = McpResponseFormatter.buildDataSection(mcpResult, normalizedQuery)
                    if (mcpDataSection.isBlank()) {
                        // MCP returned empty data — fall through to normal Stage 2 LLM
                        Log.w(LOG_TAG, "MCP ${routerResult.domain.key} returned no data; falling back to Stage 2")
                        markDuration(Stage.STAGE2, stage2StartedAtMs)
                        // (falls through to normal Stage 2 below)
                    } else {
                        // Combine: LLM intro (context) + live MCP data section
                        val combinedResponse = buildString {
                            if (!routerResult.introText.isNullOrBlank()) {
                                appendLine(routerResult.introText)
                                appendLine()
                            }
                            append(mcpDataSection)
                        }
                        markDuration(Stage.STAGE2, stage2StartedAtMs)
                        val mcpWarnings = mutableListOf<String>()
                        mcpWarnings += "MCP: ${routerResult.domain.displayName} (LLM-routed + live data)"
                        mcpWarnings += "Response backend: ${responseProvider.rawValue}"
                        mcpWarnings += "IR backend: ${irProvider.rawValue}"
                        return@withContext executeStage3WithResponse(
                            normalizedQuery = normalizedQuery,
                            stage2Response = combinedResponse,
                            stage2Prompt = "[MCP-routed:${routerResult.domain.key}] $normalizedQuery",
                            irBackend = irBackend,
                            irProvider = irProvider,
                            irModel = irModel,
                            localServerBaseUrl = localServerBaseUrl,
                            localModelPath = localModelPath,
                            stageDurationsMs = stageDurationsMs,
                            stageStreamDurationsMs = stageStreamDurationsMs,
                            extraWarnings = mcpWarnings,
                            onStageUpdate = onStageUpdate
                        )
                    }
                }

                !routerResult.fullResponse.isNullOrBlank() -> {
                    // LLM said no MCP needed and already wrote the full response
                    Log.i(LOG_TAG, "MCP router: domain=none, using LLM full_response")
                    markDuration(Stage.STAGE2, stage2StartedAtMs)
                    val mcpWarnings = mutableListOf<String>()
                    mcpWarnings += "MCP: LLM routing decided no live data needed — using LLM response"
                    mcpWarnings += "Response backend: ${responseProvider.rawValue}"
                    mcpWarnings += "IR backend: ${irProvider.rawValue}"
                    return@withContext executeStage3WithResponse(
                        normalizedQuery = normalizedQuery,
                        stage2Response = routerResult.fullResponse,
                        stage2Prompt = "[MCP-routed:none] $normalizedQuery",
                        irBackend = irBackend,
                        irProvider = irProvider,
                        irModel = irModel,
                        localServerBaseUrl = localServerBaseUrl,
                        localModelPath = localModelPath,
                        stageDurationsMs = stageDurationsMs,
                        stageStreamDurationsMs = stageStreamDurationsMs,
                        extraWarnings = mcpWarnings,
                        onStageUpdate = onStageUpdate
                    )
                }

                else -> {
                    // Router returned domain but API key not ready — fall through
                    val domainName = routerResult.domain?.key ?: "unknown"
                    Log.w(LOG_TAG, "MCP domain $domainName identified but API key not configured; falling back to normal Stage 2")
                    markDuration(Stage.STAGE2, stage2StartedAtMs)
                }
            }
        }

        val responseTemplate = runCatching {
            PipelinePromptBuilder.loadPromptAsset(appContext.assets, PipelinePromptBuilder.STAGE2_PROMPT_ASSET)
        }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE2,
                    message = "Could not load stage 2 prompt: ${it.message ?: it.javaClass.simpleName}"
                )
            }
        val stage2PromptContext = PipelinePromptBuilder.prepareStage2PromptContext(responseTemplate)
        val stage2Prompt = PipelinePromptBuilder.renderPrompt(
            stage2PromptContext.userTemplate,
            mapOf("query_text" to normalizedQuery)
        )
        val stage2CacheDeferred = if (responseProvider == InferenceBackendSettings.Provider.GEMINI) {
            async(Dispatchers.IO) {
                cacheManager.ensureStage2InstructionCache(
                    apiKey = responseApiKey,
                    model = responseModel,
                    systemPrompt = stage2PromptContext.systemPrompt
                )
            }
        } else {
            null
        }

        val genUiTemplate = runCatching {
            PipelinePromptBuilder.loadPromptAsset(appContext.assets, PipelinePromptBuilder.STAGE3_PROMPT_ASSET)
        }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}"
                )
            }
        val promptContext = PipelinePromptBuilder.prepareStage3PromptContext(genUiTemplate)
        val stage3CacheDeferred = if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
            async(Dispatchers.IO) {
                cacheManager.ensureStage3InstructionCache(
                    apiKey = irApiKey,
                    model = irModel,
                    systemPrompt = promptContext.systemPrompt
                )
            }
        } else {
            null
        }

        postUpdate(onStageUpdate, Stage.STAGE2, "Fetching response")
        val stage2Cache = if (stage2CacheDeferred != null) {
            runCatching { stage2CacheDeferred.await() }
                .getOrElse {
                    PipelineCacheManager.CacheSetupResult(
                        name = null,
                        created = false,
                        error = it.message ?: it.javaClass.simpleName
                    )
                }
        } else {
            PipelineCacheManager.CacheSetupResult(name = null, created = false, error = null)
        }
        val stage2StartedAtMs = System.currentTimeMillis()
        val stage2Temperature = if (PipelineMediaSanitizer.looksLikeFlightQuery(normalizedQuery)) 0.15 else 0.3
        val stage2Call = generateWithRetry(
            backend = responseBackend,
            provider = responseProvider,
            prompt = stage2Prompt,
            systemPrompt = if (stage2Cache.name != null) null else stage2PromptContext.systemPrompt,
            temperature = stage2Temperature,
            maxOutputTokens = stage2MaxOutputTokens,
            jsonMode = false,
            enableGoogleSearch = responseProvider == InferenceBackendSettings.Provider.GEMINI,
            cachedContentName = stage2Cache.name,
            allowCachedContent = true,
            structuredOutput = false,
            geminiCacheFallbackSystemPrompt = stage2PromptContext.systemPrompt,
            onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage2InstructionCache(reason) }
        )
        markStreamDuration(Stage.STAGE2, stage2Call.streamDurationMs)
        if (stage2Call.error != null) {
            markDuration(Stage.STAGE2, stage2StartedAtMs)
            stage2CacheDeferred?.cancel()
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
            stage2CacheDeferred?.cancel()
            stage3CacheDeferred?.cancel()
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Stage 2 returned empty output.",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        val stage2WithFlightList = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = stage2ResponseRaw,
            queryText = normalizedQuery
        )
        val stage2WithActions = PipelineMediaSanitizer.ensureFlightQuickActions(
            responseText = stage2WithFlightList,
            queryText = normalizedQuery
        )
        val stage2WithFlightMedia = PipelineMediaSanitizer.sanitizeFlightInlineMedia(
            responseText = stage2WithActions,
            queryText = normalizedQuery
        )
        val stage2WithTravelMediaSanitized = PipelineMediaSanitizer.sanitizeTravelInlineMedia(
            responseText = stage2WithFlightMedia,
            queryText = normalizedQuery
        )
        val stage2WithTravelMedia = PipelineMediaSanitizer.ensureTravelInlineMedia(
            responseText = stage2WithTravelMediaSanitized,
            queryText = normalizedQuery
        )
        val stage2WithGeneralMedia = PipelineMediaSanitizer.ensureGeneralInlineMedia(
            responseText = stage2WithTravelMedia,
            queryText = normalizedQuery
        )
        val stage2Response = PipelineMediaSanitizer.normalizeUrlTokensForDisplay(stage2WithGeneralMedia)
        val injectedFlightList = stage2WithFlightList != stage2ResponseRaw
        val normalizedBareDomains = stage2Response != stage2WithGeneralMedia
        val removedFlightMedia = stage2WithFlightMedia != stage2WithActions
        val sanitizedTravelMedia = stage2WithTravelMediaSanitized != stage2WithFlightMedia
        val injectedTravelMedia = stage2WithTravelMedia != stage2WithTravelMediaSanitized
        markDuration(Stage.STAGE2, stage2StartedAtMs)

        val catalogId = PipelineMediaSanitizer.resolveStage3CatalogId(
            appContext.getSharedPreferences(PipelineMediaSanitizer.APP_PREFS_NAME, Context.MODE_PRIVATE)
        )
        val stage3Prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = stage2Response,
            catalogId = catalogId,
            assets = emptyList()
        )
        if (irProvider == InferenceBackendSettings.Provider.LOCAL_SERVER &&
            responseProvider != InferenceBackendSettings.Provider.LOCAL_SERVER
        ) {
            postUpdate(onStageUpdate, Stage.STAGE3, "Checking local server connectivity")
            val healthResult = LocalServerBackend(localServerBaseUrl, localModelPath).checkHealth()
            if (!healthResult.healthy) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = healthResult.errorMessage ?: "Local server health check failed.",
                    stageDurationsMs = stageDurationsMs.toMap(),
                    stageStreamDurationsMs = stageStreamDurationsMs.toMap()
                )
            }
        }

        val localStage3SystemPromptCacheKey = if (irProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            cacheManager.buildLocalSystemPromptCacheKey(
                systemPrompt = promptContext.systemPrompt
            )
        } else {
            null
        }
        val warnings = mutableListOf<String>()
        warnings += "Response backend: ${responseProvider.rawValue}"
        warnings += "IR backend: ${irProvider.rawValue}"
        if (responseProvider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Gemini response model: $responseModel"
            if (stage2Cache.name != null && !stage2Cache.created) {
                warnings += "Stage 2 instruction cache hit."
            }
            if (stage2Cache.created) {
                warnings += "Stage 2 instruction cache created."
            }
            stage2Cache.error?.let {
                warnings += "Stage 2 instruction cache unavailable ($it). Using direct prompt."
            }
        } else {
            warnings += "Local server (response): $localServerBaseUrl"
            warnings += "Local model path (response): $localModelPath"
            warnings += "Local token cap (response): stage2=$stage2MaxOutputTokens"
        }
        if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Gemini IR model: $irModel"
        } else {
            warnings += "Local server (IR): $localServerBaseUrl"
            warnings += "Local model path (IR): $localModelPath"
            warnings += "Local token cap (IR): stage3=$stage3MaxOutputTokens"
            if (!localStage3SystemPromptCacheKey.isNullOrBlank()) {
                warnings += "Local stage3 prompt cache key: ${localStage3SystemPromptCacheKey.take(16)}..."
            }
        }
        if (normalizedBareDomains) {
            warnings += "Normalized bare source/action domains to https URLs."
        }
        if (injectedFlightList) {
            warnings += "Added fallback flight comparison list."
        }
        if (removedFlightMedia) {
            warnings += "Removed unrelated media lines from flight response."
        }
        if (sanitizedTravelMedia) {
            warnings += "Sanitized travel media URLs to better match itinerary content."
        }
        if (injectedTravelMedia) {
            warnings += "Added fallback inline media for travel sections missing media."
        }
        val stage3Cache = if (stage3CacheDeferred != null) {
            runCatching { stage3CacheDeferred.await() }
                .getOrElse {
                    PipelineCacheManager.CacheSetupResult(
                        name = null,
                        created = false,
                        error = it.message ?: it.javaClass.simpleName
                    )
                }
        } else {
            PipelineCacheManager.CacheSetupResult(name = null, created = false, error = null)
        }
        if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
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
        var localSendStage3SystemPrompt = cacheManager.shouldSendLocalSystemPrompt(localStage3SystemPromptCacheKey)
        if (irProvider == InferenceBackendSettings.Provider.LOCAL_SERVER && !localStage3SystemPromptCacheKey.isNullOrBlank()) {
            if (localSendStage3SystemPrompt) {
                val primeResult = cacheManager.ensureLocalSystemPromptCache(
                    localServerBaseUrl = localServerBaseUrl,
                    localModelPath = localModelPath,
                    cacheKey = localStage3SystemPromptCacheKey,
                    systemPrompt = promptContext.systemPrompt
                )
                when {
                    !primeResult.error.isNullOrBlank() -> {
                        warnings += "Local stage3 KV prefix cache prime failed (${primeResult.error})."
                        Log.w(LOG_TAG, "Local stage3 KV prefix cache prime failed: ${primeResult.error}")
                    }
                    primeResult.cacheHit == true -> {
                        warnings += "Local stage3 KV prefix cache hit."
                    }
                    primeResult.cacheHit == false -> {
                        warnings += "Local stage3 KV prefix cache miss -> primed."
                    }
                    else -> {
                        warnings += "Local stage3 KV prefix cache primed."
                    }
                }
                if (primeResult.error.isNullOrBlank()) {
                    localSendStage3SystemPrompt = cacheManager.shouldSendLocalSystemPrompt(localStage3SystemPromptCacheKey)
                }
            } else {
                warnings += "Local stage3 KV prefix cache already primed (skipping prime request)."
                Log.i(LOG_TAG, "Local stage3 KV prefix cache already primed for key=$localStage3SystemPromptCacheKey; skipping prime.")
            }
            warnings += if (localSendStage3SystemPrompt) {
                "Local stage3 KV prefix cache miss path (system_prompt sent)."
            } else {
                "Local stage3 KV prefix cache hit path (cache key only)."
            }
        }

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            backend = irBackend,
            provider = irProvider,
            prompt = stage3Prompt,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            temperature = 0.2,
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            structuredOutput = irProvider == InferenceBackendSettings.Provider.GEMINI,
            localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
            localSendSystemPrompt = localSendStage3SystemPrompt,
            geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
            onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage3InstructionCache(reason) }
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

        var stage3JsonElement = PipelineJsonExtractor.extractJsonElement(stage3Call.text)
        var usedFallback = false

        if (stage3JsonElement == null) {
            warnings += "Stage 3 JSON parse failed; running repair pass."
            val repairCall = generateWithRetry(
                backend = irBackend,
                provider = irProvider,
                prompt = PipelineJsonExtractor.buildRepairPrompt(stage3Call.text),
                systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
                temperature = 0.2,
                maxOutputTokens = stage3RepairMaxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = stage3Cache.name,
                allowCachedContent = true,
                structuredOutput = irProvider == InferenceBackendSettings.Provider.GEMINI,
                localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
                localSendSystemPrompt = localSendStage3SystemPrompt,
                geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
                onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage3InstructionCache(reason) }
            )
            markStreamDuration(Stage.STAGE3, repairCall.streamDurationMs)
            if (repairCall.error == null) {
                stage3JsonElement = PipelineJsonExtractor.extractJsonElement(repairCall.text)
            }
        }

        if (stage3JsonElement == null) {
            warnings += "Stage 3 fallback JSON was used."
            stage3JsonElement = PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId)
            usedFallback = true
        }

        val normalizedGenUi = PipelineMediaSanitizer.normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        if (!usedFallback) {
            val stage3WithInjectedImage = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
                jsonText = stage3Json,
                stage2Response = stage2Response,
                queryText = normalizedQuery
            )
            if (stage3WithInjectedImage != stage3Json) {
                stage3Json = stage3WithInjectedImage
                warnings += "Injected fallback image into IR output to preserve media."
            }
        }
        if (!usedFallback) {
            val stage3WithFlightMediaNormalized = PipelineMediaSanitizer.normalizeFlightMediaInGenUi(
                jsonText = stage3Json,
                queryText = normalizedQuery,
                stage2Response = stage2Response
            )
            if (stage3WithFlightMediaNormalized != stage3Json) {
                stage3Json = stage3WithFlightMediaNormalized
                warnings += "Normalized flight media to airline/travel-safe icons."
            }
        }
        if (!usedFallback) {
            val stage3WithStableMediaUrls = PipelineMediaSanitizer.rewriteUnstableMediaHostsInGenUi(
                jsonText = stage3Json,
                queryText = normalizedQuery
            )
            if (stage3WithStableMediaUrls != stage3Json) {
                stage3Json = stage3WithStableMediaUrls
                warnings += "Rewrote unstable media URLs to deterministic topic icons."
            }
        }
        if (!usedFallback && !PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)) {
            val stage3WithInlineTextMedia = PipelineMediaSanitizer.ensureGenUiHasInlineTextMedia(
                jsonText = stage3Json,
                queryText = normalizedQuery
            )
            if (stage3WithInlineTextMedia != stage3Json) {
                stage3Json = stage3WithInlineTextMedia
                warnings += "Injected fallback inline media text to preserve image rendering."
            }
        }
        val stage2HasInlineImage = PipelineMediaSanitizer.hasInlineImageUrl(stage2Response)
        val stage2HasInlineIcon = PipelineMediaSanitizer.hasInlineIconUrl(stage2Response)
        var stage3HasInlineImage = PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = PipelineMediaSanitizer.genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if ((missingInlineImage || missingInlineIcon) && !usedFallback) {
            warnings += "Media content was adjusted for compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId))
            usedFallback = true
        }
        if (PipelineMediaSanitizer.responseContainsActionButtons(stage2Response) && !PipelineMediaSanitizer.genUiPreservesActionButtons(stage3Json) && !usedFallback) {
            warnings += "Quick actions were adjusted for compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId))
            usedFallback = true
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        var renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

        if (renderResult.errorMessage != null && !usedFallback) {
            warnings += "Native rendering failed for stage 3 output; using fallback UI."
            val fallback = PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId)
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

        GeminiApiKeyProvider.refresh(appContext)

        val provider = InferenceBackendSettings.getIrProvider(appContext)
        val irModel = GeminiModelSettings.getIrModel(appContext)
        val localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(appContext)
        val localModelPath = InferenceBackendSettings.getLocalModelPath(appContext)
        val isLocalServer = provider == InferenceBackendSettings.Provider.LOCAL_SERVER
        val stage3MaxOutputTokens = if (isLocalServer) LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS else STAGE3_MAX_OUTPUT_TOKENS
        val stage3RepairMaxOutputTokens = stage3MaxOutputTokens

        val apiKey = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            GeminiApiKeyProvider.stage3ApiKey(appContext).trim()
        } else {
            ""
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI && apiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Gemini stage-3 key is missing. Add GEMINI_IR_API_KEY (or GEMINI_API_KEY_2) at ${GeminiApiKeyProvider.setupHintPath(appContext)}",
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
            val healthResult = LocalServerBackend(localServerBaseUrl, localModelPath).checkHealth()
            if (!healthResult.healthy) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = healthResult.errorMessage ?: "Local server health check failed.",
                    stageDurationsMs = stageDurationsMs.toMap(),
                    stageStreamDurationsMs = stageStreamDurationsMs.toMap()
                )
            }
        }

        val backend = InferenceBackendFactory.create(
            provider = provider,
            apiKey = apiKey,
            model = irModel,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath
        )

        val genUiTemplate = runCatching {
            PipelinePromptBuilder.loadPromptAsset(appContext.assets, PipelinePromptBuilder.STAGE3_PROMPT_ASSET)
        }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}",
                    stageDurationsMs = stageDurationsMs.toMap(),
                    stageStreamDurationsMs = stageStreamDurationsMs.toMap()
                )
            }
        val promptContext = PipelinePromptBuilder.prepareStage3PromptContext(genUiTemplate)
        val stage3CacheDeferred = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            async(Dispatchers.IO) {
                cacheManager.ensureStage3InstructionCache(
                    apiKey = apiKey,
                    model = irModel,
                    systemPrompt = promptContext.systemPrompt
                )
            }
        } else {
            null
        }

        val stage2WithFlightList = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = normalizedResponseRaw,
            queryText = normalizedQuery
        )
        val stage2WithActions = PipelineMediaSanitizer.ensureFlightQuickActions(
            responseText = stage2WithFlightList,
            queryText = normalizedQuery
        )
        val stage2WithFlightMedia = PipelineMediaSanitizer.sanitizeFlightInlineMedia(
            responseText = stage2WithActions,
            queryText = normalizedQuery
        )
        val stage2WithTravelMediaSanitized = PipelineMediaSanitizer.sanitizeTravelInlineMedia(
            responseText = stage2WithFlightMedia,
            queryText = normalizedQuery
        )
        val stage2WithTravelMedia = PipelineMediaSanitizer.ensureTravelInlineMedia(
            responseText = stage2WithTravelMediaSanitized,
            queryText = normalizedQuery
        )
        val stage2WithGeneralMedia = PipelineMediaSanitizer.ensureGeneralInlineMedia(
            responseText = stage2WithTravelMedia,
            queryText = normalizedQuery
        )
        val stage2Response = PipelineMediaSanitizer.normalizeUrlTokensForDisplay(stage2WithGeneralMedia)
        val injectedFlightList = stage2WithFlightList != normalizedResponseRaw
        val normalizedBareDomains = stage2Response != stage2WithGeneralMedia
        val removedFlightMedia = stage2WithFlightMedia != stage2WithActions
        val sanitizedTravelMedia = stage2WithTravelMediaSanitized != stage2WithFlightMedia
        val injectedTravelMedia = stage2WithTravelMedia != stage2WithTravelMediaSanitized

        val catalogId = PipelineMediaSanitizer.resolveStage3CatalogId(
            appContext.getSharedPreferences(PipelineMediaSanitizer.APP_PREFS_NAME, Context.MODE_PRIVATE)
        )
        val stage3Prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = stage2Response,
            catalogId = catalogId,
            assets = emptyList()
        )
        val localStage3SystemPromptCacheKey = if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            cacheManager.buildLocalSystemPromptCacheKey(
                systemPrompt = promptContext.systemPrompt
            )
        } else {
            null
        }
        val warnings = mutableListOf<String>()
        warnings += "Using preloaded IR demo response (stage 2 skipped)."
        if (provider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Gemini IR model: $irModel"
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
        if (injectedFlightList) {
            warnings += "Added fallback flight comparison list."
        }
        if (removedFlightMedia) {
            warnings += "Removed unrelated media lines from flight response."
        }
        if (sanitizedTravelMedia) {
            warnings += "Sanitized travel media URLs to better match itinerary content."
        }
        if (injectedTravelMedia) {
            warnings += "Added fallback inline media for travel sections missing media."
        }
        val stage3Cache = if (stage3CacheDeferred != null) {
            runCatching { stage3CacheDeferred.await() }
                .getOrElse {
                    PipelineCacheManager.CacheSetupResult(
                        name = null,
                        created = false,
                        error = it.message ?: it.javaClass.simpleName
                    )
                }
        } else {
            PipelineCacheManager.CacheSetupResult(name = null, created = false, error = null)
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
        var localSendStage3SystemPrompt = cacheManager.shouldSendLocalSystemPrompt(localStage3SystemPromptCacheKey)
        if (provider == InferenceBackendSettings.Provider.LOCAL_SERVER && !localStage3SystemPromptCacheKey.isNullOrBlank()) {
            if (localSendStage3SystemPrompt) {
                val primeResult = cacheManager.ensureLocalSystemPromptCache(
                    localServerBaseUrl = localServerBaseUrl,
                    localModelPath = localModelPath,
                    cacheKey = localStage3SystemPromptCacheKey,
                    systemPrompt = promptContext.systemPrompt
                )
                when {
                    !primeResult.error.isNullOrBlank() -> {
                        warnings += "Local stage3 KV prefix cache prime failed (${primeResult.error})."
                        Log.w(LOG_TAG, "Local stage3 KV prefix cache prime failed: ${primeResult.error}")
                    }
                    primeResult.cacheHit == true -> {
                        warnings += "Local stage3 KV prefix cache hit."
                    }
                    primeResult.cacheHit == false -> {
                        warnings += "Local stage3 KV prefix cache miss -> primed."
                    }
                    else -> {
                        warnings += "Local stage3 KV prefix cache primed."
                    }
                }
                if (primeResult.error.isNullOrBlank()) {
                    localSendStage3SystemPrompt = cacheManager.shouldSendLocalSystemPrompt(localStage3SystemPromptCacheKey)
                }
            } else {
                warnings += "Local stage3 KV prefix cache already primed (skipping prime request)."
                Log.i(LOG_TAG, "Local stage3 KV prefix cache already primed for key=$localStage3SystemPromptCacheKey; skipping prime.")
            }
            warnings += if (localSendStage3SystemPrompt) {
                "Local stage3 KV prefix cache miss path (system_prompt sent)."
            } else {
                "Local stage3 KV prefix cache hit path (cache key only)."
            }
        }

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            backend = backend,
            provider = provider,
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
            localSendSystemPrompt = localSendStage3SystemPrompt,
            geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
            onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage3InstructionCache(reason) }
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

        var stage3JsonElement = PipelineJsonExtractor.extractJsonElement(stage3Call.text)
        var usedFallback = false

        if (stage3JsonElement == null) {
            warnings += "Stage 3 JSON parse failed; running repair pass."
            val repairCall = generateWithRetry(
                backend = backend,
                provider = provider,
                prompt = PipelineJsonExtractor.buildRepairPrompt(stage3Call.text),
                systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
                temperature = 0.2,
                maxOutputTokens = stage3RepairMaxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = stage3Cache.name,
                allowCachedContent = true,
                structuredOutput = provider == InferenceBackendSettings.Provider.GEMINI,
                localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
                localSendSystemPrompt = localSendStage3SystemPrompt,
                geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
                onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage3InstructionCache(reason) }
            )
            markStreamDuration(Stage.STAGE3, repairCall.streamDurationMs)
            if (repairCall.error == null) {
                stage3JsonElement = PipelineJsonExtractor.extractJsonElement(repairCall.text)
            }
        }

        if (stage3JsonElement == null) {
            warnings += "Stage 3 fallback JSON was used."
            stage3JsonElement = PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId)
            usedFallback = true
        }

        val normalizedGenUi = PipelineMediaSanitizer.normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        if (!usedFallback) {
            val stage3WithInjectedImage = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
                jsonText = stage3Json,
                stage2Response = stage2Response,
                queryText = normalizedQuery
            )
            if (stage3WithInjectedImage != stage3Json) {
                stage3Json = stage3WithInjectedImage
                warnings += "Injected fallback image into IR output to preserve media."
            }
        }
        if (!usedFallback) {
            val stage3WithFlightMediaNormalized = PipelineMediaSanitizer.normalizeFlightMediaInGenUi(
                jsonText = stage3Json,
                queryText = normalizedQuery,
                stage2Response = stage2Response
            )
            if (stage3WithFlightMediaNormalized != stage3Json) {
                stage3Json = stage3WithFlightMediaNormalized
                warnings += "Normalized flight media to airline/travel-safe icons."
            }
        }
        if (!usedFallback) {
            val stage3WithStableMediaUrls = PipelineMediaSanitizer.rewriteUnstableMediaHostsInGenUi(
                jsonText = stage3Json,
                queryText = normalizedQuery
            )
            if (stage3WithStableMediaUrls != stage3Json) {
                stage3Json = stage3WithStableMediaUrls
                warnings += "Rewrote unstable media URLs to deterministic topic icons."
            }
        }
        if (!usedFallback && !PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)) {
            val stage3WithInlineTextMedia = PipelineMediaSanitizer.ensureGenUiHasInlineTextMedia(
                jsonText = stage3Json,
                queryText = normalizedQuery
            )
            if (stage3WithInlineTextMedia != stage3Json) {
                stage3Json = stage3WithInlineTextMedia
                warnings += "Injected fallback inline media text to preserve image rendering."
            }
        }
        val stage2HasInlineImage = PipelineMediaSanitizer.hasInlineImageUrl(stage2Response)
        val stage2HasInlineIcon = PipelineMediaSanitizer.hasInlineIconUrl(stage2Response)
        var stage3HasInlineImage = PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = PipelineMediaSanitizer.genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if ((missingInlineImage || missingInlineIcon) && !usedFallback) {
            warnings += "Media content was adjusted for compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId))
            usedFallback = true
        }
        if (PipelineMediaSanitizer.responseContainsActionButtons(stage2Response) && !PipelineMediaSanitizer.genUiPreservesActionButtons(stage3Json) && !usedFallback) {
            warnings += "Quick actions were adjusted for compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId))
            usedFallback = true
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        var renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

        if (renderResult.errorMessage != null && !usedFallback) {
            warnings += "Native rendering failed for stage 3 output; using fallback UI."
            val fallback = PipelineMediaSanitizer.buildFallbackGenUi(stage2Response, catalogId)
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

    /**
     * Runs Stage 3 + Stage 4 on a pre-formatted Stage 2 response (e.g. from MCP).
     * Shares the same media sanitization, IR generation, and rendering logic as execute().
     */
    private suspend fun kotlinx.coroutines.CoroutineScope.executeStage3WithResponse(
        normalizedQuery: String,
        stage2Response: String,
        stage2Prompt: String,
        irBackend: InferenceBackend,
        irProvider: InferenceBackendSettings.Provider,
        irModel: String,
        localServerBaseUrl: String,
        localModelPath: String,
        stageDurationsMs: LinkedHashMap<Stage, Long>,
        stageStreamDurationsMs: LinkedHashMap<Stage, Long>,
        extraWarnings: List<String>,
        onStageUpdate: (StageUpdate) -> Unit
    ): Outcome {
        fun markDuration(stage: Stage, startMs: Long) {
            if (startMs <= 0L) return
            stageDurationsMs[stage] = (System.currentTimeMillis() - startMs).coerceAtLeast(0L)
        }
        fun markStreamDuration(stage: Stage, durationMs: Long?) {
            val duration = durationMs ?: return
            stageStreamDurationsMs[stage] = (stageStreamDurationsMs[stage] ?: 0L) + duration.coerceAtLeast(0L)
        }

        // Media sanitization chain (same as main execute path)
        val stage2WithFlightList = PipelineMediaSanitizer.ensureFlightListContent(
            responseText = stage2Response, queryText = normalizedQuery
        )
        val stage2WithActions = PipelineMediaSanitizer.ensureFlightQuickActions(
            responseText = stage2WithFlightList, queryText = normalizedQuery
        )
        val stage2WithFlightMedia = PipelineMediaSanitizer.sanitizeFlightInlineMedia(
            responseText = stage2WithActions, queryText = normalizedQuery
        )
        val stage2WithTravelMediaSanitized = PipelineMediaSanitizer.sanitizeTravelInlineMedia(
            responseText = stage2WithFlightMedia, queryText = normalizedQuery
        )
        val stage2WithTravelMedia = PipelineMediaSanitizer.ensureTravelInlineMedia(
            responseText = stage2WithTravelMediaSanitized, queryText = normalizedQuery
        )
        val stage2WithGeneralMedia = PipelineMediaSanitizer.ensureGeneralInlineMedia(
            responseText = stage2WithTravelMedia, queryText = normalizedQuery
        )
        val sanitizedResponse = PipelineMediaSanitizer.normalizeUrlTokensForDisplay(stage2WithGeneralMedia)

        // Shorten long URLs to compact tokens before sending to Stage 3 LLM
        val urlShortenResult = com.samsung.genuicraft.mcp.McpUrlShortener.shorten(sanitizedResponse)
        val stage3InputResponse = urlShortenResult.shortenedText
        val urlMap = urlShortenResult.urlMap

        val warnings = extraWarnings.toMutableList()

        val catalogId = PipelineMediaSanitizer.resolveStage3CatalogId(
            appContext.getSharedPreferences(PipelineMediaSanitizer.APP_PREFS_NAME, android.content.Context.MODE_PRIVATE)
        )

        val genUiTemplate = runCatching {
            PipelinePromptBuilder.loadPromptAsset(appContext.assets, PipelinePromptBuilder.STAGE3_PROMPT_ASSET)
        }.getOrElse {
            return Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}",
                stage2Response = sanitizedResponse,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        val promptContext = PipelinePromptBuilder.prepareStage3PromptContext(genUiTemplate)
        val stage3Prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = stage3InputResponse,
            catalogId = catalogId,
            assets = emptyList()
        )

        val stage3CacheDeferred = if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
            async(Dispatchers.IO) {
                cacheManager.ensureStage3InstructionCache(
                    apiKey = when (irBackend) {
                        is com.samsung.genuicraft.inference.GeminiBackend -> "" // key already in backend
                        else -> ""
                    },
                    model = irModel,
                    systemPrompt = promptContext.systemPrompt
                )
            }
        } else {
            null
        }

        val irApiKey = if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
            GeminiApiKeyProvider.stage3ApiKey(appContext).trim()
        } else {
            ""
        }
        val stage3Cache = if (stage3CacheDeferred != null) {
            runCatching { stage3CacheDeferred }
                .getOrElse {
                    PipelineCacheManager.CacheSetupResult(name = null, created = false, error = it.message ?: it.javaClass.simpleName)
                }
        } else {
            PipelineCacheManager.CacheSetupResult(name = null, created = false, error = null)
        }
        // For MCP path, skip cache complexity — use direct prompt
        val stage3CacheResult = PipelineCacheManager.CacheSetupResult(name = null, created = false, error = null)

        val localStage3SystemPromptCacheKey: String? = null
        val localSendStage3SystemPrompt = true

        val stage3MaxOutputTokens = if (irProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS
        } else {
            STAGE3_MAX_OUTPUT_TOKENS
        }

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            backend = irBackend,
            provider = irProvider,
            prompt = stage3Prompt,
            systemPrompt = promptContext.systemPrompt,
            temperature = 0.2,
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = null,
            allowCachedContent = false,
            structuredOutput = irProvider == InferenceBackendSettings.Provider.GEMINI
        )
        markStreamDuration(Stage.STAGE3, stage3Call.streamDurationMs)

        if (stage3Call.error != null) {
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            return Outcome.Failure(
                stage = Stage.STAGE3,
                message = stage3Call.error,
                stage2Response = sanitizedResponse,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        var stage3JsonElement = PipelineJsonExtractor.extractJsonElement(stage3Call.text)
        var usedFallback = false

        if (stage3JsonElement == null) {
            warnings += "Stage 3 JSON parse failed; running repair pass."
            val repairCall = generateWithRetry(
                backend = irBackend,
                provider = irProvider,
                prompt = PipelineJsonExtractor.buildRepairPrompt(stage3Call.text),
                systemPrompt = promptContext.systemPrompt,
                temperature = 0.2,
                maxOutputTokens = stage3MaxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = null,
                allowCachedContent = false,
                structuredOutput = irProvider == InferenceBackendSettings.Provider.GEMINI
            )
            markStreamDuration(Stage.STAGE3, repairCall.streamDurationMs)
            if (repairCall.error == null) {
                stage3JsonElement = PipelineJsonExtractor.extractJsonElement(repairCall.text)
            }
        }

        if (stage3JsonElement == null) {
            warnings += "Stage 3 fallback JSON was used."
            stage3JsonElement = PipelineMediaSanitizer.buildFallbackGenUi(sanitizedResponse, catalogId)
            usedFallback = true
        }

        val normalizedGenUi = PipelineMediaSanitizer.normalizeGenUiPayload(stage3JsonElement)
        // Restore shortened URL placeholders back to real URLs
        var stage3Json = com.samsung.genuicraft.mcp.McpUrlShortener.restore(gson.toJson(normalizedGenUi), urlMap)
        if (!usedFallback) {
            val stage3WithInjectedImage = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
                jsonText = stage3Json, stage2Response = sanitizedResponse, queryText = normalizedQuery
            )
            if (stage3WithInjectedImage != stage3Json) {
                stage3Json = stage3WithInjectedImage
                warnings += "Injected fallback image into IR output to preserve media."
            }
        }
        if (!usedFallback) {
            val stage3WithFlightMediaNormalized = PipelineMediaSanitizer.normalizeFlightMediaInGenUi(
                jsonText = stage3Json, queryText = normalizedQuery, stage2Response = sanitizedResponse
            )
            if (stage3WithFlightMediaNormalized != stage3Json) {
                stage3Json = stage3WithFlightMediaNormalized
                warnings += "Normalized flight media to airline/travel-safe icons."
            }
        }
        if (!usedFallback) {
            val stage3WithStableMediaUrls = PipelineMediaSanitizer.rewriteUnstableMediaHostsInGenUi(
                jsonText = stage3Json, queryText = normalizedQuery
            )
            if (stage3WithStableMediaUrls != stage3Json) {
                stage3Json = stage3WithStableMediaUrls
                warnings += "Rewrote unstable media URLs to deterministic topic icons."
            }
        }
        if (!usedFallback && !PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)) {
            val stage3WithInlineTextMedia = PipelineMediaSanitizer.ensureGenUiHasInlineTextMedia(
                jsonText = stage3Json, queryText = normalizedQuery
            )
            if (stage3WithInlineTextMedia != stage3Json) {
                stage3Json = stage3WithInlineTextMedia
                warnings += "Injected fallback inline media text to preserve image rendering."
            }
        }
        val stage2HasInlineImage = PipelineMediaSanitizer.hasInlineImageUrl(sanitizedResponse)
        val stage2HasInlineIcon = PipelineMediaSanitizer.hasInlineIconUrl(sanitizedResponse)
        val stage3HasInlineImage = PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = PipelineMediaSanitizer.genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if ((missingInlineImage || missingInlineIcon) && !usedFallback) {
            warnings += "Media content was adjusted for compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(sanitizedResponse, catalogId))
            usedFallback = true
        }
        if (PipelineMediaSanitizer.responseContainsActionButtons(sanitizedResponse) &&
            !PipelineMediaSanitizer.genUiPreservesActionButtons(stage3Json) && !usedFallback
        ) {
            warnings += "Quick actions were adjusted for compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(sanitizedResponse, catalogId))
            usedFallback = true
        }
        // Ensure Tags: lines are preserved as chip rows
        val hasTags = sanitizedResponse.lines().any { it.trim().startsWith("Tags:", ignoreCase = true) }
        val stage3HasChips = stage3Json.contains("\"chip\"", ignoreCase = true)
        if (hasTags && !stage3HasChips && !usedFallback) {
            warnings += "Tags were adjusted for chip rendering compatibility."
            stage3Json = gson.toJson(PipelineMediaSanitizer.buildFallbackGenUi(sanitizedResponse, catalogId))
            usedFallback = true
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        var renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

        if (renderResult.errorMessage != null && !usedFallback) {
            warnings += "Native rendering failed for stage 3 output; using fallback UI."
            val fallback = PipelineMediaSanitizer.buildFallbackGenUi(sanitizedResponse, catalogId)
            stage3Json = gson.toJson(fallback)
            renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)
            usedFallback = true
        }

        if (renderResult.errorMessage != null) {
            markDuration(Stage.STAGE4, stage4StartedAtMs)
            return Outcome.Failure(
                stage = Stage.STAGE4,
                message = renderResult.errorMessage,
                stage2Response = sanitizedResponse,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        markDuration(Stage.STAGE4, stage4StartedAtMs)

        return Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = stage2Prompt,
                stage2Response = sanitizedResponse,
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

    private fun generateWithRetry(
        backend: InferenceBackend,
        provider: InferenceBackendSettings.Provider,
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
        localSendSystemPrompt: Boolean = true,
        geminiCacheFallbackSystemPrompt: String? = null,
        onGeminiCachedContentMissing: ((String) -> Unit)? = null
    ): InferenceBackend.GenerateResponse {
        var attempt = 0
        var accumulatedStreamMs = 0L
        var hasStreamSample = false
        var last: InferenceBackend.GenerateResponse = InferenceBackend.GenerateResponse(
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
            last = backend.generate(
                InferenceBackend.GenerateRequest(
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
                    cacheManager.markLocalSystemPromptCacheKeyReady(localSystemPromptCacheKey)
                }
                return last.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            val errorClass = backend.classifyError(last.error!!)
            if (
                provider == InferenceBackendSettings.Provider.GEMINI &&
                enableGoogleSearch &&
                errorClass == InferenceBackend.ErrorClass.SEARCH_TOOL_CONFIG
            ) {
                val fallback = backend.generate(
                    InferenceBackend.GenerateRequest(
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
                errorClass == InferenceBackend.ErrorClass.STRUCTURED_OUTPUT_CONFIG
            ) {
                val fallback = backend.generate(
                    InferenceBackend.GenerateRequest(
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
                errorClass == InferenceBackend.ErrorClass.LOCAL_CACHE_MISS
            ) {
                Log.w(
                    LOG_TAG,
                    "Local stage3 KV prefix cache miss detected for key=$localSystemPromptCacheKey, retrying with inline system prompt."
                )
                val cacheRecovery = backend.generate(
                    InferenceBackend.GenerateRequest(
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
                )
                cacheRecovery.streamDurationMs?.let {
                    accumulatedStreamMs += it
                    hasStreamSample = true
                }
                if (cacheRecovery.error == null) {
                    cacheManager.markLocalSystemPromptCacheKeyReady(localSystemPromptCacheKey)
                    Log.i(LOG_TAG, "Local stage3 KV prefix cache recovery succeeded for key=$localSystemPromptCacheKey")
                } else {
                    Log.w(LOG_TAG, "Local stage3 KV prefix cache recovery failed for key=$localSystemPromptCacheKey: ${cacheRecovery.error}")
                }
                return cacheRecovery.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            if (
                provider == InferenceBackendSettings.Provider.GEMINI &&
                !effectiveCachedContentName.isNullOrBlank() &&
                errorClass == InferenceBackend.ErrorClass.CACHED_CONTENT_MISSING
            ) {
                val reason = "Gemini cached content became unavailable (${effectiveCachedContentName.take(64)})."
                onGeminiCachedContentMissing?.invoke(reason)
                Log.w(
                    LOG_TAG,
                    "Gemini cached content not found, retrying with inline system prompt and no cachedContent."
                )
                val cacheRecovery = backend.generate(
                    InferenceBackend.GenerateRequest(
                        prompt = prompt,
                        systemPrompt = geminiCacheFallbackSystemPrompt ?: systemPrompt,
                        temperature = temperature,
                        maxOutputTokens = maxOutputTokens,
                        jsonMode = jsonMode,
                        enableGoogleSearch = enableGoogleSearch,
                        cachedContentName = null,
                        structuredOutput = structuredOutput,
                        localSystemPromptCacheKey = localSystemPromptCacheKey,
                        localSendSystemPrompt = localSendSystemPrompt
                    )
                )
                cacheRecovery.streamDurationMs?.let {
                    accumulatedStreamMs += it
                    hasStreamSample = true
                }
                return cacheRecovery.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            val retryable = errorClass == InferenceBackend.ErrorClass.TRANSIENT
            if (!retryable || attempt >= 3) {
                return last.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
            }
            Thread.sleep(1000L * attempt)
        }
        return last.copy(streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null)
    }

    private companion object {
        const val MODEL_GEMINI_2_5_PRO = "gemini-2.5-pro"
        const val STAGE2_MAX_OUTPUT_TOKENS = 4096
        const val STAGE3_MAX_OUTPUT_TOKENS = 8192
        const val LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS = 2048
        const val LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS = 15000
        const val LOG_TAG = "GenUiStagePipeline"
        val gson = GsonBuilder().disableHtmlEscaping().create()
    }
}
