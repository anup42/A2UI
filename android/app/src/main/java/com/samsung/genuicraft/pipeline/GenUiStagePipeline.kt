package com.samsung.genuicraft

import android.content.Context
import android.util.Log
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.InferenceBackendFactory
import com.samsung.genuicraft.inference.LocalServerBackend
import com.samsung.genuicraft.mcp.McpClient
import com.samsung.genuicraft.mcp.McpLlmRouter
import com.samsung.genuicraft.mcp.McpResponseFormatter
import com.samsung.genuicraft.mcp.McpSettings
import com.samsung.genuicraft.pipeline.PipelineCacheManager
import com.samsung.genuicraft.pipeline.FlatSpecContract
import com.samsung.genuicraft.pipeline.PipelineImageResolver
import com.samsung.genuicraft.pipeline.PipelineJsonExtractor
import com.samsung.genuicraft.pipeline.PipelineMediaSanitizer
import com.samsung.genuicraft.pipeline.PipelinePromptBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.withContext
import java.io.File

class GenUiStagePipeline(private val appContext: Context) {
    enum class Stage {
        STAGE2,
        STAGE3,
        STAGE4
    }

    data class StageUpdate(
        val stage: Stage,
        val message: String,
        val debugLog: String? = null,
        val stage2Response: String? = null,
        val stage3Json: String? = null,
        val renderResult: GenUiNativeRenderer.RenderResult? = null,
        val llmInputTokens: Int? = null,
        val llmOutputTokens: Int? = null
    )

    data class PipelineResult(
        val queryText: String,
        val stage2Prompt: String,
        val stage2Response: String,
        val stage3Prompt: String,
        val stage3SystemPrompt: String?,
        val stage3Json: String,
        val stage3InputTokens: Int?,
        val stage3OutputTokens: Int?,
        val stageDurationsMs: Map<Stage, Long>,
        val stageStreamDurationsMs: Map<Stage, Long>,
        val usedFallback: Boolean,
        val warnings: List<String>,
        val renderResult: GenUiNativeRenderer.RenderResult
    )

    private data class Stage3RepairDiagnostics(
        val rawStage3Text: String,
        val selectedJsonCandidateText: String?,
        var initialValidationError: String? = null,
        val repairValidationErrors: MutableList<String> = mutableListOf(),
        val repairOutputTexts: MutableList<String> = mutableListOf(),
        val repairSelectedCandidateTexts: MutableList<String?> = mutableListOf(),
        var tableDiagnostics: FlatSpecContract.TableDiagnostics? = null
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

    private fun loadStage3PromptTemplate(provider: InferenceBackendSettings.Provider): String {
        val assetPath = if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            PipelinePromptBuilder.STAGE3_GEMMA_PROMPT_ASSET
        } else {
            PipelinePromptBuilder.STAGE3_PROMPT_ASSET
        }
        return PipelinePromptBuilder.loadPromptAsset(appContext.assets, assetPath)
    }

    private fun stage3MaxOutputTokensFor(
        provider: InferenceBackendSettings.Provider,
        model: String = ""
    ): Int {
        return when (provider) {
            InferenceBackendSettings.Provider.LOCAL_SERVER -> LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS
            InferenceBackendSettings.Provider.ON_DEVICE_LITERT -> ON_DEVICE_STAGE3_MAX_OUTPUT_TOKENS
            InferenceBackendSettings.Provider.GEMINI ->
                if (model.trim().lowercase().startsWith("gemma-")) {
                    GEMMA_STAGE3_MAX_OUTPUT_TOKENS
                } else {
                    STAGE3_MAX_OUTPUT_TOKENS
                }
            else -> STAGE3_MAX_OUTPUT_TOKENS
        }
    }

    private fun stage3TemperatureFor(provider: InferenceBackendSettings.Provider): Double {
        return if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            0.0
        } else {
            0.2
        }
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

        GeminiApiKeyProvider.refresh(appContext)

        val responseProvider = InferenceBackendSettings.getResponseProvider(appContext)
        val irProvider = InferenceBackendSettings.getIrProvider(appContext)
        val geminiApiMode = InferenceBackendSettings.getGeminiApiMode(appContext)
        val vertexProjectId = InferenceBackendSettings.getVertexProjectId(appContext)
        val vertexLocation = InferenceBackendSettings.getVertexLocation(appContext)
        val vertexAccessToken = InferenceBackendSettings.getVertexAccessToken(appContext)
        val vertexExpressApiKey = GeminiApiKeyProvider.vertexExpressApiKey(appContext).trim()
        val azureOpenAiApiKey = GeminiApiKeyProvider.azureOpenAiApiKey(appContext).trim()
        val azureOpenAiResponsesEndpoint = InferenceBackendSettings.getAzureOpenAiResponsesEndpoint(appContext)
        val azureOpenAiDeployment = InferenceBackendSettings.getAzureOpenAiDeployment(appContext)
        val responseModel = if (responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
            azureOpenAiDeployment
        } else {
            GeminiModelSettings.getResponseModel(appContext)
        }
        val irModel = if (irProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
            azureOpenAiDeployment
        } else {
            GeminiModelSettings.getIrModel(appContext)
        }
        val localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(appContext)
        val localModelPath = InferenceBackendSettings.getLocalModelPath(appContext)
        val onDeviceModelPath = InferenceBackendSettings.getOnDeviceModelPath(appContext)
        val stage2MaxOutputTokens = if (responseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER) {
            LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS
        } else {
            STAGE2_MAX_OUTPUT_TOKENS
        }
        val stage3MaxOutputTokens = stage3MaxOutputTokensFor(irProvider, irModel)
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
        if (responseProvider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT &&
            responseApiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE2,
                message = "Gemini key is missing. Stage 2 now uses Stage 3 key; add GEMINI_STAGE3_API_KEY (or GEMINI_IR_API_KEY / GEMINI_API_KEY_2) at ${GeminiApiKeyProvider.setupHintPath(appContext)}"
            )
        }
        if (irProvider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT &&
            irApiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Gemini stage-3 key is missing. Add GEMINI_IR_API_KEY (or GEMINI_API_KEY_2) at ${GeminiApiKeyProvider.setupHintPath(appContext)}"
            )
        }
        if ((responseProvider == InferenceBackendSettings.Provider.GEMINI ||
                irProvider == InferenceBackendSettings.Provider.GEMINI) &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.VERTEX_AI_OAUTH &&
            vertexProjectId.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = if (responseProvider == InferenceBackendSettings.Provider.GEMINI) Stage.STAGE2 else Stage.STAGE3,
                message = "Vertex project id is missing. Open Settings and set Vertex project id (e.g. gen-lang-client-0741138863)."
            )
        }
        if ((responseProvider == InferenceBackendSettings.Provider.GEMINI ||
                irProvider == InferenceBackendSettings.Provider.GEMINI) &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.VERTEX_AI_OAUTH &&
            vertexAccessToken.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = if (responseProvider == InferenceBackendSettings.Provider.GEMINI) Stage.STAGE2 else Stage.STAGE3,
                message = "Vertex OAuth access token is missing. Open Settings and paste a valid OAuth token."
            )
        }
        if ((responseProvider == InferenceBackendSettings.Provider.GEMINI ||
                irProvider == InferenceBackendSettings.Provider.GEMINI) &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY &&
            vertexExpressApiKey.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = if (responseProvider == InferenceBackendSettings.Provider.GEMINI) Stage.STAGE2 else Stage.STAGE3,
                message = "Vertex Express API key is missing. Add VERTEX_EXPRESS_API_KEY at ${GeminiApiKeyProvider.setupHintPath(appContext)}"
            )
        }
        if ((responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI ||
                irProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) &&
            azureOpenAiApiKey.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = if (responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) Stage.STAGE2 else Stage.STAGE3,
                message = "Azure OpenAI key is missing. Add AZURE_OPENAI_API_KEY (or AZURE_OPENAI_SUBSCRIPTION_KEY) at ${GeminiApiKeyProvider.setupHintPath(appContext)}"
            )
        }
        if (irProvider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT && onDeviceModelPath.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "On-device IR model path is missing. Open Settings and select an exported model package."
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
        if (irProvider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            postUpdate(onStageUpdate, Stage.STAGE3, "Checking on-device IR model")
            val healthResult = com.samsung.genuicraft.inference.OnDeviceLitertBackend(onDeviceModelPath).checkHealth()
            if (!healthResult.healthy) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = healthResult.errorMessage ?: "On-device IR model is not ready."
                )
            }
        }

        val responseBackend = InferenceBackendFactory.create(
            provider = responseProvider,
            apiKey = responseApiKey,
            model = responseModel,
            geminiApiMode = geminiApiMode,
            vertexProjectId = vertexProjectId,
            vertexLocation = vertexLocation,
            vertexAccessToken = vertexAccessToken,
            vertexExpressApiKey = vertexExpressApiKey,
            azureOpenAiApiKey = azureOpenAiApiKey,
            azureOpenAiResponsesEndpoint = azureOpenAiResponsesEndpoint,
            azureOpenAiDeployment = azureOpenAiDeployment,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath,
            onDeviceModelPath = onDeviceModelPath
        )
        val irBackend = InferenceBackendFactory.create(
            provider = irProvider,
            apiKey = irApiKey,
            model = irModel,
            geminiApiMode = geminiApiMode,
            vertexProjectId = vertexProjectId,
            vertexLocation = vertexLocation,
            vertexAccessToken = vertexAccessToken,
            vertexExpressApiKey = vertexExpressApiKey,
            azureOpenAiApiKey = azureOpenAiApiKey,
            azureOpenAiResponsesEndpoint = azureOpenAiResponsesEndpoint,
            azureOpenAiDeployment = azureOpenAiDeployment,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath,
            onDeviceModelPath = onDeviceModelPath
        )

        // -- MCP path: LLM routes query -> optional live data fetch ----------
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
                    // Router call failed entirely - fall through to normal Stage 2
                    Log.w(LOG_TAG, "MCP router failed: ${routerResult.error}; falling back to normal Stage 2")
                    postUpdate(
                        onStageUpdate,
                        Stage.STAGE2,
                        "MCP router fallback to standard response generation",
                        debugLog = "MCP router error: ${routerResult.error}"
                    )
                    markDuration(Stage.STAGE2, stage2StartedAtMs)
                }

                routerResult.domain != null && McpSettings.isDomainReady(appContext, routerResult.domain) -> {
                    // LLM identified a live-data domain -> fetch MCP data
                    Log.i(LOG_TAG, "MCP router: domain=${routerResult.domain.key} entities=${routerResult.entities}")
                    postUpdate(
                        onStageUpdate,
                        Stage.STAGE2,
                        "MCP domain selected: ${routerResult.domain.displayName}",
                        debugLog = "MCP entities (${routerResult.domain.key}): ${formatDebugEntities(routerResult.entities)}"
                    )
                    postUpdate(onStageUpdate, Stage.STAGE2, "Fetching live ${routerResult.domain.displayName} data")
                    val mcpApiKey = McpSettings.getApiKey(appContext, routerResult.domain)
                    val mcpResult = McpClient.fetch(
                        domain = routerResult.domain,
                        entities = routerResult.entities,
                        apiKey = mcpApiKey,
                        queryText = normalizedQuery
                    )
                    mcpResult.requestDebug?.let { requestDebug ->
                        postUpdate(
                            onStageUpdate,
                            Stage.STAGE2,
                            "MCP API call completed",
                            debugLog = "MCP API call (${routerResult.domain.key}): $requestDebug"
                        )
                    }
                    if (!mcpResult.success && !mcpResult.error.isNullOrBlank()) {
                        postUpdate(
                            onStageUpdate,
                            Stage.STAGE2,
                            "MCP API returned an error",
                            debugLog = "MCP API error (${routerResult.domain.key}): ${mcpResult.error}"
                        )
                    }
                    val mcpDataSection = McpResponseFormatter.buildDataSection(mcpResult, normalizedQuery)
                    if (mcpDataSection.isBlank()) {
                        // MCP returned empty data - fall through to normal Stage 2 LLM
                        Log.w(LOG_TAG, "MCP ${routerResult.domain.key} returned no data; falling back to Stage 2")
                        postUpdate(
                            onStageUpdate,
                            Stage.STAGE2,
                            "MCP data empty, using standard Stage 2 response",
                            debugLog = "MCP ${routerResult.domain.key}: data section empty after API call."
                        )
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
                        if (responseProvider == InferenceBackendSettings.Provider.GEMINI ||
                            irProvider == InferenceBackendSettings.Provider.GEMINI) {
                            mcpWarnings += "Gemini route: ${geminiApiMode.rawValue}"
                        }
                        if (responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI ||
                            irProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
                            mcpWarnings += "Azure OpenAI deployment: $azureOpenAiDeployment"
                        }
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
                    postUpdate(
                        onStageUpdate,
                        Stage.STAGE2,
                        "MCP routing skipped live API call",
                        debugLog = "MCP entities (none route): ${formatDebugEntities(routerResult.entities)}"
                    )
                    markDuration(Stage.STAGE2, stage2StartedAtMs)
                    val mcpWarnings = mutableListOf<String>()
                    mcpWarnings += "MCP: LLM routing decided no live data needed - using LLM response"
                    mcpWarnings += "Response backend: ${responseProvider.rawValue}"
                    mcpWarnings += "IR backend: ${irProvider.rawValue}"
                    if (responseProvider == InferenceBackendSettings.Provider.GEMINI ||
                        irProvider == InferenceBackendSettings.Provider.GEMINI) {
                        mcpWarnings += "Gemini route: ${geminiApiMode.rawValue}"
                    }
                    if (responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI ||
                        irProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
                        mcpWarnings += "Azure OpenAI deployment: $azureOpenAiDeployment"
                    }
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
                    // Router returned domain but API key not ready - fall through
                    val domainName = routerResult.domain?.key ?: "unknown"
                    Log.w(LOG_TAG, "MCP domain $domainName identified but API key not configured; falling back to normal Stage 2")
                    postUpdate(
                        onStageUpdate,
                        Stage.STAGE2,
                        "MCP key missing, using standard Stage 2 response",
                        debugLog = "MCP domain=$domainName entities=${formatDebugEntities(routerResult.entities)}"
                    )
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
        val stage2CacheDeferred = if (responseProvider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT) {
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
            loadStage3PromptTemplate(irProvider)
        }
            .getOrElse {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = "Could not load stage 3 prompt: ${it.message ?: it.javaClass.simpleName}"
                )
            }
        val promptContext = PipelinePromptBuilder.prepareStage3PromptContext(genUiTemplate)
        val stage3CacheDeferred = if (irProvider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT) {
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
            enableGoogleSearch = responseProvider == InferenceBackendSettings.Provider.GEMINI ||
                responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI,
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
        val stage2Response = McpResponseFormatter.normalizeForStage3(
            PipelineMediaSanitizer.normalizeUrlTokensForDisplay(stage2WithGeneralMedia)
        )
        val injectedFlightList = stage2WithFlightList != stage2ResponseRaw
        val normalizedBareDomains = stage2Response != stage2WithGeneralMedia
        val removedFlightMedia = stage2WithFlightMedia != stage2WithActions
        val sanitizedTravelMedia = stage2WithTravelMediaSanitized != stage2WithFlightMedia
        val injectedTravelMedia = stage2WithTravelMedia != stage2WithTravelMediaSanitized
        markDuration(Stage.STAGE2, stage2StartedAtMs)
        postUpdate(
            onStageUpdate,
            Stage.STAGE2,
            "Rich response ready",
            stage2Response = stage2Response
        )

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
        if (responseProvider == InferenceBackendSettings.Provider.GEMINI ||
            irProvider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Gemini route: ${geminiApiMode.rawValue}"
        }
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
        } else if (responseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
            warnings += "Azure OpenAI response deployment: $azureOpenAiDeployment"
            warnings += "Azure OpenAI endpoint: $azureOpenAiResponsesEndpoint"
        } else {
            warnings += "Local server (response): $localServerBaseUrl"
            warnings += "Local model path (response): $localModelPath"
            warnings += "Local token cap (response): stage2=$stage2MaxOutputTokens"
        }
        if (irProvider == InferenceBackendSettings.Provider.GEMINI) {
            warnings += "Gemini IR model: $irModel"
        } else if (irProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
            warnings += "Azure OpenAI IR deployment: $azureOpenAiDeployment"
            warnings += "Azure OpenAI endpoint: $azureOpenAiResponsesEndpoint"
        } else if (irProvider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            warnings += "On-device Gemma IR model: $onDeviceModelPath"
            warnings += "On-device Gemma prompt: ${PipelinePromptBuilder.STAGE3_GEMMA_PROMPT_ASSET}"
            warnings += "On-device token cap (IR): stage3=$stage3MaxOutputTokens repairAttempts=$ON_DEVICE_STAGE3_REPAIR_ATTEMPTS"
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
        val stage3StructuredOutput = shouldUseStructuredOutput(irProvider, geminiApiMode)
        if (irProvider == InferenceBackendSettings.Provider.GEMINI && !stage3StructuredOutput) {
            warnings += "Stage 3 structured schema disabled for Vertex Express (prevents empty-elements outputs)."
        }

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            backend = irBackend,
            provider = irProvider,
            prompt = stage3Prompt,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            temperature = stage3TemperatureFor(irProvider),
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            structuredOutput = stage3StructuredOutput,
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

        val stage3InitialCandidate = PipelineJsonExtractor.extractJsonElement(stage3Call.text)
        val stage3Diagnostics = Stage3RepairDiagnostics(
            rawStage3Text = stage3Call.text,
            selectedJsonCandidateText = stage3InitialCandidate?.toString()
        )
        val stage3JsonElement = repairAndValidateFlatSpec(
            stage3RawText = stage3Call.text,
            initialJsonElement = stage3InitialCandidate,
            backend = irBackend,
            provider = irProvider,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            stage3RepairMaxOutputTokens = stage3RepairMaxOutputTokens,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
            localSendSystemPrompt = localSendStage3SystemPrompt,
            geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
            onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage3InstructionCache(reason) },
            onMarkStreamDuration = { markStreamDuration(Stage.STAGE3, it) },
            structuredOutput = stage3StructuredOutput,
            warnings = warnings,
            diagnostics = stage3Diagnostics
        ) ?: run {
            val strictDebug = buildStrictStage3FailureDebugLog(stage3Diagnostics)
            persistStage3DiagnosticsArtifacts(stage3Diagnostics)
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            postUpdate(
                onStageUpdate,
                Stage.STAGE3,
                "Stage 3 strict validation failed; no safe IR rendered",
                debugLog = strictDebug,
                stage2Response = stage2Response
            )
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Stage 3 output failed strict validation after repair. No fallback IR was rendered.",
                stage2Response = stage2Response,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        val normalizedGenUi = PipelineMediaSanitizer.normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        // Preserve real Stage 2 image media when Stage 3 drops it.
        val stage3WithFlightMediaNormalized = PipelineMediaSanitizer.normalizeFlightMediaInGenUi(
            jsonText = stage3Json,
            queryText = normalizedQuery,
            stage2Response = stage2Response
        )
        if (stage3WithFlightMediaNormalized != stage3Json) {
            stage3Json = stage3WithFlightMediaNormalized
            warnings += "Normalized flight media to airline/travel-safe icons."
        }
        val stage3WithStableMediaUrls = PipelineMediaSanitizer.rewriteUnstableMediaHostsInGenUi(
            jsonText = stage3Json,
            queryText = normalizedQuery
        )
        if (stage3WithStableMediaUrls != stage3Json) {
            stage3Json = stage3WithStableMediaUrls
            warnings += "Rewrote unstable media URLs to deterministic topic icons."
        }
        val stage2HasInlineImage = PipelineMediaSanitizer.hasInlineImageUrl(stage2Response)
        if (stage2HasInlineImage &&
            !PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        ) {
            val stage3WithImageComponent = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
                jsonText = stage3Json,
                stage2Response = stage2Response,
                queryText = normalizedQuery
            )
            if (stage3WithImageComponent != stage3Json) {
                stage3Json = stage3WithImageComponent
                warnings += "Injected fallback image component to preserve response media."
            }
        }
        val imageRepairResult = PipelineImageResolver.repairImageUrls(
            jsonText = stage3Json,
            queryText = normalizedQuery,
            stage2Response = stage2Response
        )
        if (imageRepairResult.jsonText != stage3Json) {
            stage3Json = imageRepairResult.jsonText
            warnings += "Resolved ${imageRepairResult.resolvedCount} generated image URL(s); replaced ${imageRepairResult.replacedCount} unreachable image URL(s)."
        }
        val finalSafetyResult = enforceFinalStage3Safety(stage3Json, warnings)
        if (finalSafetyResult.error != null || finalSafetyResult.jsonText == null) {
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            postUpdate(
                onStageUpdate,
                Stage.STAGE3,
                "Stage 3 failed final safety validation",
                debugLog = finalSafetyResult.error,
                stage2Response = stage2Response,
                stage3Json = stage3Json
            )
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Stage 3 output failed final safety validation: ${finalSafetyResult.error}",
                stage2Response = stage2Response,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        stage3Json = finalSafetyResult.jsonText
        val stage2HasInlineIcon = PipelineMediaSanitizer.hasInlineIconUrl(stage2Response)
        val stage3HasInlineImage = PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = PipelineMediaSanitizer.genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if (missingInlineIcon) {
            warnings += "Stage 3 did not preserve inline icon content."
        } else if (missingInlineImage) {
            warnings += "Stage 3 did not preserve inline image; layout retained."
        }
        if (PipelineMediaSanitizer.responseContainsActionButtons(stage2Response) &&
            !PipelineMediaSanitizer.genUiPreservesActionButtons(stage3Json)
        ) {
            warnings += "Stage 3 output omitted one or more quick actions."
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)
        postUpdate(
            onStageUpdate,
            Stage.STAGE3,
            "GenUI JSON ready",
            debugLog = buildTableDiagnosticsDebugLog(stage3Diagnostics.tableDiagnostics),
            stage3Json = stage3Json,
            llmInputTokens = stage3Call.inputTokens,
            llmOutputTokens = stage3Call.outputTokens
        )

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        val renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

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
        postUpdate(
            onStageUpdate,
            Stage.STAGE4,
            "Native render ready",
            renderResult = renderResult
        )

        return@withContext Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = stage2Prompt,
                stage2Response = stage2Response,
                stage3Prompt = stage3Prompt,
                stage3SystemPrompt = promptContext.systemPrompt,
                stage3Json = stage3Json,
                stage3InputTokens = stage3Call.inputTokens,
                stage3OutputTokens = stage3Call.outputTokens,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap(),
                usedFallback = false,
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
        val geminiApiMode = InferenceBackendSettings.getGeminiApiMode(appContext)
        val vertexProjectId = InferenceBackendSettings.getVertexProjectId(appContext)
        val vertexLocation = InferenceBackendSettings.getVertexLocation(appContext)
        val vertexAccessToken = InferenceBackendSettings.getVertexAccessToken(appContext)
        val vertexExpressApiKey = GeminiApiKeyProvider.vertexExpressApiKey(appContext).trim()
        val azureOpenAiApiKey = GeminiApiKeyProvider.azureOpenAiApiKey(appContext).trim()
        val azureOpenAiResponsesEndpoint = InferenceBackendSettings.getAzureOpenAiResponsesEndpoint(appContext)
        val azureOpenAiDeployment = InferenceBackendSettings.getAzureOpenAiDeployment(appContext)
        val irModel = if (provider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
            azureOpenAiDeployment
        } else {
            GeminiModelSettings.getIrModel(appContext)
        }
        val localServerBaseUrl = InferenceBackendSettings.getLocalServerBaseUrl(appContext)
        val localModelPath = InferenceBackendSettings.getLocalModelPath(appContext)
        val onDeviceModelPath = InferenceBackendSettings.getOnDeviceModelPath(appContext)
        val isLocalServer = provider == InferenceBackendSettings.Provider.LOCAL_SERVER
        val stage3MaxOutputTokens = stage3MaxOutputTokensFor(provider, irModel)
        val stage3RepairMaxOutputTokens = stage3MaxOutputTokens

        val apiKey = if (provider == InferenceBackendSettings.Provider.GEMINI) {
            GeminiApiKeyProvider.stage3ApiKey(appContext).trim()
        } else {
            ""
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT &&
            apiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Gemini stage-3 key is missing. Add GEMINI_IR_API_KEY (or GEMINI_API_KEY_2) at ${GeminiApiKeyProvider.setupHintPath(appContext)}",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.VERTEX_AI_OAUTH &&
            vertexProjectId.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Vertex project id is missing. Open Settings and set Vertex project id (e.g. gen-lang-client-0741138863).",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.VERTEX_AI_OAUTH &&
            vertexAccessToken.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Vertex OAuth access token is missing. Open Settings and paste a valid OAuth token.",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY &&
            vertexExpressApiKey.isBlank()
        ) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Vertex Express API key is missing. Add VERTEX_EXPRESS_API_KEY at ${GeminiApiKeyProvider.setupHintPath(appContext)}",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.AZURE_OPENAI && azureOpenAiApiKey.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Azure OpenAI key is missing. Add AZURE_OPENAI_API_KEY (or AZURE_OPENAI_SUBSCRIPTION_KEY) at ${GeminiApiKeyProvider.setupHintPath(appContext)}",
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT && onDeviceModelPath.isBlank()) {
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "On-device IR model path is missing. Open Settings and select an exported model package.",
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
        if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            postUpdate(onStageUpdate, Stage.STAGE3, "Checking on-device IR model")
            val healthResult = com.samsung.genuicraft.inference.OnDeviceLitertBackend(onDeviceModelPath).checkHealth()
            if (!healthResult.healthy) {
                return@withContext Outcome.Failure(
                    stage = Stage.STAGE3,
                    message = healthResult.errorMessage ?: "On-device IR model is not ready.",
                    stageDurationsMs = stageDurationsMs.toMap(),
                    stageStreamDurationsMs = stageStreamDurationsMs.toMap()
                )
            }
        }

        val backend = InferenceBackendFactory.create(
            provider = provider,
            apiKey = apiKey,
            model = irModel,
            geminiApiMode = geminiApiMode,
            vertexProjectId = vertexProjectId,
            vertexLocation = vertexLocation,
            vertexAccessToken = vertexAccessToken,
            vertexExpressApiKey = vertexExpressApiKey,
            azureOpenAiApiKey = azureOpenAiApiKey,
            azureOpenAiResponsesEndpoint = azureOpenAiResponsesEndpoint,
            azureOpenAiDeployment = azureOpenAiDeployment,
            localServerBaseUrl = localServerBaseUrl,
            localModelPath = localModelPath,
            onDeviceModelPath = onDeviceModelPath
        )

        val genUiTemplate = runCatching {
            loadStage3PromptTemplate(provider)
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
        val stage3CacheDeferred = if (provider == InferenceBackendSettings.Provider.GEMINI &&
            geminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT) {
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
        val stage2Response = McpResponseFormatter.normalizeForStage3(
            PipelineMediaSanitizer.normalizeUrlTokensForDisplay(stage2WithGeneralMedia)
        )
        val injectedFlightList = stage2WithFlightList != normalizedResponseRaw
        val normalizedBareDomains = stage2Response != stage2WithGeneralMedia
        val removedFlightMedia = stage2WithFlightMedia != stage2WithActions
        val sanitizedTravelMedia = stage2WithTravelMediaSanitized != stage2WithFlightMedia
        val injectedTravelMedia = stage2WithTravelMedia != stage2WithTravelMediaSanitized
        postUpdate(
            onStageUpdate,
            Stage.STAGE2,
            "Rich response ready",
            stage2Response = stage2Response
        )

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
            warnings += "Gemini route: ${geminiApiMode.rawValue}"
            warnings += "Gemini IR model: $irModel"
        } else if (provider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
            warnings += "Azure OpenAI IR deployment: $azureOpenAiDeployment"
            warnings += "Azure OpenAI endpoint: $azureOpenAiResponsesEndpoint"
        } else if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            warnings += "On-device Gemma IR model: $onDeviceModelPath"
            warnings += "On-device Gemma prompt: ${PipelinePromptBuilder.STAGE3_GEMMA_PROMPT_ASSET}"
            warnings += "On-device token cap: stage3=$stage3MaxOutputTokens repairAttempts=$ON_DEVICE_STAGE3_REPAIR_ATTEMPTS"
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
        val stage3StructuredOutput = shouldUseStructuredOutput(provider, geminiApiMode)
        if (provider == InferenceBackendSettings.Provider.GEMINI && !stage3StructuredOutput) {
            warnings += "Stage 3 structured schema disabled for Vertex Express (prevents empty-elements outputs)."
        }

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            backend = backend,
            provider = provider,
            prompt = stage3Prompt,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            temperature = stage3TemperatureFor(provider),
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            structuredOutput = stage3StructuredOutput,
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

        val stage3InitialCandidate = PipelineJsonExtractor.extractJsonElement(stage3Call.text)
        val stage3Diagnostics = Stage3RepairDiagnostics(
            rawStage3Text = stage3Call.text,
            selectedJsonCandidateText = stage3InitialCandidate?.toString()
        )
        val stage3JsonElement = repairAndValidateFlatSpec(
            stage3RawText = stage3Call.text,
            initialJsonElement = stage3InitialCandidate,
            backend = backend,
            provider = provider,
            systemPrompt = if (stage3Cache.name != null) null else promptContext.systemPrompt,
            stage3RepairMaxOutputTokens = stage3RepairMaxOutputTokens,
            cachedContentName = stage3Cache.name,
            allowCachedContent = true,
            localSystemPromptCacheKey = localStage3SystemPromptCacheKey,
            localSendSystemPrompt = localSendStage3SystemPrompt,
            geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
            onGeminiCachedContentMissing = { reason -> cacheManager.invalidateStage3InstructionCache(reason) },
            onMarkStreamDuration = { markStreamDuration(Stage.STAGE3, it) },
            structuredOutput = stage3StructuredOutput,
            warnings = warnings,
            diagnostics = stage3Diagnostics
        ) ?: run {
            val strictDebug = buildStrictStage3FailureDebugLog(stage3Diagnostics)
            persistStage3DiagnosticsArtifacts(stage3Diagnostics)
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            postUpdate(
                onStageUpdate,
                Stage.STAGE3,
                "Stage 3 strict validation failed; no safe IR rendered",
                debugLog = strictDebug,
                stage2Response = stage2Response
            )
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Stage 3 output failed strict validation after repair. No fallback IR was rendered.",
                stage2Response = stage2Response,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        val normalizedGenUi = PipelineMediaSanitizer.normalizeGenUiPayload(stage3JsonElement)
        var stage3Json = gson.toJson(normalizedGenUi)
        // Preserve real Stage 2 image media when Stage 3 drops it.
        val stage3WithFlightMediaNormalized = PipelineMediaSanitizer.normalizeFlightMediaInGenUi(
            jsonText = stage3Json,
            queryText = normalizedQuery,
            stage2Response = stage2Response
        )
        if (stage3WithFlightMediaNormalized != stage3Json) {
            stage3Json = stage3WithFlightMediaNormalized
            warnings += "Normalized flight media to airline/travel-safe icons."
        }
        val stage3WithStableMediaUrls = PipelineMediaSanitizer.rewriteUnstableMediaHostsInGenUi(
            jsonText = stage3Json,
            queryText = normalizedQuery
        )
        if (stage3WithStableMediaUrls != stage3Json) {
            stage3Json = stage3WithStableMediaUrls
            warnings += "Rewrote unstable media URLs to deterministic topic icons."
        }
        val stage2HasInlineImage = PipelineMediaSanitizer.hasInlineImageUrl(stage2Response)
        if (stage2HasInlineImage &&
            !PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        ) {
            val stage3WithImageComponent = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
                jsonText = stage3Json,
                stage2Response = stage2Response,
                queryText = normalizedQuery
            )
            if (stage3WithImageComponent != stage3Json) {
                stage3Json = stage3WithImageComponent
                warnings += "Injected fallback image component to preserve response media."
            }
        }
        val imageRepairResult = PipelineImageResolver.repairImageUrls(
            jsonText = stage3Json,
            queryText = normalizedQuery,
            stage2Response = stage2Response
        )
        if (imageRepairResult.jsonText != stage3Json) {
            stage3Json = imageRepairResult.jsonText
            warnings += "Resolved ${imageRepairResult.resolvedCount} generated image URL(s); replaced ${imageRepairResult.replacedCount} unreachable image URL(s)."
        }
        val finalSafetyResult = enforceFinalStage3Safety(stage3Json, warnings)
        if (finalSafetyResult.error != null || finalSafetyResult.jsonText == null) {
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            postUpdate(
                onStageUpdate,
                Stage.STAGE3,
                "Stage 3 failed final safety validation",
                debugLog = finalSafetyResult.error,
                stage2Response = stage2Response,
                stage3Json = stage3Json
            )
            return@withContext Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Stage 3 output failed final safety validation: ${finalSafetyResult.error}",
                stage2Response = stage2Response,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        stage3Json = finalSafetyResult.jsonText
        val stage2HasInlineIcon = PipelineMediaSanitizer.hasInlineIconUrl(stage2Response)
        val stage3HasInlineImage = PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = PipelineMediaSanitizer.genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if (missingInlineIcon) {
            warnings += "Stage 3 did not preserve inline icon content."
        } else if (missingInlineImage) {
            warnings += "Stage 3 did not preserve inline image; layout retained."
        }
        if (PipelineMediaSanitizer.responseContainsActionButtons(stage2Response) &&
            !PipelineMediaSanitizer.genUiPreservesActionButtons(stage3Json)
        ) {
            warnings += "Stage 3 output omitted one or more quick actions."
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)
        postUpdate(
            onStageUpdate,
            Stage.STAGE3,
            "GenUI JSON ready",
            debugLog = buildTableDiagnosticsDebugLog(stage3Diagnostics.tableDiagnostics),
            stage3Json = stage3Json,
            llmInputTokens = stage3Call.inputTokens,
            llmOutputTokens = stage3Call.outputTokens
        )

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        val renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

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
        postUpdate(
            onStageUpdate,
            Stage.STAGE4,
            "Native render ready",
            renderResult = renderResult
        )

        return@withContext Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = "IR demo preloaded response (stage 2 skipped).",
                stage2Response = stage2Response,
                stage3Prompt = stage3Prompt,
                stage3SystemPrompt = promptContext.systemPrompt,
                stage3Json = stage3Json,
                stage3InputTokens = stage3Call.inputTokens,
                stage3OutputTokens = stage3Call.outputTokens,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap(),
                usedFallback = false,
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
        val isMcpLiveRoute = !stage2Prompt.startsWith("[MCP-routed:none]", ignoreCase = true)
        val stage2WithGeneralMedia = if (isMcpLiveRoute) {
            // MCP live-data responses already carry domain-specific structure/media.
            // Do not inject synthetic generic media lines that create duplicate top images.
            stage2WithTravelMedia
        } else {
            PipelineMediaSanitizer.ensureGeneralInlineMedia(
                responseText = stage2WithTravelMedia,
                queryText = normalizedQuery
            )
        }
        val sanitizedResponse = McpResponseFormatter.normalizeForStage3(
            PipelineMediaSanitizer.normalizeUrlTokensForDisplay(stage2WithGeneralMedia)
        )
        postUpdate(
            onStageUpdate,
            Stage.STAGE2,
            "Rich response ready",
            stage2Response = sanitizedResponse
        )

        // Shorten long URLs to compact tokens before sending to Stage 3 LLM
        val urlShortenResult = com.samsung.genuicraft.mcp.McpUrlShortener.shorten(sanitizedResponse)
        val stage3InputResponse = urlShortenResult.shortenedText
        val urlMap = urlShortenResult.urlMap

        val warnings = extraWarnings.toMutableList()

        val catalogId = PipelineMediaSanitizer.resolveStage3CatalogId(
            appContext.getSharedPreferences(PipelineMediaSanitizer.APP_PREFS_NAME, android.content.Context.MODE_PRIVATE)
        )

        val genUiTemplate = runCatching {
            loadStage3PromptTemplate(irProvider)
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
        // For MCP path, skip cache complexity - use direct prompt
        val stage3CacheResult = PipelineCacheManager.CacheSetupResult(name = null, created = false, error = null)

        val localStage3SystemPromptCacheKey: String? = null
        val localSendStage3SystemPrompt = true

        val stage3MaxOutputTokens = stage3MaxOutputTokensFor(irProvider, irModel)
        val stage3StructuredOutput = shouldUseStructuredOutput(
            irProvider,
            InferenceBackendSettings.getGeminiApiMode(appContext)
        )
        if (irProvider == InferenceBackendSettings.Provider.GEMINI && !stage3StructuredOutput) {
            warnings += "Stage 3 structured schema disabled for Vertex Express (prevents empty-elements outputs)."
        }

        postUpdate(onStageUpdate, Stage.STAGE3, "Converting response into GenUICraft IR JSON")
        val stage3StartedAtMs = System.currentTimeMillis()
        val stage3Call = generateWithRetry(
            backend = irBackend,
            provider = irProvider,
            prompt = stage3Prompt,
            systemPrompt = promptContext.systemPrompt,
            temperature = stage3TemperatureFor(irProvider),
            maxOutputTokens = stage3MaxOutputTokens,
            jsonMode = true,
            enableGoogleSearch = false,
            cachedContentName = null,
            allowCachedContent = false,
            structuredOutput = stage3StructuredOutput
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

        val stage3InitialCandidate = PipelineJsonExtractor.extractJsonElement(stage3Call.text)
        val stage3Diagnostics = Stage3RepairDiagnostics(
            rawStage3Text = stage3Call.text,
            selectedJsonCandidateText = stage3InitialCandidate?.toString()
        )
        val stage3JsonElement = repairAndValidateFlatSpec(
            stage3RawText = stage3Call.text,
            initialJsonElement = stage3InitialCandidate,
            backend = irBackend,
            provider = irProvider,
            systemPrompt = promptContext.systemPrompt,
            stage3RepairMaxOutputTokens = stage3MaxOutputTokens,
            cachedContentName = null,
            allowCachedContent = false,
            localSystemPromptCacheKey = null,
            localSendSystemPrompt = true,
            geminiCacheFallbackSystemPrompt = promptContext.systemPrompt,
            onGeminiCachedContentMissing = null,
            onMarkStreamDuration = { markStreamDuration(Stage.STAGE3, it) },
            structuredOutput = stage3StructuredOutput,
            warnings = warnings,
            diagnostics = stage3Diagnostics
        ) ?: run {
            val strictDebug = buildStrictStage3FailureDebugLog(stage3Diagnostics)
            persistStage3DiagnosticsArtifacts(stage3Diagnostics)
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            postUpdate(
                onStageUpdate,
                Stage.STAGE3,
                "Stage 3 strict validation failed; no safe IR rendered",
                debugLog = strictDebug,
                stage2Response = sanitizedResponse
            )
            return Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Stage 3 output failed strict validation after repair. No fallback IR was rendered.",
                stage2Response = sanitizedResponse,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }

        val normalizedGenUi = PipelineMediaSanitizer.normalizeGenUiPayload(stage3JsonElement)
        // Restore shortened URL placeholders back to real URLs
        var stage3Json = com.samsung.genuicraft.mcp.McpUrlShortener.restore(gson.toJson(normalizedGenUi), urlMap)
        // Preserve real Stage 2 image media when Stage 3 drops it.
        val stage3WithFlightMediaNormalized = PipelineMediaSanitizer.normalizeFlightMediaInGenUi(
            jsonText = stage3Json, queryText = normalizedQuery, stage2Response = sanitizedResponse
        )
        if (stage3WithFlightMediaNormalized != stage3Json) {
            stage3Json = stage3WithFlightMediaNormalized
            warnings += "Normalized flight media to airline/travel-safe icons."
        }
        val stage3WithRestaurantPhotos = PipelineMediaSanitizer.preserveRestaurantPhotosInGenUi(
            jsonText = stage3Json,
            stage2Response = sanitizedResponse
        )
        if (stage3WithRestaurantPhotos != stage3Json) {
            stage3Json = stage3WithRestaurantPhotos
            warnings += "Preserved restaurant photos/contact data from Google Places rows."
        }
        val stage3WithStableMediaUrls = PipelineMediaSanitizer.rewriteUnstableMediaHostsInGenUi(
            jsonText = stage3Json, queryText = normalizedQuery
        )
        if (stage3WithStableMediaUrls != stage3Json) {
            stage3Json = stage3WithStableMediaUrls
            warnings += "Rewrote unstable media URLs to deterministic topic icons."
        }
        val stage2HasInlineImage = PipelineMediaSanitizer.hasInlineImageUrl(sanitizedResponse)
        if (stage2HasInlineImage &&
            !PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        ) {
            val stage3WithImageComponent = PipelineMediaSanitizer.ensureGenUiHasImageComponent(
                jsonText = stage3Json,
                stage2Response = sanitizedResponse,
                queryText = normalizedQuery
            )
            if (stage3WithImageComponent != stage3Json) {
                stage3Json = stage3WithImageComponent
                warnings += "Injected fallback image component to preserve response media."
            }
        }
        val imageRepairResult = PipelineImageResolver.repairImageUrls(
            jsonText = stage3Json,
            queryText = normalizedQuery,
            stage2Response = sanitizedResponse
        )
        if (imageRepairResult.jsonText != stage3Json) {
            stage3Json = imageRepairResult.jsonText
            warnings += "Resolved ${imageRepairResult.resolvedCount} generated image URL(s); replaced ${imageRepairResult.replacedCount} unreachable image URL(s)."
        }
        val finalSafetyResult = enforceFinalStage3Safety(stage3Json, warnings)
        if (finalSafetyResult.error != null || finalSafetyResult.jsonText == null) {
            markDuration(Stage.STAGE3, stage3StartedAtMs)
            postUpdate(
                onStageUpdate,
                Stage.STAGE3,
                "Stage 3 failed final safety validation",
                debugLog = finalSafetyResult.error,
                stage2Response = sanitizedResponse,
                stage3Json = stage3Json
            )
            return Outcome.Failure(
                stage = Stage.STAGE3,
                message = "Stage 3 output failed final safety validation: ${finalSafetyResult.error}",
                stage2Response = sanitizedResponse,
                stage3Json = stage3Json,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap()
            )
        }
        stage3Json = finalSafetyResult.jsonText
        val stage2HasInlineIcon = PipelineMediaSanitizer.hasInlineIconUrl(sanitizedResponse)
        val stage3HasInlineImage = PipelineMediaSanitizer.genUiPreservesInlineImages(stage3Json)
        val stage3HasInlineIcon = PipelineMediaSanitizer.genUiPreservesInlineIcons(stage3Json)
        val missingInlineImage = stage2HasInlineImage && !stage3HasInlineImage
        val missingInlineIcon = stage2HasInlineIcon && !stage3HasInlineIcon
        if (missingInlineIcon) {
            warnings += "Stage 3 did not preserve inline icon content."
        } else if (missingInlineImage) {
            warnings += "Stage 3 did not preserve inline image; layout retained."
        }
        if (PipelineMediaSanitizer.responseContainsActionButtons(sanitizedResponse) &&
            !PipelineMediaSanitizer.genUiPreservesActionButtons(stage3Json)
        ) {
            warnings += "Stage 3 output omitted one or more quick actions."
        }
        // Ensure Tags: lines are preserved as chip rows
        val hasTags = sanitizedResponse.lines().any { it.trim().startsWith("Tags:", ignoreCase = true) }
        val stage3HasChips = stage3Json.contains("\"chip\"", ignoreCase = true)
        if (hasTags && !stage3HasChips) {
            warnings += "Stage 3 output omitted expected chip-style tags."
        }
        markDuration(Stage.STAGE3, stage3StartedAtMs)
        postUpdate(
            onStageUpdate,
            Stage.STAGE3,
            "GenUI JSON ready",
            debugLog = buildTableDiagnosticsDebugLog(stage3Diagnostics.tableDiagnostics),
            stage3Json = stage3Json,
            llmInputTokens = stage3Call.inputTokens,
            llmOutputTokens = stage3Call.outputTokens
        )

        postUpdate(onStageUpdate, Stage.STAGE4, "Rendering output")
        val stage4StartedAtMs = System.currentTimeMillis()
        val renderResult = GenUiNativeRenderer.render(stage3Json, sourceDir = null)

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
        postUpdate(
            onStageUpdate,
            Stage.STAGE4,
            "Native render ready",
            renderResult = renderResult
        )

        return Outcome.Success(
            result = PipelineResult(
                queryText = normalizedQuery,
                stage2Prompt = stage2Prompt,
                stage2Response = sanitizedResponse,
                stage3Prompt = stage3Prompt,
                stage3SystemPrompt = promptContext.systemPrompt,
                stage3Json = stage3Json,
                stage3InputTokens = stage3Call.inputTokens,
                stage3OutputTokens = stage3Call.outputTokens,
                stageDurationsMs = stageDurationsMs.toMap(),
                stageStreamDurationsMs = stageStreamDurationsMs.toMap(),
                usedFallback = false,
                warnings = warnings,
                renderResult = renderResult
            )
        )
    }

    private suspend fun repairAndValidateFlatSpec(
        stage3RawText: String,
        initialJsonElement: JsonElement?,
        backend: InferenceBackend,
        provider: InferenceBackendSettings.Provider,
        systemPrompt: String?,
        stage3RepairMaxOutputTokens: Int,
        cachedContentName: String?,
        allowCachedContent: Boolean,
        localSystemPromptCacheKey: String?,
        localSendSystemPrompt: Boolean,
        geminiCacheFallbackSystemPrompt: String?,
        onGeminiCachedContentMissing: ((String) -> Unit)?,
        onMarkStreamDuration: (Long?) -> Unit,
        structuredOutput: Boolean,
        warnings: MutableList<String>,
        diagnostics: Stage3RepairDiagnostics
    ): JsonElement? {
        val initialCoerce = FlatSpecContract.coerceAndValidate(initialJsonElement)
        if (initialCoerce.warnings.isNotEmpty()) {
            warnings += initialCoerce.warnings
        }
        diagnostics.tableDiagnostics = initialCoerce.tableDiagnostics
        if (initialCoerce.isValid) {
            if (initialCoerce.convertedFromLegacy) {
                warnings += "Stage 3 returned legacy format; converted to flat spec."
            }
            return initialCoerce.spec
        }

        val initialReason = flatSpecValidationFailureReason(
            jsonElement = initialJsonElement,
            coerceError = initialCoerce.error,
            parseFailureReason = "Stage 3 JSON parse failed."
        )
        diagnostics.initialValidationError = initialReason
        warnings += "$initialReason Running strict flat-spec repair."

        var rawForRepair = stage3RawText
        var reasonForRepair = initialReason
        val repairAttempts = if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
            ON_DEVICE_STAGE3_REPAIR_ATTEMPTS
        } else {
            3
        }

        for (attempt in 1..repairAttempts) {
            val escalatedReason = if (attempt == 1) {
                reasonForRepair
            } else {
                "$reasonForRepair (strict attempt $attempt/$repairAttempts: enforce non-empty elements and valid root reference)."
            }
            val repairCall = generateWithRetry(
                backend = backend,
                provider = provider,
                prompt = PipelineJsonExtractor.buildFlatSpecRepairPrompt(
                    rawText = rawForRepair,
                    failureReason = escalatedReason,
                    mode = if (provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
                        PipelineJsonExtractor.FlatSpecRepairMode.ON_DEVICE_LITERT
                    } else {
                        PipelineJsonExtractor.FlatSpecRepairMode.GENERAL
                    }
                ),
                systemPrompt = systemPrompt,
                temperature = stage3TemperatureFor(provider),
                maxOutputTokens = stage3RepairMaxOutputTokens,
                jsonMode = true,
                enableGoogleSearch = false,
                cachedContentName = cachedContentName,
                allowCachedContent = allowCachedContent,
                structuredOutput = structuredOutput,
                localSystemPromptCacheKey = localSystemPromptCacheKey,
                localSendSystemPrompt = localSendSystemPrompt,
                geminiCacheFallbackSystemPrompt = geminiCacheFallbackSystemPrompt,
                onGeminiCachedContentMissing = onGeminiCachedContentMissing
            )
            onMarkStreamDuration(repairCall.streamDurationMs)
            if (repairCall.error != null) {
                val error = "Repair attempt $attempt failed with backend error: ${repairCall.error}"
                diagnostics.repairValidationErrors += error
                warnings += error
                continue
            }

            diagnostics.repairOutputTexts += repairCall.text
            val repairedElement = PipelineJsonExtractor.extractJsonElement(repairCall.text)
            diagnostics.repairSelectedCandidateTexts += repairedElement?.toString()
            val repairedCoerce = FlatSpecContract.coerceAndValidate(repairedElement)
            if (repairedCoerce.warnings.isNotEmpty()) {
                warnings += repairedCoerce.warnings
            }
            diagnostics.tableDiagnostics = repairedCoerce.tableDiagnostics
            if (repairedCoerce.isValid) {
                if (repairedCoerce.convertedFromLegacy) {
                    warnings += "Stage 3 repair returned legacy format; converted to flat spec."
                }
                return repairedCoerce.spec
            }

            val repairedReason = flatSpecValidationFailureReason(
                jsonElement = repairedElement,
                coerceError = repairedCoerce.error,
                parseFailureReason = "Repair attempt $attempt produced unparseable JSON."
            )
            diagnostics.repairValidationErrors += "Attempt $attempt: $repairedReason"
            warnings += "Stage 3 repair attempt $attempt invalid ($repairedReason)."
            reasonForRepair = repairedReason
            rawForRepair = repairCall.text
        }

        warnings += "Stage 3 repaired output is still invalid after $repairAttempts attempts."
        return null
    }

    private fun flatSpecValidationFailureReason(
        jsonElement: JsonElement?,
        coerceError: String?,
        parseFailureReason: String
    ): String {
        if (jsonElement == null) {
            return parseFailureReason
        }
        if (!FlatSpecContract.looksLikeFlatSpec(jsonElement)) {
            return "Stage 3 output did not contain a valid flat-spec object with root/elements."
        }
        return coerceError ?: "Stage 3 flat-spec validation failed."
    }

    private fun shouldUseStructuredOutput(
        provider: InferenceBackendSettings.Provider,
        geminiApiMode: InferenceBackendSettings.GeminiApiMode
    ): Boolean {
        if (provider != InferenceBackendSettings.Provider.GEMINI) {
            return false
        }
        return geminiApiMode != InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY
    }

    private fun buildTableDiagnosticsDebugLog(
        diagnostics: FlatSpecContract.TableDiagnostics?
    ): String? {
        val table = diagnostics ?: return null
        if (!table.tableDetected &&
            table.canonicalizationRewrites.isEmpty() &&
            table.irElementCount == 0 &&
            table.irByteSize == 0
        ) {
            return null
        }
        val rewrites = if (table.canonicalizationRewrites.isEmpty()) {
            "none"
        } else {
            table.canonicalizationRewrites.joinToString(separator = "; ")
        }
        val mappingWarnings = if (table.mappingWarnings.isEmpty()) {
            "none"
        } else {
            table.mappingWarnings.joinToString(separator = "; ")
        }
        return buildString {
            appendLine("Stage 3 table diagnostics")
            appendLine("table_detected: ${table.tableDetected}")
            appendLine("columns: ${table.columns}")
            appendLine("rows: ${table.rows}")
            appendLine("table_domain: ${table.tableDomain}")
            appendLine("preferred_presentation: ${table.preferredPresentation}")
            appendLine("presentation_chosen: ${table.presentationChosen}")
            appendLine("render_mode: ${table.renderMode}")
            appendLine("card_mapping_status: ${table.cardMappingStatus}")
            appendLine("mapping_warnings: $mappingWarnings")
            appendLine("ir_element_count: ${table.irElementCount}")
            appendLine("ir_byte_size: ${table.irByteSize}")
            appendLine("compaction_applied: ${table.compactionApplied}")
            appendLine("removed_field_count: ${table.removedFieldCount}")
            append("canonicalization_rewrites: $rewrites")
        }
    }

    private data class FinalStage3SafetyResult(
        val jsonText: String?,
        val error: String?
    )

    private fun enforceFinalStage3Safety(
        stage3Json: String,
        warnings: MutableList<String>
    ): FinalStage3SafetyResult {
        var candidate = stage3Json
        val safeResult = PipelineMediaSanitizer.enforceSafeGenUiContent(candidate)
        if (safeResult.changed) {
            candidate = safeResult.jsonText
            warnings += "Removed unsafe GenUI content: media=${safeResult.removedMediaCount}, actions=${safeResult.removedActionCount}."
        }

        val parsed = runCatching { JsonParser.parseString(candidate) }.getOrElse { error ->
            return FinalStage3SafetyResult(
                jsonText = null,
                error = "Stage 3 safe IR is not valid JSON: ${error.message.orEmpty()}"
            )
        }
        val validation = FlatSpecContract.coerceAndValidate(parsed)
        if (!validation.isValid || validation.spec == null) {
            return FinalStage3SafetyResult(
                jsonText = null,
                error = validation.error ?: "Stage 3 safe IR failed strict validation."
            )
        }
        return FinalStage3SafetyResult(
            jsonText = gson.toJson(validation.spec),
            error = null
        )
    }

    private fun buildStrictStage3FailureDebugLog(diagnostics: Stage3RepairDiagnostics): String {
        val initial = diagnostics.initialValidationError
            ?.takeIf { it.isNotBlank() }
            ?: "Unknown Stage 3 validation error."
        val repairErrors = if (diagnostics.repairValidationErrors.isEmpty()) {
            "none"
        } else {
            diagnostics.repairValidationErrors.joinToString("\n- ", prefix = "- ")
        }
        val selectedSnippet = truncateSnippet(diagnostics.selectedJsonCandidateText, 1200)
        val rawSnippet = truncateSnippet(diagnostics.rawStage3Text, 1600)
        val repairSnippet = diagnostics.repairOutputTexts.lastOrNull()?.let { truncateSnippet(it, 1600) } ?: "n/a"
        val tableDebug = buildTableDiagnosticsDebugLog(diagnostics.tableDiagnostics)
        return buildString {
            appendLine("Stage 3 strict failure diagnostics")
            appendLine("Initial validation error: $initial")
            appendLine("Repair errors:")
            appendLine(repairErrors)
            if (!tableDebug.isNullOrBlank()) {
                appendLine()
                appendLine(tableDebug)
            }
            appendLine()
            appendLine("Selected JSON candidate (truncated):")
            appendLine(selectedSnippet)
            appendLine()
            appendLine("Stage 3 raw text (truncated):")
            appendLine(rawSnippet)
            appendLine()
            appendLine("Last repair output (truncated):")
            appendLine(repairSnippet)
        }.trim()
    }

    private fun persistStage3DiagnosticsArtifacts(diagnostics: Stage3RepairDiagnostics) {
        if (!BuildConfig.DEBUG) {
            return
        }
        runCatching {
            val root = File(appContext.filesDir, "result/stage3_debug")
            if (!root.exists()) {
                root.mkdirs()
            }
            val runDir = File(root, "run_${System.currentTimeMillis()}")
            runDir.mkdirs()

            File(runDir, "stage3_raw.txt").writeText(diagnostics.rawStage3Text)
            diagnostics.selectedJsonCandidateText?.let {
                File(runDir, "stage3_selected_candidate.json").writeText(it)
            }
            diagnostics.repairOutputTexts.forEachIndexed { index, text ->
                File(runDir, "stage3_repair_attempt_${index + 1}.txt").writeText(text)
            }
            diagnostics.repairSelectedCandidateTexts.forEachIndexed { index, text ->
                if (!text.isNullOrBlank()) {
                    File(runDir, "stage3_repair_candidate_${index + 1}.json").writeText(text)
                }
            }
            File(runDir, "stage3_diagnostics.txt").writeText(buildStrictStage3FailureDebugLog(diagnostics))
        }.onFailure { error ->
            Log.w(LOG_TAG, "Failed to persist Stage 3 diagnostics artifacts: ${error.message}")
        }
    }

    private fun truncateSnippet(text: String?, maxChars: Int): String {
        val normalized = text?.trim().orEmpty()
        if (normalized.isBlank()) {
            return "n/a"
        }
        return if (normalized.length <= maxChars) {
            normalized
        } else {
            normalized.take(maxChars) + "...(truncated)"
        }
    }

    private suspend fun postUpdate(
        callback: (StageUpdate) -> Unit,
        stage: Stage,
        message: String,
        debugLog: String? = null,
        stage2Response: String? = null,
        stage3Json: String? = null,
        renderResult: GenUiNativeRenderer.RenderResult? = null,
        llmInputTokens: Int? = null,
        llmOutputTokens: Int? = null
    ) {
        withContext(Dispatchers.Main) {
            callback(
                StageUpdate(
                    stage = stage,
                    message = message,
                    debugLog = debugLog,
                    stage2Response = stage2Response,
                    stage3Json = stage3Json,
                    renderResult = renderResult,
                    llmInputTokens = llmInputTokens,
                    llmOutputTokens = llmOutputTokens
                )
            )
        }
    }

    private fun formatDebugEntities(entities: Map<String, String>): String {
        if (entities.isEmpty()) {
            return "none"
        }
        return entities
            .toSortedMap()
            .entries
            .joinToString(", ") { (key, value) -> "$key=${value.trim()}" }
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
        var accumulatedInputTokens = 0L
        var hasInputTokenSample = false
        var accumulatedOutputTokens = 0L
        var hasOutputTokenSample = false
        var last: InferenceBackend.GenerateResponse = InferenceBackend.GenerateResponse(
            text = "",
            rawResponse = null,
            error = "Unknown generation error",
            streamDurationMs = null,
            inputTokens = null,
            outputTokens = null
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
            last.inputTokens?.let {
                accumulatedInputTokens += it.toLong()
                hasInputTokenSample = true
            }
            last.outputTokens?.let {
                accumulatedOutputTokens += it.toLong()
                hasOutputTokenSample = true
            }
            if (last.error == null) {
                if (
                    provider == InferenceBackendSettings.Provider.LOCAL_SERVER &&
                    localSendSystemPrompt &&
                    !localSystemPromptCacheKey.isNullOrBlank()
                ) {
                    cacheManager.markLocalSystemPromptCacheKeyReady(localSystemPromptCacheKey)
                }
                return last.copy(
                    streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
                    inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
                    outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
                )
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
                fallback.inputTokens?.let {
                    accumulatedInputTokens += it.toLong()
                    hasInputTokenSample = true
                }
                fallback.outputTokens?.let {
                    accumulatedOutputTokens += it.toLong()
                    hasOutputTokenSample = true
                }
                return fallback.copy(
                    streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
                    inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
                    outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
                )
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
                fallback.inputTokens?.let {
                    accumulatedInputTokens += it.toLong()
                    hasInputTokenSample = true
                }
                fallback.outputTokens?.let {
                    accumulatedOutputTokens += it.toLong()
                    hasOutputTokenSample = true
                }
                return fallback.copy(
                    streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
                    inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
                    outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
                )
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
                cacheRecovery.inputTokens?.let {
                    accumulatedInputTokens += it.toLong()
                    hasInputTokenSample = true
                }
                cacheRecovery.outputTokens?.let {
                    accumulatedOutputTokens += it.toLong()
                    hasOutputTokenSample = true
                }
                if (cacheRecovery.error == null) {
                    cacheManager.markLocalSystemPromptCacheKeyReady(localSystemPromptCacheKey)
                    Log.i(LOG_TAG, "Local stage3 KV prefix cache recovery succeeded for key=$localSystemPromptCacheKey")
                } else {
                    Log.w(LOG_TAG, "Local stage3 KV prefix cache recovery failed for key=$localSystemPromptCacheKey: ${cacheRecovery.error}")
                }
                return cacheRecovery.copy(
                    streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
                    inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
                    outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
                )
            }
            if (
                provider == InferenceBackendSettings.Provider.GEMINI &&
                !effectiveCachedContentName.isNullOrBlank() &&
                (
                    errorClass == InferenceBackend.ErrorClass.CACHED_CONTENT_MISSING ||
                        (
                            last.error!!.contains("HTTP 403", ignoreCase = true) &&
                                (
                                    last.error!!.contains("cached", ignoreCase = true) ||
                                        last.error!!.contains("PERMISSION_DENIED", ignoreCase = true)
                                    )
                            )
                    )
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
                cacheRecovery.inputTokens?.let {
                    accumulatedInputTokens += it.toLong()
                    hasInputTokenSample = true
                }
                cacheRecovery.outputTokens?.let {
                    accumulatedOutputTokens += it.toLong()
                    hasOutputTokenSample = true
                }
                return cacheRecovery.copy(
                    streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
                    inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
                    outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
                )
            }
            val retryable = errorClass == InferenceBackend.ErrorClass.TRANSIENT
            if (!retryable || attempt >= 3) {
                return last.copy(
                    streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
                    inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
                    outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
                )
            }
            Log.w(
                LOG_TAG,
                "Transient backend error (attempt $attempt/3). Retrying immediately. " +
                    "error=${last.error.orEmpty().take(180)}"
            )
        }
        return last.copy(
            streamDurationMs = if (hasStreamSample) accumulatedStreamMs else null,
            inputTokens = if (hasInputTokenSample) accumulatedInputTokens.toInt() else null,
            outputTokens = if (hasOutputTokenSample) accumulatedOutputTokens.toInt() else null
        )
    }

    private companion object {
        const val MODEL_GEMINI_2_5_PRO = "gemini-2.5-pro"
        const val STAGE2_MAX_OUTPUT_TOKENS = 4096
        const val STAGE3_MAX_OUTPUT_TOKENS = 8192
        const val GEMMA_STAGE3_MAX_OUTPUT_TOKENS = 4096
        const val ON_DEVICE_STAGE3_MAX_OUTPUT_TOKENS = 3072
        const val ON_DEVICE_STAGE3_REPAIR_ATTEMPTS = 1
        const val LOCAL_SERVER_STAGE2_MAX_OUTPUT_TOKENS = 2048
        const val LOCAL_SERVER_STAGE3_MAX_OUTPUT_TOKENS = 15000
        const val LOG_TAG = "GenUiStagePipeline"
        val gson = GsonBuilder().disableHtmlEscaping().create()
    }
}


