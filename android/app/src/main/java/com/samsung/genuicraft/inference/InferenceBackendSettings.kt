package com.samsung.genuicraft

import android.content.Context
import com.samsung.genuicraft.inference.OnDeviceModelCatalog

object InferenceBackendSettings {
    private const val PREFS_NAME = "inference_backend_settings"
    private const val SDK_DEMO_PREFS_NAME = "genuicraft_sdk_demo"
    private const val KEY_PROVIDER = "provider"
    private const val KEY_RESPONSE_PROVIDER = "response_provider"
    private const val KEY_IR_PROVIDER = "ir_provider"
    private const val KEY_GEMINI_API_MODE = "gemini_api_mode"
    private const val KEY_VERTEX_PROJECT_ID = "vertex_project_id"
    private const val KEY_VERTEX_LOCATION = "vertex_location"
    private const val KEY_VERTEX_ACCESS_TOKEN = "vertex_access_token"
    private const val KEY_DEFAULT_PROVIDER_MIGRATION = "default_provider_migration_azure_openai"
    private const val KEY_HYBRID_PROVIDER_MIGRATION = "default_provider_migration_gemini_response_azure_ir"
    private const val KEY_GPT54_RESPONSE_MIGRATION = "default_provider_migration_gpt54_response_azure_ir"
    private const val KEY_AZURE_OPENAI_RESPONSES_ENDPOINT = "azure_openai_responses_endpoint"
    private const val KEY_AZURE_OPENAI_DEPLOYMENT = "azure_openai_deployment"
    private const val KEY_LOCAL_SERVER_BASE_URL = "local_server_base_url"
    private const val KEY_LOCAL_MODEL_PATH = "local_model_path"
    private const val KEY_ON_DEVICE_MODEL_PATH = "on_device_model_path"
    private const val KEY_ON_DEVICE_ACCELERATOR = "on_device_accelerator"
    private const val KEY_ON_DEVICE_MTP_ENABLED = "e2b_mtp_enabled"
    private const val KEY_RENDER_WITHOUT_OUTER_CARD = "render_without_outer_card"
    private const val KEY_RENDER_CARD_TRANSPARENCY = "render_card_transparency"
    private const val KEY_RENDER_BACKGROUND_TRANSPARENCY = "render_background_transparency"

    const val DEFAULT_AZURE_OPENAI_RESPONSES_ENDPOINT =
        "https://genui1.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
    const val DEFAULT_AZURE_OPENAI_DEPLOYMENT = "gpt-5.4"
    const val DEFAULT_LOCAL_SERVER_BASE_URL = "http://10.0.2.2:8000"
    const val DEFAULT_LOCAL_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
    const val DEFAULT_ON_DEVICE_MODEL_PATH = ""
    val DEFAULT_ON_DEVICE_ACCELERATOR = Accelerator.AUTO
    const val DEFAULT_ON_DEVICE_MTP_ENABLED = true
    const val DEFAULT_RENDER_WITHOUT_OUTER_CARD = true
    const val DEFAULT_RENDER_CARD_TRANSPARENCY = 0.22f
    const val MIN_RENDER_CARD_TRANSPARENCY = 0.08f
    const val MAX_RENDER_CARD_TRANSPARENCY = 0.38f
    const val DEFAULT_RENDER_BACKGROUND_TRANSPARENCY = 0.24f
    const val MIN_RENDER_BACKGROUND_TRANSPARENCY = 0.00f
    const val MAX_RENDER_BACKGROUND_TRANSPARENCY = 0.45f
    private const val FALLBACK_VERTEX_PROJECT_ID = "gen-lang-client-0741138863"
    const val DEFAULT_VERTEX_LOCATION = "us-central1"

    enum class Provider(val rawValue: String) {
        AZURE_OPENAI("azure_openai"),
        GEMINI("gemini"),
        LOCAL_SERVER("local_server"),
        ON_DEVICE_LITERT("on_device_litert");

        companion object {
            fun fromRawValue(value: String?): Provider {
                val normalized = value?.trim().orEmpty()
                return entries.firstOrNull { it.rawValue.equals(normalized, ignoreCase = true) }
                    ?: AZURE_OPENAI
            }
        }
    }

    enum class Accelerator(val rawValue: String) {
        AUTO("auto"),
        GPU("gpu"),
        CPU("cpu"),
        NPU("npu");

        companion object {
            fun fromRawValue(value: String?): Accelerator {
                val normalized = value?.trim().orEmpty()
                return entries.firstOrNull { it.rawValue.equals(normalized, ignoreCase = true) }
                    ?: AUTO
            }
        }
    }

    enum class GeminiApiMode(val rawValue: String) {
        AI_STUDIO_DIRECT("ai_studio_direct"),
        VERTEX_AI_OAUTH("vertex_ai_oauth"),
        VERTEX_AI_EXPRESS_API_KEY("vertex_ai_express_api_key");

        companion object {
            fun fromRawValue(value: String?): GeminiApiMode {
                val normalized = value?.trim().orEmpty()
                if (normalized.equals("vertex_ai_express", ignoreCase = true)) {
                    // Backward compatibility with the previous mode name.
                    return VERTEX_AI_EXPRESS_API_KEY
                }
                return entries.firstOrNull { it.rawValue.equals(normalized, ignoreCase = true) }
                    ?: VERTEX_AI_EXPRESS_API_KEY
            }
        }
    }

    fun getProvider(context: Context): Provider {
        return getResponseProvider(context)
    }

    fun setProvider(context: Context, provider: Provider) {
        setResponseProvider(context, provider)
        setIrProvider(context, provider)
    }

    fun getResponseProvider(context: Context): Provider {
        migrateDefaultProviderIfNeeded(context)
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val responseRaw = prefs.getString(KEY_RESPONSE_PROVIDER, null)
        if (!responseRaw.isNullOrBlank()) {
            return Provider.fromRawValue(responseRaw)
        }
        val legacy = prefs.getString(KEY_PROVIDER, Provider.AZURE_OPENAI.rawValue)
        return Provider.fromRawValue(legacy)
    }

    fun setResponseProvider(context: Context, provider: Provider) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RESPONSE_PROVIDER, provider.rawValue)
            .putString(KEY_PROVIDER, provider.rawValue)
            .apply()
    }

    fun getIrProvider(context: Context): Provider {
        migrateDefaultProviderIfNeeded(context)
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val irRaw = prefs.getString(KEY_IR_PROVIDER, null)
        if (!irRaw.isNullOrBlank()) {
            return Provider.fromRawValue(irRaw)
        }
        val legacy = prefs.getString(KEY_PROVIDER, Provider.AZURE_OPENAI.rawValue)
        return Provider.fromRawValue(legacy)
    }

    fun setIrProvider(context: Context, provider: Provider) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_IR_PROVIDER, provider.rawValue)
            .apply()
    }

    fun getGeminiApiMode(context: Context): GeminiApiMode {
        val forcedMode = GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(
            KEY_GEMINI_API_MODE,
            forcedMode.rawValue
        )
        val resolved = GeminiApiMode.fromRawValue(raw)
        if (resolved != forcedMode) {
            prefs.edit().putString(KEY_GEMINI_API_MODE, forcedMode.rawValue).apply()
        }
        return forcedMode
    }

    fun setGeminiApiMode(context: Context, mode: GeminiApiMode) {
        val forcedMode = GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_GEMINI_API_MODE, forcedMode.rawValue)
            .apply()
    }

    fun getVertexProjectId(context: Context): String {
        val defaultProjectId = defaultVertexProjectId()
        val runtimeProjectId = GeminiApiKeyProvider.runtimeVertexProjectId(context).trim()
        if (runtimeProjectId.isNotBlank()) {
            return runtimeProjectId
        }
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_VERTEX_PROJECT_ID, defaultProjectId)
            .orEmpty()
            .trim()
            .ifBlank { defaultProjectId }
    }

    fun setVertexProjectId(context: Context, value: String) {
        val normalized = value.trim().ifBlank { defaultVertexProjectId() }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_VERTEX_PROJECT_ID, normalized)
            .apply()
    }

    fun getVertexLocation(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_VERTEX_LOCATION, DEFAULT_VERTEX_LOCATION)
            .orEmpty()
            .trim()
            .ifBlank { DEFAULT_VERTEX_LOCATION }
    }

    fun setVertexLocation(context: Context, value: String) {
        val normalized = value.trim().ifBlank { DEFAULT_VERTEX_LOCATION }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_VERTEX_LOCATION, normalized)
            .apply()
    }

    fun getVertexAccessToken(context: Context): String {
        val runtimeToken = GeminiApiKeyProvider.runtimeVertexOauthAccessToken(context).trim()
        if (runtimeToken.isNotBlank()) {
            return runtimeToken
        }
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_VERTEX_ACCESS_TOKEN, null)
            .orEmpty()
            .trim()
        if (stored.isNotBlank()) {
            return stored
        }
        return BuildConfig.VERTEX_OAUTH_ACCESS_TOKEN_DEFAULT.trim()
    }

    fun setVertexAccessToken(context: Context, value: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_VERTEX_ACCESS_TOKEN, value.trim())
            .apply()
    }

    fun getAzureOpenAiResponsesEndpoint(context: Context): String {
        val defaultEndpoint = BuildConfig.AZURE_OPENAI_RESPONSES_ENDPOINT_DEFAULT
            .trim()
            .ifBlank { DEFAULT_AZURE_OPENAI_RESPONSES_ENDPOINT }
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return normalizeHttpsUrl(
            prefs.getString(KEY_AZURE_OPENAI_RESPONSES_ENDPOINT, defaultEndpoint).orEmpty()
        ).ifBlank { defaultEndpoint }
    }

    fun setAzureOpenAiResponsesEndpoint(context: Context, value: String) {
        val normalized = normalizeHttpsUrl(value).ifBlank { DEFAULT_AZURE_OPENAI_RESPONSES_ENDPOINT }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_AZURE_OPENAI_RESPONSES_ENDPOINT, normalized)
            .apply()
    }

    fun getAzureOpenAiDeployment(context: Context): String {
        val defaultDeployment = BuildConfig.AZURE_OPENAI_DEPLOYMENT_DEFAULT
            .trim()
            .ifBlank { DEFAULT_AZURE_OPENAI_DEPLOYMENT }
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_AZURE_OPENAI_DEPLOYMENT, defaultDeployment)
            .orEmpty()
            .trim()
            .ifBlank { defaultDeployment }
    }

    fun setAzureOpenAiDeployment(context: Context, value: String) {
        val normalized = value.trim().ifBlank { DEFAULT_AZURE_OPENAI_DEPLOYMENT }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_AZURE_OPENAI_DEPLOYMENT, normalized)
            .apply()
    }

    fun getLocalServerBaseUrl(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_LOCAL_SERVER_BASE_URL, DEFAULT_LOCAL_SERVER_BASE_URL).orEmpty()
        return normalizeBaseUrl(stored).ifBlank { DEFAULT_LOCAL_SERVER_BASE_URL }
    }

    fun setLocalServerBaseUrl(context: Context, value: String) {
        val normalized = normalizeBaseUrl(value).ifBlank { DEFAULT_LOCAL_SERVER_BASE_URL }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LOCAL_SERVER_BASE_URL, normalized)
            .apply()
    }

    fun getLocalModelPath(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_LOCAL_MODEL_PATH, DEFAULT_LOCAL_MODEL_PATH)
            .orEmpty()
            .trim()
            .ifBlank { DEFAULT_LOCAL_MODEL_PATH }
    }

    fun setLocalModelPath(context: Context, value: String) {
        val normalized = value.trim().ifBlank { DEFAULT_LOCAL_MODEL_PATH }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LOCAL_MODEL_PATH, normalized)
            .apply()
    }

    fun getOnDeviceModelPath(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_ON_DEVICE_MODEL_PATH, DEFAULT_ON_DEVICE_MODEL_PATH)
            .orEmpty()
            .trim()
        val migrated = OnDeviceModelCatalog.migrateLegacySelection(context, stored)
        if (migrated != stored) {
            // Only the model path moves; response and IR provider choices remain untouched.
            prefs.edit().putString(KEY_ON_DEVICE_MODEL_PATH, migrated).apply()
        }
        return migrated
    }

    fun setOnDeviceModelPath(context: Context, value: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_ON_DEVICE_MODEL_PATH, value.trim())
            .apply()
    }

    fun getOnDeviceAccelerator(context: Context): Accelerator {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return Accelerator.fromRawValue(
            prefs.getString(KEY_ON_DEVICE_ACCELERATOR, DEFAULT_ON_DEVICE_ACCELERATOR.rawValue)
        )
    }

    fun setOnDeviceAccelerator(context: Context, accelerator: Accelerator) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_ON_DEVICE_ACCELERATOR, accelerator.rawValue)
            .apply()
    }

    /** Shares the SDK demo's existing MTP preference so both settings surfaces stay in sync. */
    fun getOnDeviceMtpEnabled(context: Context): Boolean {
        return context.getSharedPreferences(SDK_DEMO_PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_ON_DEVICE_MTP_ENABLED, DEFAULT_ON_DEVICE_MTP_ENABLED)
    }

    /** Shares the SDK demo's existing MTP preference so installed choices are preserved. */
    fun setOnDeviceMtpEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(SDK_DEMO_PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ON_DEVICE_MTP_ENABLED, enabled)
            .apply()
    }

    fun getRenderWithoutOuterCard(context: Context): Boolean {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getBoolean(KEY_RENDER_WITHOUT_OUTER_CARD, DEFAULT_RENDER_WITHOUT_OUTER_CARD)
    }

    fun setRenderWithoutOuterCard(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_RENDER_WITHOUT_OUTER_CARD, enabled)
            .apply()
    }

    fun getRenderCardTransparency(context: Context): Float {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getFloat(KEY_RENDER_CARD_TRANSPARENCY, DEFAULT_RENDER_CARD_TRANSPARENCY)
            .coerceIn(MIN_RENDER_CARD_TRANSPARENCY, MAX_RENDER_CARD_TRANSPARENCY)
    }

    fun setRenderCardTransparency(context: Context, value: Float) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putFloat(
                KEY_RENDER_CARD_TRANSPARENCY,
                value.coerceIn(MIN_RENDER_CARD_TRANSPARENCY, MAX_RENDER_CARD_TRANSPARENCY)
            )
            .apply()
    }

    fun getRenderCardOpacity(context: Context): Float {
        return (1f - getRenderCardTransparency(context)).coerceIn(0.60f, 0.96f)
    }

    fun getRenderBackgroundTransparency(context: Context): Float {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getFloat(KEY_RENDER_BACKGROUND_TRANSPARENCY, DEFAULT_RENDER_BACKGROUND_TRANSPARENCY)
            .coerceIn(MIN_RENDER_BACKGROUND_TRANSPARENCY, MAX_RENDER_BACKGROUND_TRANSPARENCY)
    }

    fun setRenderBackgroundTransparency(context: Context, value: Float) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putFloat(
                KEY_RENDER_BACKGROUND_TRANSPARENCY,
                value.coerceIn(MIN_RENDER_BACKGROUND_TRANSPARENCY, MAX_RENDER_BACKGROUND_TRANSPARENCY)
            )
            .apply()
    }

    fun getRenderBackgroundOpacity(context: Context): Float {
        return (1f - getRenderBackgroundTransparency(context)).coerceIn(0.55f, 0.94f)
    }

    private fun normalizeBaseUrl(value: String): String {
        var normalized = value.trim()
        if (normalized.isBlank()) {
            return ""
        }
        if (!normalized.startsWith("http://", ignoreCase = true) &&
            !normalized.startsWith("https://", ignoreCase = true)
        ) {
            normalized = "http://$normalized"
        }
        return normalized.trimEnd('/')
    }

    private fun normalizeHttpsUrl(value: String): String {
        var normalized = value.trim()
        if (normalized.isBlank()) {
            return ""
        }
        if (!normalized.startsWith("http://", ignoreCase = true) &&
            !normalized.startsWith("https://", ignoreCase = true)
        ) {
            normalized = "https://$normalized"
        }
        return normalized
    }

    private fun defaultVertexProjectId(): String {
        return BuildConfig.VERTEX_PROJECT_ID_DEFAULT.trim().ifBlank { FALLBACK_VERTEX_PROJECT_ID }
    }

    private fun migrateDefaultProviderIfNeeded(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        if (prefs.getBoolean(KEY_DEFAULT_PROVIDER_MIGRATION, false)) {
            migrateHybridDefaultsIfNeeded(context)
            migrateGpt54ResponseDefaultsIfNeeded(context)
            return
        }

        val legacyRaw = prefs.getString(KEY_PROVIDER, null)
        val legacyProvider = Provider.fromRawValue(legacyRaw)
        val shouldMigrateLegacy = legacyRaw.isNullOrBlank() || legacyProvider == Provider.GEMINI
        val responseRaw = prefs.getString(KEY_RESPONSE_PROVIDER, null)
        val irRaw = prefs.getString(KEY_IR_PROVIDER, null)

        val editor = prefs.edit()
        if (shouldMigrateLegacy && responseRaw.isNullOrBlank()) {
            editor.putString(KEY_RESPONSE_PROVIDER, Provider.GEMINI.rawValue)
        }
        if (shouldMigrateLegacy && irRaw.isNullOrBlank()) {
            editor.putString(KEY_IR_PROVIDER, Provider.AZURE_OPENAI.rawValue)
        }
        if (shouldMigrateLegacy && legacyRaw.isNullOrBlank()) {
            editor.putString(KEY_PROVIDER, Provider.GEMINI.rawValue)
        }
        editor.putBoolean(KEY_DEFAULT_PROVIDER_MIGRATION, true).apply()

        migrateHybridDefaultsIfNeeded(context)
        migrateGpt54ResponseDefaultsIfNeeded(context)
    }

    private fun migrateHybridDefaultsIfNeeded(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        if (prefs.getBoolean(KEY_HYBRID_PROVIDER_MIGRATION, false)) {
            return
        }

        val responseRaw = prefs.getString(KEY_RESPONSE_PROVIDER, null)
        val irRaw = prefs.getString(KEY_IR_PROVIDER, null)
        val responseProvider = Provider.fromRawValue(responseRaw)
        val irProvider = Provider.fromRawValue(irRaw)
        val shouldApplyHybridDefault = responseRaw.isNullOrBlank() ||
            irRaw.isNullOrBlank() ||
            (
                responseProvider != Provider.LOCAL_SERVER &&
                    irProvider != Provider.LOCAL_SERVER &&
                    (responseProvider != Provider.GEMINI || irProvider != Provider.AZURE_OPENAI)
                )

        val editor = prefs.edit()
        if (shouldApplyHybridDefault) {
            editor
                .putString(KEY_RESPONSE_PROVIDER, Provider.GEMINI.rawValue)
                .putString(KEY_IR_PROVIDER, Provider.AZURE_OPENAI.rawValue)
                .putString(KEY_PROVIDER, Provider.GEMINI.rawValue)
        }
        editor.putBoolean(KEY_HYBRID_PROVIDER_MIGRATION, true).apply()

        val modelPrefs = context.getSharedPreferences("gemini_model_settings", Context.MODE_PRIVATE)
        val responseModel = modelPrefs.getString("selected_response_model", null).orEmpty()
        val legacyModel = modelPrefs.getString("selected_model", null).orEmpty()
        if (responseModel.isBlank() || responseModel == "gemini-2.5-flash" || legacyModel == "gemini-2.5-flash") {
            modelPrefs.edit()
                .putString("selected_response_model", GeminiModelSettings.DEFAULT_RESPONSE_MODEL)
                .apply()
        }
    }

    private fun migrateGpt54ResponseDefaultsIfNeeded(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        if (prefs.getBoolean(KEY_GPT54_RESPONSE_MIGRATION, false)) {
            return
        }

        val responseProvider = Provider.fromRawValue(prefs.getString(KEY_RESPONSE_PROVIDER, null))
        val irProvider = Provider.fromRawValue(prefs.getString(KEY_IR_PROVIDER, null))
        val shouldApplyAzureDefault =
            responseProvider != Provider.LOCAL_SERVER &&
                irProvider != Provider.LOCAL_SERVER

        val currentDeployment = prefs.getString(KEY_AZURE_OPENAI_DEPLOYMENT, null)
            .orEmpty()
            .trim()
        val shouldUpdateDeployment = currentDeployment.isBlank() ||
            currentDeployment.equals("gpt-5.4-mini", ignoreCase = true)

        val editor = prefs.edit()
        if (shouldApplyAzureDefault) {
            editor
                .putString(KEY_RESPONSE_PROVIDER, Provider.AZURE_OPENAI.rawValue)
                .putString(KEY_IR_PROVIDER, Provider.AZURE_OPENAI.rawValue)
                .putString(KEY_PROVIDER, Provider.AZURE_OPENAI.rawValue)
        }
        if (shouldUpdateDeployment) {
            editor.putString(KEY_AZURE_OPENAI_DEPLOYMENT, DEFAULT_AZURE_OPENAI_DEPLOYMENT)
        }
        editor.putBoolean(KEY_GPT54_RESPONSE_MIGRATION, true).apply()
    }
}
