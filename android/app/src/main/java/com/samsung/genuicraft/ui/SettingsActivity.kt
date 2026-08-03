package com.samsung.genuicraft

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.TextButton
import com.samsung.genuicraft.mcp.McpSettings
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.inference.OnDeviceModelDownloader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

class SettingsActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        setContent {
            GenUiCraftTheme {
                SettingsScreen(
                    onBack = { finish() }
                )
            }
        }
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun SettingsScreen(
    onBack: () -> Unit
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 14.dp
        DeviceSizeClass.Medium -> 20.dp
        DeviceSizeClass.Expanded -> 26.dp
    }

    var selectedResponseProvider by remember {
        mutableStateOf(InferenceBackendSettings.getResponseProvider(context))
    }
    var selectedIrProvider by remember {
        mutableStateOf(InferenceBackendSettings.getIrProvider(context))
    }
    var selectedResponseModel by remember {
        mutableStateOf(GeminiModelSettings.getResponseModel(context))
    }
    var selectedIrModel by remember {
        mutableStateOf(GeminiModelSettings.getIrModel(context))
    }
    var selectedGeminiApiMode by remember {
        mutableStateOf(InferenceBackendSettings.getGeminiApiMode(context))
    }
    var azureOpenAiResponsesEndpoint by remember {
        mutableStateOf(InferenceBackendSettings.getAzureOpenAiResponsesEndpoint(context))
    }
    var azureOpenAiDeployment by remember {
        mutableStateOf(InferenceBackendSettings.getAzureOpenAiDeployment(context))
    }
    var localServerBaseUrl by remember {
        mutableStateOf(InferenceBackendSettings.getLocalServerBaseUrl(context))
    }
    var localModelPath by remember {
        mutableStateOf(InferenceBackendSettings.getLocalModelPath(context))
    }
    var onDeviceModelPath by remember {
        mutableStateOf(InferenceBackendSettings.getOnDeviceModelPath(context))
    }
    var onDeviceAccelerator by remember {
        mutableStateOf(InferenceBackendSettings.getOnDeviceAccelerator(context))
    }
    var onDeviceModelRefreshKey by remember { mutableIntStateOf(0) }
    var onDeviceDownloadError by remember { mutableStateOf<String?>(null) }
    val onDeviceDownloadProgress = remember { mutableStateMapOf<String, Float?>() }
    val onDeviceDownloading = remember { mutableStateMapOf<String, Boolean>() }
    var availableModels by remember {
        mutableStateOf(defaultModelOptions(selectedResponseModel, selectedIrModel))
    }
    var loading by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }
    var renderWithoutOuterCard by remember {
        mutableStateOf(InferenceBackendSettings.getRenderWithoutOuterCard(context))
    }
    var renderCardTransparency by remember {
        mutableStateOf(InferenceBackendSettings.getRenderCardTransparency(context))
    }
    var renderBackgroundTransparency by remember {
        mutableStateOf(InferenceBackendSettings.getRenderBackgroundTransparency(context))
    }

    var mcpEnabled by remember {
        mutableStateOf(McpSettings.isEnabled(context))
    }

    val mcpApiKeys = remember {
        val map = androidx.compose.runtime.mutableStateMapOf<McpSettings.Domain, String>()
        McpSettings.Domain.entries.forEach { domain ->
            map[domain] = McpSettings.getUserApiKey(context, domain)
        }
        map
    }

    fun usesGeminiBackend(): Boolean {
        return selectedResponseProvider == InferenceBackendSettings.Provider.GEMINI ||
            selectedIrProvider == InferenceBackendSettings.Provider.GEMINI
    }

    fun usesAzureOpenAiBackend(): Boolean {
        return selectedResponseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI ||
            selectedIrProvider == InferenceBackendSettings.Provider.AZURE_OPENAI
    }

    fun usesLocalBackend(): Boolean {
        return selectedResponseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER ||
            selectedIrProvider == InferenceBackendSettings.Provider.LOCAL_SERVER
    }

    fun usesOnDeviceBackend(): Boolean {
        return selectedIrProvider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT
    }

    fun refreshModels() {
        GeminiApiKeyProvider.refresh(context)
        if (!usesGeminiBackend()) {
            loading = false
            errorText = null
            availableModels = defaultModelOptions(selectedResponseModel, selectedIrModel)
            return
        }
        if (selectedGeminiApiMode != InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT) {
            loading = false
            errorText = context.getString(R.string.settings_model_vertex_catalog_note)
            availableModels = defaultModelOptions(selectedResponseModel, selectedIrModel)
            return
        }
        if (loading) return
        val apiKey = GeminiApiKeyProvider.modelCatalogApiKey(context).trim()
        if (apiKey.isBlank()) {
            errorText = context.getString(R.string.settings_model_api_key_missing)
            availableModels = defaultModelOptions(selectedResponseModel, selectedIrModel)
            return
        }

        coroutineScope.launch {
            loading = true
            val result = withContext(Dispatchers.IO) {
                GeminiModelCatalog.fetchAvailableModels(apiKey)
            }
            if (result.isSuccess) {
                val models = result.getOrNull().orEmpty()
                availableModels = mergeSelectedModels(models, selectedResponseModel, selectedIrModel)
                errorText = null
            } else {
                errorText = result.exceptionOrNull()?.message ?: context.getString(R.string.settings_model_fetch_failed)
                availableModels = defaultModelOptions(selectedResponseModel, selectedIrModel)
            }
            loading = false
        }
    }

    LaunchedEffect(selectedResponseProvider, selectedIrProvider, selectedGeminiApiMode) {
        refreshModels()
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = stringResource(id = R.string.settings_title),
                        style = MaterialTheme.typography.headlineSmall
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(id = R.string.back),
                            tint = MaterialTheme.colorScheme.onBackground
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = ::refreshModels,
                        enabled = !loading &&
                            usesGeminiBackend() &&
                            selectedGeminiApiMode == InferenceBackendSettings.GeminiApiMode.AI_STUDIO_DIRECT
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Refresh,
                            contentDescription = stringResource(id = R.string.settings_refresh_models),
                            tint = MaterialTheme.colorScheme.onBackground
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
            LazyColumn(
                modifier = backgroundModifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .consumeWindowInsets(innerPadding)
                    .imePadding()
                    .padding(horizontal = horizontalPadding, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = stringResource(id = R.string.settings_provider_title),
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = stringResource(id = R.string.settings_provider_description),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = stringResource(id = R.string.settings_provider_response_title),
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            val responseProviderRows = listOf(
                                InferenceBackendSettings.Provider.AZURE_OPENAI to stringResource(id = R.string.settings_provider_azure_openai),
                                InferenceBackendSettings.Provider.GEMINI to stringResource(id = R.string.settings_provider_gemini),
                                InferenceBackendSettings.Provider.LOCAL_SERVER to stringResource(id = R.string.settings_provider_local_server)
                            )
                            responseProviderRows.forEach { (provider, label) ->
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            selectedResponseProvider = provider
                                            InferenceBackendSettings.setResponseProvider(context, provider)
                                        }
                                        .padding(vertical = 2.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = selectedResponseProvider == provider,
                                        onClick = {
                                            selectedResponseProvider = provider
                                            InferenceBackendSettings.setResponseProvider(context, provider)
                                        }
                                    )
                                    Text(
                                        text = label,
                                        style = MaterialTheme.typography.bodyLarge,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                            Text(
                                text = stringResource(id = R.string.settings_provider_ir_title),
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            val irProviderRows = listOf(
                                InferenceBackendSettings.Provider.AZURE_OPENAI to stringResource(id = R.string.settings_provider_azure_openai),
                                InferenceBackendSettings.Provider.GEMINI to stringResource(id = R.string.settings_provider_gemini),
                                InferenceBackendSettings.Provider.LOCAL_SERVER to stringResource(id = R.string.settings_provider_local_server),
                                InferenceBackendSettings.Provider.ON_DEVICE_LITERT to stringResource(id = R.string.settings_provider_on_device_litert)
                            )
                            irProviderRows.forEach { (provider, label) ->
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            selectedIrProvider = provider
                                            InferenceBackendSettings.setIrProvider(context, provider)
                                        }
                                        .padding(vertical = 2.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = selectedIrProvider == provider,
                                        onClick = {
                                            selectedIrProvider = provider
                                            InferenceBackendSettings.setIrProvider(context, provider)
                                        }
                                    )
                                    Text(
                                        text = label,
                                        style = MaterialTheme.typography.bodyLarge,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                            if (usesGeminiBackend()) {
                                Text(
                                    text = stringResource(id = R.string.settings_gemini_api_mode_title),
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    modifier = Modifier.padding(top = 4.dp)
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_gemini_api_mode_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 2.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = true,
                                        onClick = null
                                    )
                                    Text(
                                        text = stringResource(id = R.string.settings_gemini_api_mode_vertex_express),
                                        style = MaterialTheme.typography.bodyLarge,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Text(
                                    text = stringResource(id = R.string.settings_vertex_express_key_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }

                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(
                                modifier = Modifier.weight(1f),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_rendering_title),
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_render_without_outer_card_title),
                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_render_without_outer_card_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Switch(
                                checked = renderWithoutOuterCard,
                                onCheckedChange = { enabled ->
                                    renderWithoutOuterCard = enabled
                                    InferenceBackendSettings.setRenderWithoutOuterCard(context, enabled)
                                },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = MaterialTheme.colorScheme.primary,
                                    checkedTrackColor = MaterialTheme.colorScheme.primaryContainer,
                                    uncheckedThumbColor = MaterialTheme.colorScheme.outline,
                                    uncheckedTrackColor = MaterialTheme.colorScheme.surfaceVariant
                                )
                            )
                        }
                    }
                }

                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            val transparencyPercent = (renderCardTransparency * 100).roundToInt()
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_render_card_transparency_title),
                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(
                                        id = R.string.settings_render_card_transparency_value,
                                        transparencyPercent
                                    ),
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            Text(
                                text = stringResource(id = R.string.settings_render_card_transparency_description),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Slider(
                                value = renderCardTransparency,
                                onValueChange = { value ->
                                    renderCardTransparency = value
                                    InferenceBackendSettings.setRenderCardTransparency(context, value)
                                },
                                valueRange = InferenceBackendSettings.MIN_RENDER_CARD_TRANSPARENCY..InferenceBackendSettings.MAX_RENDER_CARD_TRANSPARENCY,
                                steps = 5
                            )
                        }
                    }
                }

                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            val transparencyPercent = (renderBackgroundTransparency * 100).roundToInt()
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_render_background_transparency_title),
                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(
                                        id = R.string.settings_render_background_transparency_value,
                                        transparencyPercent
                                    ),
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            Text(
                                text = stringResource(id = R.string.settings_render_background_transparency_description),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Slider(
                                value = renderBackgroundTransparency,
                                onValueChange = { value ->
                                    renderBackgroundTransparency = value
                                    InferenceBackendSettings.setRenderBackgroundTransparency(context, value)
                                },
                                valueRange = InferenceBackendSettings.MIN_RENDER_BACKGROUND_TRANSPARENCY..InferenceBackendSettings.MAX_RENDER_BACKGROUND_TRANSPARENCY,
                                steps = 8
                            )
                        }
                    }
                }

                if (usesAzureOpenAiBackend()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                            colors = genUiCardColors(GenUiCardTone.Neutral),
                            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 14.dp, vertical = 12.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_azure_openai_title),
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_azure_openai_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                OutlinedTextField(
                                    value = azureOpenAiResponsesEndpoint,
                                    onValueChange = {
                                        azureOpenAiResponsesEndpoint = it
                                        InferenceBackendSettings.setAzureOpenAiResponsesEndpoint(context, it)
                                    },
                                    label = { Text(stringResource(id = R.string.settings_azure_openai_endpoint_label)) },
                                    singleLine = true,
                                    modifier = Modifier.fillMaxWidth()
                                )
                                OutlinedTextField(
                                    value = azureOpenAiDeployment,
                                    onValueChange = {
                                        azureOpenAiDeployment = it
                                        InferenceBackendSettings.setAzureOpenAiDeployment(context, it)
                                    },
                                    label = { Text(stringResource(id = R.string.settings_azure_openai_deployment_label)) },
                                    singleLine = true,
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_azure_openai_key_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }

                if (usesLocalBackend()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                            colors = genUiCardColors(GenUiCardTone.Neutral),
                            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 14.dp, vertical = 12.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_local_server_title),
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                OutlinedTextField(
                                    value = localServerBaseUrl,
                                    onValueChange = {
                                        localServerBaseUrl = it
                                        InferenceBackendSettings.setLocalServerBaseUrl(context, it)
                                    },
                                    label = { Text(stringResource(id = R.string.settings_local_server_url_label)) },
                                    singleLine = true,
                                    modifier = Modifier.fillMaxWidth()
                                )
                                OutlinedTextField(
                                    value = localModelPath,
                                    onValueChange = {
                                        localModelPath = it
                                        InferenceBackendSettings.setLocalModelPath(context, it)
                                    },
                                    label = { Text(stringResource(id = R.string.settings_local_model_path_label)) },
                                    singleLine = true,
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_local_server_hint),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }

                if (usesOnDeviceBackend()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                            colors = genUiCardColors(GenUiCardTone.Neutral),
                            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 14.dp, vertical = 12.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_on_device_title),
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_on_device_hint),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_on_device_accelerator_title),
                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_on_device_accelerator_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                InferenceBackendSettings.Accelerator.entries.forEach { accelerator ->
                                    val label = when (accelerator) {
                                        InferenceBackendSettings.Accelerator.AUTO -> stringResource(id = R.string.settings_on_device_accelerator_auto)
                                        InferenceBackendSettings.Accelerator.GPU -> stringResource(id = R.string.settings_on_device_accelerator_gpu)
                                        InferenceBackendSettings.Accelerator.CPU -> stringResource(id = R.string.settings_on_device_accelerator_cpu)
                                        InferenceBackendSettings.Accelerator.NPU -> stringResource(id = R.string.settings_on_device_accelerator_npu)
                                    }
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clickable {
                                                onDeviceAccelerator = accelerator
                                                InferenceBackendSettings.setOnDeviceAccelerator(context, accelerator)
                                            },
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        RadioButton(
                                            selected = onDeviceAccelerator == accelerator,
                                            onClick = {
                                                onDeviceAccelerator = accelerator
                                                InferenceBackendSettings.setOnDeviceAccelerator(context, accelerator)
                                            }
                                        )
                                        Text(
                                            text = label,
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                    }
                                }
                                OnDeviceModelCatalog.entries.forEach { entry ->
                                    val downloaded = onDeviceModelRefreshKey.let { entry.isDownloaded(context) }
                                    val selected = downloaded && onDeviceModelPath == entry.localPath(context)
                                    val isDownloading = onDeviceDownloading[entry.id] == true
                                    val progress = onDeviceDownloadProgress[entry.id]
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
                                                .padding(horizontal = 10.dp, vertical = 8.dp),
                                            verticalArrangement = Arrangement.spacedBy(6.dp)
                                        ) {
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .clickable(enabled = downloaded) {
                                                        onDeviceModelPath = entry.localPath(context)
                                                        InferenceBackendSettings.setOnDeviceModelPath(context, onDeviceModelPath)
                                                    },
                                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                RadioButton(
                                                    selected = selected,
                                                    enabled = downloaded,
                                                    onClick = if (downloaded) {
                                                        {
                                                            onDeviceModelPath = entry.localPath(context)
                                                            InferenceBackendSettings.setOnDeviceModelPath(context, onDeviceModelPath)
                                                        }
                                                    } else {
                                                        null
                                                    }
                                                )
                                                Column(
                                                    modifier = Modifier.weight(1f),
                                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                                ) {
                                                    Text(
                                                        text = entry.displayName,
                                                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                                                        color = MaterialTheme.colorScheme.onSurface,
                                                        maxLines = 1,
                                                        overflow = TextOverflow.Ellipsis
                                                    )
                                                    Text(
                                                        text = entry.repoId,
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                        maxLines = 1,
                                                        overflow = TextOverflow.Ellipsis
                                                    )
                                                    Text(
                                                        text = "${entry.quantization} · ${entry.subtitle}",
                                                        style = MaterialTheme.typography.labelSmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                        maxLines = 2,
                                                        overflow = TextOverflow.Ellipsis
                                                    )
                                                    Text(
                                                        text = if (downloaded) {
                                                            stringResource(id = R.string.settings_on_device_downloaded)
                                                        } else if (!entry.isDownloadable) {
                                                            "${stringResource(id = R.string.settings_on_device_sideload_required)} - ${entry.approximateSize}"
                                                        } else {
                                                            "${stringResource(id = R.string.settings_on_device_not_downloaded)} - ${entry.approximateSize}"
                                                        },
                                                        style = MaterialTheme.typography.labelSmall,
                                                        color = if (downloaded) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                }
                                                TextButton(
                                                    enabled = !isDownloading && (!downloaded || !selected) &&
                                                        (downloaded || entry.isDownloadable),
                                                    onClick = {
                                                        if (downloaded) {
                                                            onDeviceModelPath = entry.localPath(context)
                                                            InferenceBackendSettings.setOnDeviceModelPath(context, onDeviceModelPath)
                                                        } else {
                                                            onDeviceDownloadError = null
                                                            onDeviceDownloading[entry.id] = true
                                                            onDeviceDownloadProgress[entry.id] = null
                                                            coroutineScope.launch {
                                                                val result = withContext(Dispatchers.IO) {
                                                                    runCatching {
                                                                        OnDeviceModelDownloader.download(context.applicationContext, entry) { itemProgress ->
                                                                            coroutineScope.launch {
                                                                                onDeviceDownloadProgress[entry.id] = itemProgress.fraction
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                onDeviceDownloading[entry.id] = false
                                                                onDeviceDownloadProgress.remove(entry.id)
                                                                result.onSuccess { file ->
                                                                    onDeviceModelPath = file.absolutePath
                                                                    InferenceBackendSettings.setOnDeviceModelPath(context, file.absolutePath)
                                                                    onDeviceModelRefreshKey++
                                                                }.onFailure {
                                                                    onDeviceDownloadError = it.message ?: it.javaClass.simpleName
                                                                }
                                                            }
                                                        }
                                                    }
                                                ) {
                                                    Text(
                                                        text = when {
                                                            isDownloading -> stringResource(id = R.string.settings_on_device_downloading)
                                                            downloaded && selected -> stringResource(id = R.string.settings_on_device_selected)
                                                            downloaded -> stringResource(id = R.string.settings_on_device_select)
                                                            !entry.isDownloadable -> stringResource(id = R.string.settings_on_device_sideload)
                                                            else -> stringResource(id = R.string.settings_on_device_download)
                                                        },
                                                        maxLines = 1
                                                    )
                                                }
                                            }
                                            if (isDownloading) {
                                                if (progress != null) {
                                                    LinearProgressIndicator(
                                                        progress = { progress },
                                                        modifier = Modifier.fillMaxWidth()
                                                    )
                                                } else {
                                                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                                                }
                                            }
                                        }
                                    }
                                }
                                if (!onDeviceDownloadError.isNullOrBlank()) {
                                    Text(
                                        text = "${stringResource(id = R.string.settings_on_device_download_failed)}: ${onDeviceDownloadError.orEmpty()}",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.error
                                    )
                                }
                            }
                        }
                    }
                }

                if (usesGeminiBackend()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                            colors = genUiCardColors(GenUiCardTone.Neutral),
                            elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 14.dp, vertical = 12.dp),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Text(
                                    text = stringResource(id = R.string.settings_model_title),
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_gemini_api_mode_vertex_express),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = if (selectedResponseProvider == InferenceBackendSettings.Provider.GEMINI) {
                                        "${stringResource(id = R.string.settings_model_response_title)}: $selectedResponseModel"
                                    } else if (selectedResponseProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
                                        "${stringResource(id = R.string.settings_model_response_title)}: $azureOpenAiDeployment"
                                    } else if (selectedResponseProvider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
                                        "${stringResource(id = R.string.settings_model_response_title)}: ${stringResource(id = R.string.settings_provider_on_device_litert)}"
                                    } else {
                                        "${stringResource(id = R.string.settings_model_response_title)}: ${stringResource(id = R.string.settings_provider_local_server)}"
                                    },
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = if (selectedIrProvider == InferenceBackendSettings.Provider.GEMINI) {
                                        "${stringResource(id = R.string.settings_model_ir_title)}: $selectedIrModel"
                                    } else if (selectedIrProvider == InferenceBackendSettings.Provider.AZURE_OPENAI) {
                                        "${stringResource(id = R.string.settings_model_ir_title)}: $azureOpenAiDeployment"
                                    } else if (selectedIrProvider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT) {
                                        "${stringResource(id = R.string.settings_model_ir_title)}: ${stringResource(id = R.string.settings_provider_on_device_litert)}"
                                    } else {
                                        "${stringResource(id = R.string.settings_model_ir_title)}: ${stringResource(id = R.string.settings_provider_local_server)}"
                                    },
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_model_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }

                    if (loading) {
                        item {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 4.dp, vertical = 2.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(14.dp),
                                    strokeWidth = 2.dp
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_model_loading),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }

                    if (!errorText.isNullOrBlank()) {
                        item {
                            Text(
                                text = errorText.orEmpty(),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                    }

                    item {
                        Text(
                            text = stringResource(id = R.string.settings_model_response_title),
                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }

                    if (selectedResponseProvider == InferenceBackendSettings.Provider.GEMINI) {
                        items(
                            availableModels.filter(GeminiModelSettings::isVertexExpressCompatibleModel),
                            key = { "response_$it" },
                        ) { model ->
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable {
                                        selectedResponseModel = model
                                        GeminiModelSettings.setResponseModel(context, model)
                                    },
                                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                                colors = genUiCardColors(GenUiCardTone.Neutral),
                                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 10.dp, vertical = 8.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = model == selectedResponseModel,
                                        onClick = {
                                            selectedResponseModel = model
                                            GeminiModelSettings.setResponseModel(context, model)
                                        }
                                    )
                                    Column(
                                        modifier = Modifier.weight(1f),
                                        verticalArrangement = Arrangement.spacedBy(2.dp)
                                    ) {
                                        Text(
                                            text = model,
                                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurface,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                        Text(
                                            text = "models/$model",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }
                                }
                            }
                        }
                    } else {
                        item {
                            Text(
                                text = stringResource(id = R.string.settings_model_not_used_local),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    item {
                        Text(
                            text = stringResource(id = R.string.settings_model_ir_title),
                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }

                    if (selectedIrProvider == InferenceBackendSettings.Provider.GEMINI) {
                        items(
                            availableModels.filter(GeminiModelSettings::isVertexExpressCompatibleModel),
                            key = { "ir_$it" },
                        ) { model ->
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable {
                                        selectedIrModel = model
                                        GeminiModelSettings.setIrModel(context, model)
                                    },
                                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                                colors = genUiCardColors(GenUiCardTone.Neutral),
                                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 10.dp, vertical = 8.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = model == selectedIrModel,
                                        onClick = {
                                            selectedIrModel = model
                                            GeminiModelSettings.setIrModel(context, model)
                                        }
                                    )
                                    Column(
                                        modifier = Modifier.weight(1f),
                                        verticalArrangement = Arrangement.spacedBy(2.dp)
                                    ) {
                                        Text(
                                            text = model,
                                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                                            color = MaterialTheme.colorScheme.onSurface,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                        Text(
                                            text = "models/$model",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }
                                }
                            }
                        }
                    } else {
                        item {
                            Text(
                                text = stringResource(id = R.string.settings_model_not_used_local),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                        colors = genUiCardColors(GenUiCardTone.Neutral),
                        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = stringResource(id = R.string.settings_mcp_title),
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    Text(
                                        text = stringResource(id = R.string.settings_mcp_description),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                Switch(
                                    checked = mcpEnabled,
                                    onCheckedChange = {
                                        mcpEnabled = it
                                        McpSettings.setEnabled(context, it)
                                    },
                                    colors = SwitchDefaults.colors(
                                        checkedThumbColor = MaterialTheme.colorScheme.primary,
                                        checkedTrackColor = MaterialTheme.colorScheme.primaryContainer,
                                        uncheckedThumbColor = MaterialTheme.colorScheme.outline,
                                        uncheckedTrackColor = MaterialTheme.colorScheme.surfaceVariant
                                    )
                                )
                            }

                            if (mcpEnabled) {
                                Text(
                                    text = stringResource(id = R.string.settings_mcp_api_keys_title),
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    modifier = Modifier.padding(top = 8.dp)
                                )
                                Text(
                                    text = stringResource(id = R.string.settings_mcp_api_keys_description),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                // Weather uses Open-Meteo without a key.
                                Text(
                                    text = "Weather: Open-Meteo (no key required)",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                val keyedDomains = listOf(
                                    McpSettings.Domain.FLIGHTS to stringResource(id = R.string.settings_mcp_key_flights),
                                    McpSettings.Domain.RESTAURANTS to stringResource(id = R.string.settings_mcp_key_restaurants),
                                    McpSettings.Domain.HOTELS to stringResource(id = R.string.settings_mcp_key_hotels),
                                    McpSettings.Domain.PLACES to stringResource(id = R.string.settings_mcp_key_places),
                                    McpSettings.Domain.NEWS to stringResource(id = R.string.settings_mcp_key_news)
                                )
                                keyedDomains.forEach { (domain, label) ->
                                    OutlinedTextField(
                                        value = mcpApiKeys[domain] ?: "",
                                        onValueChange = { newVal ->
                                            mcpApiKeys[domain] = newVal
                                            McpSettings.setApiKey(context, domain, newVal)
                                        },
                                        label = { Text(label) },
                                        placeholder = { Text("Optional custom key") },
                                        singleLine = true,
                                        modifier = Modifier.fillMaxWidth()
                                    )
                                }
                            }
                        }
                    }
                }

                item {
                    HorizontalDivider(
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f)
                    )
                }
            }
        }
    }
}

private fun defaultModelOptions(selectedResponseModel: String, selectedIrModel: String): List<String> {
    val defaults = listOf(
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-pro-latest",
        "gemini-flash-latest",
        "gemini-flash-lite-latest"
    )
    return mergeSelectedModels(defaults, selectedResponseModel, selectedIrModel)
}

private fun mergeSelectedModels(
    models: List<String>,
    selectedResponseModel: String,
    selectedIrModel: String
): List<String> {
    val normalized = models
        .map { GeminiModelSettings.normalizeModelName(it) }
        .filter { it.isNotBlank() }
        .distinct()
        .toMutableList()
    if (selectedResponseModel.isNotBlank() && normalized.none { it == selectedResponseModel }) {
        normalized.add(0, selectedResponseModel)
    }
    if (selectedIrModel.isNotBlank() && normalized.none { it == selectedIrModel }) {
        normalized.add(0, selectedIrModel)
    }
    return normalized
}
