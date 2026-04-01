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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import com.samsung.genuicraft.mcp.McpSettings
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.samsung.genuicraft.pipeline.IrPromptVersionSettings

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
    var localServerBaseUrl by remember {
        mutableStateOf(InferenceBackendSettings.getLocalServerBaseUrl(context))
    }
    var localModelPath by remember {
        mutableStateOf(InferenceBackendSettings.getLocalModelPath(context))
    }
    var availableModels by remember {
        mutableStateOf(defaultModelOptions(selectedResponseModel, selectedIrModel))
    }
    val irPromptOptions = remember { IrPromptVersionSettings.options() }
    var selectedIrPromptVersionId by remember {
        mutableStateOf(IrPromptVersionSettings.getSelectedVersionId(context))
    }
    var loading by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }
    
    var mcpEnabled by remember {
        mutableStateOf(McpSettings.isEnabled(context))
    }
    
    val mcpApiKeys = remember {
        val map = androidx.compose.runtime.mutableStateMapOf<McpSettings.Domain, String>()
        McpSettings.Domain.entries.forEach { domain ->
            map[domain] = McpSettings.getApiKey(context, domain)
        }
        map
    }

    fun usesGeminiBackend(): Boolean {
        return selectedResponseProvider == InferenceBackendSettings.Provider.GEMINI ||
            selectedIrProvider == InferenceBackendSettings.Provider.GEMINI
    }

    fun usesLocalBackend(): Boolean {
        return selectedResponseProvider == InferenceBackendSettings.Provider.LOCAL_SERVER ||
            selectedIrProvider == InferenceBackendSettings.Provider.LOCAL_SERVER
    }

    fun refreshModels() {
        GeminiApiKeyProvider.refresh(context)
        if (!usesGeminiBackend()) {
            loading = false
            errorText = null
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

    LaunchedEffect(selectedResponseProvider, selectedIrProvider) {
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
                        enabled = !loading && usesGeminiBackend()
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
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .consumeWindowInsets(innerPadding)
        ) { backgroundModifier ->
            LazyColumn(
                modifier = backgroundModifier
                    .fillMaxSize()
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
                                InferenceBackendSettings.Provider.GEMINI to stringResource(id = R.string.settings_provider_gemini),
                                InferenceBackendSettings.Provider.LOCAL_SERVER to stringResource(id = R.string.settings_provider_local_server)
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
                        val selectedPromptOption = irPromptOptions.firstOrNull {
                            it.id == selectedIrPromptVersionId
                        } ?: IrPromptVersionSettings.defaultOption()
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = stringResource(id = R.string.settings_ir_prompt_title),
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = stringResource(id = R.string.settings_ir_prompt_description),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = stringResource(
                                    id = R.string.settings_ir_prompt_selected,
                                    selectedPromptOption.title
                                ),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            irPromptOptions.forEach { option ->
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            selectedIrPromptVersionId = option.id
                                            IrPromptVersionSettings.setSelectedVersionId(context, option.id)
                                        }
                                        .padding(vertical = 2.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = selectedIrPromptVersionId == option.id,
                                        onClick = {
                                            selectedIrPromptVersionId = option.id
                                            IrPromptVersionSettings.setSelectedVersionId(context, option.id)
                                        }
                                    )
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(1.dp)
                                    ) {
                                        Text(
                                            text = option.title,
                                            style = MaterialTheme.typography.bodyLarge,
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                        Text(
                                            text = option.description,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
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
                                    text = if (selectedResponseProvider == InferenceBackendSettings.Provider.GEMINI) {
                                        "${stringResource(id = R.string.settings_model_response_title)}: $selectedResponseModel"
                                    } else {
                                        "${stringResource(id = R.string.settings_model_response_title)}: ${stringResource(id = R.string.settings_provider_local_server)}"
                                    },
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = if (selectedIrProvider == InferenceBackendSettings.Provider.GEMINI) {
                                        "${stringResource(id = R.string.settings_model_ir_title)}: $selectedIrModel"
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
                        items(availableModels, key = { "response_$it" }) { model ->
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
                        items(availableModels, key = { "ir_$it" }) { model ->
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
                                // Weather uses Open-Meteo — no key needed
                                Text(
                                    text = "Weather: Open-Meteo (free, no key required)",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                // Show API key fields for all other domains
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
                                        placeholder = { Text("Paste API key here") },
                                        singleLine = true,
                                        modifier = Modifier.fillMaxWidth()
                                    )
                                    // Hint for the key source
                                    val hintRes = when (domain) {
                                        McpSettings.Domain.FLIGHTS, McpSettings.Domain.HOTELS ->
                                            R.string.settings_mcp_key_hint_serpapi
                                        McpSettings.Domain.RESTAURANTS, McpSettings.Domain.PLACES ->
                                            R.string.settings_mcp_key_hint_places
                                        McpSettings.Domain.NEWS ->
                                            R.string.settings_mcp_key_hint_news
                                        else -> null
                                    }
                                    if (hintRes != null) {
                                        Text(
                                            text = stringResource(id = hintRes),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
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
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
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
