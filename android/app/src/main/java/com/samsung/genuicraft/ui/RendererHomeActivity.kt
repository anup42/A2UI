package com.samsung.genuicraft

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import java.io.File

private data class StatusMessage(
    val text: String,
    val isError: Boolean
)

class RendererHomeActivity : AppCompatActivity() {
    private var statusMessage by mutableStateOf(StatusMessage("", isError = false))
    private var nativeRenderingEnabled by mutableStateOf(true)

    private val filePicker =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri == null) {
                showStatus(getString(R.string.status_file_picker_cancelled), isError = false)
            } else {
                loadFromUri(uri)
            }
        }

    private val sampleSetPicker =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode != RESULT_OK) {
                return@registerForActivityResult
            }
            val selectedId = result.data?.getStringExtra(SampleDatasetPickerActivity.EXTRA_DATASET_ID)
            val dataset = SampleDatasets.byId(selectedId)
            if (dataset == null) {
                showStatus("Could not resolve selected sample set.", isError = true)
                return@registerForActivityResult
            }
            loadBundledSample(dataset)
        }

    private val wallpaperPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            val granted = result.values.any { it }
            if (granted) {
                showStatus("Wallpaper blur enabled.", isError = false)
            } else {
                showStatus("Wallpaper blur fallback disabled (media permission denied).", isError = false)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        statusMessage = StatusMessage(getString(R.string.status_ready_select_source), isError = false)
        requestWallpaperPermissionIfNeeded()

        setContent {
            GenUiCraftTheme {
                HomeScreen(
                    nativeRenderingEnabled = nativeRenderingEnabled,
                    onNativeRenderingChanged = { nativeRenderingEnabled = it },
                    onSelectFile = { filePicker.launch(arrayOf("application/json", "text/plain", "*/*")) },
                    onLoadSample = { openSampleSetPicker() }
                )
            }
        }
    }

    private fun loadFromUri(uri: Uri) {
        try {
            contentResolver.openInputStream(uri).use { input ->
                if (input == null) {
                    showStatus(getString(R.string.error_uri_open), isError = true)
                    return
                }
                val raw = input.bufferedReader(Charsets.UTF_8).use { it.readText() }
                val sourceLabel = resolveDisplayName(uri)
                val records = GenUiRecordParser.parseRecords(
                    raw = raw,
                    sourceDir = resolveFileSourceDir(uri),
                    sourceLabel = sourceLabel
                )
                openItemList(sourceLabel, records)
            }
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun openSampleSetPicker() {
        if (SampleDatasets.all.size == 1) {
            loadBundledSample(SampleDatasets.all.first())
            return
        }
        sampleSetPicker.launch(Intent(this, SampleDatasetPickerActivity::class.java))
    }

    private fun requestWallpaperPermissionIfNeeded() {
        val permissions = wallpaperPermissionsForCurrentApi()
        if (permissions.isEmpty()) {
            return
        }
        val allGranted = permissions.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
        if (allGranted) {
            return
        }
        wallpaperPermissionLauncher.launch(permissions)
    }

    private fun wallpaperPermissionsForCurrentApi(): Array<String> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
        } else {
            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun loadBundledSample(dataset: SampleDataset) {
        try {
            val raw = assets.open(dataset.assetFileName).bufferedReader(Charsets.UTF_8).use { it.readText() }
            val records = GenUiRecordParser.parseRecords(
                raw = raw,
                sourceDir = null,
                sourceLabel = dataset.sourceLabel
            )
            openItemList(dataset.sourceLabel, records)
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun openItemList(sourceLabel: String, records: List<GenUiRecord>) {
        if (records.isEmpty()) {
            showStatus(getString(R.string.error_no_valid_json, sourceLabel), isError = true)
            return
        }

        val renderMode = if (nativeRenderingEnabled) RenderMode.NATIVE else RenderMode.WEB
        RenderSessionStore.update(
            sourceLabel = sourceLabel,
            records = records,
            renderMode = renderMode
        )

        showStatus(getString(R.string.status_loaded_records, records.size, sourceLabel), isError = false)
        startActivity(Intent(this, ItemListActivity::class.java))
    }

    private fun resolveDisplayName(uri: Uri): String {
        if (uri.scheme.equals("content", ignoreCase = true)) {
            contentResolver
                .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0 && cursor.moveToFirst()) {
                        val name = cursor.getString(index)
                        if (!name.isNullOrBlank()) {
                            return name
                        }
                    }
                }
        }

        val path = uri.lastPathSegment
        return if (!path.isNullOrBlank()) path.substringAfterLast('/') else uri.toString()
    }

    private fun resolveFileSourceDir(uri: Uri): File? {
        if (!uri.scheme.equals("file", ignoreCase = true)) {
            return null
        }
        return uri.path?.let { File(it).parentFile }
    }

    private fun showError(exc: Exception) {
        val msg = exc.message?.trim().takeUnless { it.isNullOrEmpty() } ?: exc.javaClass.simpleName
        showStatus(getString(R.string.error_render_failed, msg), isError = true)
    }

    private fun showStatus(message: String, isError: Boolean) {
        statusMessage = StatusMessage(text = message, isError = isError)
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun HomeScreen(
    nativeRenderingEnabled: Boolean,
    onNativeRenderingChanged: (Boolean) -> Unit,
    onSelectFile: () -> Unit,
    onLoadSample: () -> Unit
) {
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 20.dp
        DeviceSizeClass.Medium -> 26.dp
        DeviceSizeClass.Expanded -> 32.dp
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = stringResourceCompat(R.string.home_title),
                        style = MaterialTheme.typography.headlineSmall
                    )
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
            Column(
                modifier = backgroundModifier
                .padding(horizontal = horizontalPadding, vertical = 16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
            Text(
                text = stringResourceCompat(R.string.home_subtitle),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth(),
                shape = RoundedCornerShape(GenUiTokens.RadiusXl),
                colors = genUiCardColors(GenUiCardTone.Primary),
                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationMd),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = stringResourceCompat(R.string.home_source_title),
                        style = MaterialTheme.typography.titleLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = stringResourceCompat(R.string.home_source_subtitle),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Button(
                        onClick = onSelectFile,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusPill)
                    ) {
                        Text(
                            text = stringResourceCompat(R.string.select_genui_file),
                            style = MaterialTheme.typography.labelLarge
                        )
                    }

                    OutlinedButton(
                        onClick = onLoadSample,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(GenUiTokens.RadiusPill)
                    ) {
                        Text(
                            text = stringResourceCompat(R.string.open_sample_dataset),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth(),
                shape = RoundedCornerShape(GenUiTokens.RadiusLg),
                colors = genUiCardColors(GenUiCardTone.Neutral),
                elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
                border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = stringResourceCompat(R.string.native_rendering_title),
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = stringResourceCompat(R.string.native_rendering_subtitle),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Switch(
                        checked = nativeRenderingEnabled,
                        onCheckedChange = onNativeRenderingChanged
                    )
                }
            }
            }
        }
    }
}

@Composable
private fun stringResourceCompat(id: Int): String = androidx.compose.ui.res.stringResource(id = id)
