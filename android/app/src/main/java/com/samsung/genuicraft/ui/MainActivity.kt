package com.samsung.genuicraft

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()
        setContent {
            GenUiCraftTheme {
                LauncherScreen(
                    onOpenRenderer = { startActivity(Intent(this, RendererHomeActivity::class.java)) },
                    onOpenGenUi = { startActivity(Intent(this, GenUiAssistantActivity::class.java)) },
                    onOpenIrDemo = { startActivity(Intent(this, IrDemoListActivity::class.java)) },
                    onOpenSdk = { startActivity(Intent(this, GenUiSdkDemoActivity::class.java)) },
                    onOpenSettings = { startActivity(Intent(this, SettingsActivity::class.java)) }
                )
            }
        }
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun LauncherScreen(
    onOpenRenderer: () -> Unit,
    onOpenGenUi: () -> Unit,
    onOpenIrDemo: () -> Unit,
    onOpenSdk: () -> Unit,
    onOpenSettings: () -> Unit
) {
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 18.dp
        DeviceSizeClass.Medium -> 24.dp
        DeviceSizeClass.Expanded -> 30.dp
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = stringResource(id = R.string.launcher_title),
                        style = MaterialTheme.typography.headlineSmall
                    )
                },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(
                            imageVector = Icons.Filled.Settings,
                            contentDescription = stringResource(id = R.string.settings_title),
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
            Column(
                modifier = backgroundModifier
                    .padding(innerPadding)
                    .consumeWindowInsets(innerPadding)
                    .padding(horizontal = horizontalPadding, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                LauncherCard(
                    title = stringResource(id = R.string.launcher_genui_title),
                    onClick = onOpenGenUi
                )

                LauncherCard(
                    title = stringResource(id = R.string.launcher_renderer_title),
                    onClick = onOpenRenderer
                )

                LauncherCard(
                    title = stringResource(id = R.string.launcher_ir_demo_title),
                    onClick = onOpenIrDemo
                )
                LauncherCard(title = "GenUICraft SDK · Bixby50", onClick = onOpenSdk)
            }
        }
    }
}

@Composable
private fun LauncherCard(
    title: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(GenUiTokens.RadiusXl),
        colors = genUiCardColors(GenUiCardTone.Neutral),
        elevation = CardDefaults.cardElevation(defaultElevation = GenUiTokens.ElevationSm),
        border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(0.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}
