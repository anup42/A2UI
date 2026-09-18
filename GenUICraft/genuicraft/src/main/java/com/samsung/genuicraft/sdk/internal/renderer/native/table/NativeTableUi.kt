package com.samsung.genuicraft.sdk.internal.renderer.native.table

import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import com.samsung.genuicraft.sdk.internal.theme.GenUiTokens

internal object NativeTableUi {
    @Composable
    fun TableCellDivider() {
        val dark = isSystemInDarkTheme()
        Box(
            modifier = Modifier
                .fillMaxHeight()
                .width(GenUiTokens.BorderSm)
                .background(MaterialTheme.colorScheme.outlineVariant.copy(alpha = if (dark) 0.24f else 0.18f))
        )
    }

    @Composable
    fun tableRowBackground(isHeader: Boolean, rowIndex: Int): Color {
        val dark = isSystemInDarkTheme()
        return when {
            isHeader -> MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.08f else 0.045f)
            rowIndex % 2 == 0 -> Color.Transparent
            else -> MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.04f else 0.020f)
        }
    }
}
