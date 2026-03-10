package com.samsung.genuicraft

import android.Manifest
import android.app.WallpaperManager
import android.content.pm.PackageManager
import android.graphics.RenderEffect
import android.graphics.Shader
import android.graphics.drawable.Drawable
import android.os.Build
import android.util.Log
import android.widget.ImageView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat

private const val FALLBACK_WALLPAPER_BLUR_RADIUS_PX = 48f

@Composable
fun GenUiScreenBackground(
    modifier: Modifier = Modifier,
    content: @Composable (Modifier) -> Unit
) {
    val context = LocalContext.current
    val blurAvailable = remember(context) { context.isCrossWindowBlurActive() }
    val dark = androidx.compose.foundation.isSystemInDarkTheme()

    Box(modifier = modifier.fillMaxSize()) {
        if (!blurAvailable) {
            BlurredWallpaperLayer(modifier = Modifier.fillMaxSize())
        }
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    MaterialTheme.colorScheme.surfaceContainerLowest.copy(alpha = if (dark) 0.14f else 0.12f)
                )
        )
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(genUiBackgroundBrush())
        )
        content(Modifier.matchParentSize())
    }
}

@Composable
private fun BlurredWallpaperLayer(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val wallpaperDrawable = remember(context) { loadWallpaperDrawable(context) }
    AndroidView(
        modifier = modifier,
        factory = { viewContext ->
            ImageView(viewContext).apply {
                scaleType = ImageView.ScaleType.CENTER_CROP
                setImageDrawable(wallpaperDrawable)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    setRenderEffect(
                        RenderEffect.createBlurEffect(
                            FALLBACK_WALLPAPER_BLUR_RADIUS_PX,
                            FALLBACK_WALLPAPER_BLUR_RADIUS_PX,
                            Shader.TileMode.CLAMP
                        )
                    )
                }
            }
        },
        update = { imageView ->
            if (imageView.drawable == null) {
                imageView.setImageDrawable(wallpaperDrawable)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                imageView.setRenderEffect(
                    RenderEffect.createBlurEffect(
                        FALLBACK_WALLPAPER_BLUR_RADIUS_PX,
                        FALLBACK_WALLPAPER_BLUR_RADIUS_PX,
                        Shader.TileMode.CLAMP
                    )
                )
            }
        }
    )
}

private fun loadWallpaperDrawable(context: android.content.Context): Drawable? {
    if (!hasWallpaperReadPermission(context)) {
        Log.d("GenUiBackdrop", "Wallpaper drawable unavailable: permission not granted")
        return null
    }
    return runCatching {
        WallpaperManager.getInstance(context).drawable
    }.onFailure { error ->
        Log.w("GenUiBackdrop", "Wallpaper load failed: ${error.message}")
    }.getOrNull().also { drawable ->
        Log.d("GenUiBackdrop", "Wallpaper drawable available=${drawable != null}")
    }
}

private fun hasWallpaperReadPermission(context: android.content.Context): Boolean {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        val imagesGranted = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.READ_MEDIA_IMAGES
        ) == PackageManager.PERMISSION_GRANTED
        val externalGranted = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.READ_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED
        return imagesGranted || externalGranted
    }
    return ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.READ_EXTERNAL_STORAGE
    ) == PackageManager.PERMISSION_GRANTED
}
