package com.samsung.genuicraft

import android.app.WallpaperManager
import android.graphics.RenderEffect
import android.graphics.Shader
import android.graphics.drawable.Drawable
import android.os.Build
import android.util.Log
import android.widget.ImageView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView

private const val FALLBACK_WALLPAPER_BLUR_RADIUS_PX = 96f

@Composable
fun GenUiScreenBackground(
    modifier: Modifier = Modifier,
    content: @Composable (Modifier) -> Unit
) {
    val context = LocalContext.current
    val blurAvailable = remember(context) { context.isCrossWindowBlurActive() }
    Box(modifier = modifier.fillMaxSize()) {
        if (!blurAvailable) {
            BlurredWallpaperLayer(modifier = Modifier.fillMaxSize())
        }
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(genUiScreenOverlayColor())
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
    val manager = WallpaperManager.getInstance(context)
    val drawable = sequenceOf(
        "peekDrawable",
        "getFastDrawable",
        "getDrawable"
    ).mapNotNull { methodName ->
        runCatching {
            val method = WallpaperManager::class.java.getMethod(methodName)
            method.invoke(manager) as? Drawable
        }.onFailure { error ->
            Log.d("GenUiBackdrop", "Wallpaper method $methodName unavailable: ${error.message}")
        }.getOrNull()
    }.firstOrNull()
    if (drawable == null) {
        Log.w("GenUiBackdrop", "Wallpaper drawable unavailable for fallback frost layer.")
    }
    return drawable.also {
        Log.d("GenUiBackdrop", "Wallpaper drawable available=${drawable != null}")
    }
}
