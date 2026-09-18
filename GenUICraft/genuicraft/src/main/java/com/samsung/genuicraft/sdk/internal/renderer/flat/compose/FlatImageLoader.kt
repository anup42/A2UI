package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

import com.samsung.genuicraft.sdk.internal.renderer.*

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.ui.platform.LocalContext
import coil.ImageLoader
import coil.decode.SvgDecoder
import coil.imageLoader

/**
 * Single Coil [ImageLoader] for the renderer.
 *
 * Every image and icon composable previously built its own loader inside
 * `remember(context)`, so a surface with N media elements created N loaders,
 * each with its own memory and disk cache. That multiplied memory, defeated
 * shared caching, and made it impossible for a test to substitute a fake
 * loader.
 *
 * `allowHardware(false)` is set here rather than being incidental: it is
 * required for SVG rasterization and for `DatasetRenderCaptureActivity`'s
 * bitmap capture path, so it must hold for every request the renderer makes.
 */
fun buildFlatImageLoader(context: Context): ImageLoader =
    ImageLoader.Builder(context)
        .components { add(SvgDecoder.Factory()) }
        .allowHardware(false)
        .build()

/**
 * Test/host override. When null the renderer uses Coil's application-wide
 * loader, so extracted SDK rendering shares the host application's caches.
 */
val LocalFlatImageLoader = compositionLocalOf<ImageLoader?> { null }

@Composable
internal fun rememberFlatImageLoader(): ImageLoader =
    LocalFlatImageLoader.current ?: LocalContext.current.imageLoader
