package com.samsung.genuicraft.renderer.native.media

import android.net.Uri
import androidx.compose.ui.layout.ContentScale
import java.util.Locale

internal object NativeMediaVisualUtils {
    fun looksLikeImagePath(value: String, labelHint: String? = null): Boolean {
        val normalized = value.trim()
        if (normalized.isBlank() || isLikelyPlaceholderMediaToken(normalized)) {
            return false
        }
        val normalizedLower = normalized.lowercase(Locale.US)
        val pathWithoutQuery = normalizedLower.substringBefore('?').substringBefore('#')
        if (pathWithoutQuery.contains("<") || pathWithoutQuery.contains(">")) {
            return false
        }
        if (
            pathWithoutQuery.endsWith(".png") ||
            pathWithoutQuery.endsWith(".jpg") ||
            pathWithoutQuery.endsWith(".jpeg") ||
            pathWithoutQuery.endsWith(".svg") ||
            pathWithoutQuery.endsWith(".webp")
        ) {
            return true
        }

        val uri = runCatching { Uri.parse(normalized) }.getOrNull()
        val scheme = uri?.scheme?.lowercase(Locale.US)
        if (scheme == "http" || scheme == "https") {
            val host = uri.host?.lowercase(Locale.US).orEmpty()
            val path = uri.path.orEmpty().lowercase(Locale.US)
            if (
                host.contains("loremflickr.com") ||
                host.contains("picsum.photos") ||
                host.contains("placehold.co") ||
                host.contains("dummyimage.com") ||
                host.contains("cdn.jsdelivr.net") ||
                host.contains("raw.githubusercontent.com") ||
                host.contains("upload.wikimedia.org") ||
                host.contains("imgur.com") ||
                host.contains("gstatic.com") ||
                host.contains("twimg.com") ||
                host.contains("places.googleapis.com")
            ) {
                return true
            }
            if (
                path.contains("/icon") ||
                path.contains("/icons/") ||
                path.contains("/image") ||
                path.contains("/images/") ||
                path.contains("/media")
            ) {
                return true
            }
        }
        return false
    }

    fun looksLikeCompactIconUrl(value: String): Boolean {
        val normalized = value.lowercase(Locale.US)
        val iconPathLike = Regex("""(?:^|/)(?:icon|icons)(?:/|[-_.]|$)""")
            .containsMatchIn(normalized)
        return iconPathLike ||
            normalized.contains("weatherapi.com/weather/") ||
            normalized.contains("/weather/64x64/") ||
            normalized.contains("/weather/128x128/") ||
            normalized.contains("/weather/icons/") ||
            normalized.contains("openweathermap.org/img/wn/") ||
            normalized.contains("/img/wn/") ||
            Regex("""/\d{2}[dn](?:@\dx)?\.(png|webp|jpg|jpeg)(?:[?#].*)?$""").containsMatchIn(normalized)
    }

    fun shouldShowMediaLabel(label: String, sanitizeDisplayText: (String) -> String): Boolean {
        val normalized = sanitizeDisplayText(label).lowercase(Locale.US)
        if (normalized.isBlank()) {
            return false
        }
        if (normalized in setOf("image", "icon", "photo", "logo", "media")) {
            return false
        }
        return !Regex("""^(image|icon|photo|logo|media)\s*[:#-]?\s*\d*$""", RegexOption.IGNORE_CASE)
            .matches(normalized)
    }

    fun isVectorImagePath(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        val pathWithoutQuery = normalized.substringBefore('?').substringBefore('#')
        return pathWithoutQuery.endsWith(".svg")
    }

    fun isRasterImagePath(value: String): Boolean {
        val normalized = value.trim().lowercase(Locale.US)
        val pathWithoutQuery = normalized.substringBefore('?').substringBefore('#')
        return pathWithoutQuery.endsWith(".png") ||
            pathWithoutQuery.endsWith(".jpg") ||
            pathWithoutQuery.endsWith(".jpeg") ||
            pathWithoutQuery.endsWith(".webp") ||
            (normalized.contains("places.googleapis.com") && normalized.contains("/media"))
    }

    fun isPhotoLikeImageUrl(value: String): Boolean {
        val normalized = value.trim()
        if (normalized.isBlank()) {
            return false
        }
        val lower = normalized.lowercase(Locale.US)
        if (isVectorImagePath(lower) || looksLikeCompactIconUrl(lower)) {
            return false
        }
        if (isRasterImagePath(lower)) {
            return true
        }

        val uri = runCatching { Uri.parse(normalized) }.getOrNull() ?: return false
        val scheme = uri.scheme?.lowercase(Locale.US)
        if (scheme != "http" && scheme != "https") {
            return false
        }

        val host = uri.host?.lowercase(Locale.US).orEmpty()
        val path = uri.path?.lowercase(Locale.US).orEmpty()
        if (host.isBlank()) {
            return false
        }

        if (
            host.contains("googleusercontent.com") ||
            host.contains("gstatic.com") ||
            host.contains("wikimedia.org") ||
            host.contains("imgur.com") ||
            host.contains("twimg.com") ||
            host.contains("unsplash.com") ||
            host.contains("pexels.com")
        ) {
            return true
        }

        return path.contains("/media") ||
            path.contains("/image") ||
            path.contains("/images/") ||
            path.contains("/img/") ||
            path.contains("/photo") ||
            path.contains("/photos/") ||
            path.contains("/thumbnail") ||
            path.contains("/thumb/") ||
            path.contains("/cover") ||
            path.contains("/hero")
    }

    fun defaultImageScale(
        rawUrl: String,
        fitValue: String?,
        defaultCoverForRaster: Boolean,
        normalizeLayoutToken: (String) -> String
    ): ContentScale {
        val normalizedFit = fitValue?.let(normalizeLayoutToken)
        return when (normalizedFit) {
            "contain", "fit", "inside" -> ContentScale.Fit
            "cover", "crop", "fill", "fillbounds", "fillwidth", "fillheight" -> ContentScale.Crop
            else -> {
                if (defaultCoverForRaster && isPhotoLikeImageUrl(rawUrl)) {
                    ContentScale.Crop
                } else {
                    ContentScale.Fit
                }
            }
        }
    }

    private fun isLikelyPlaceholderMediaToken(value: String): Boolean {
        val normalized = value
            .trim()
            .trim('\'', '"')
            .lowercase(Locale.US)
        if (normalized.isBlank()) {
            return true
        }
        return normalized in setOf(
            "<image_url>",
            "<icon_url>",
            "<url>",
            "image_url",
            "icon_url",
            "url",
            "n/a",
            "na",
            "none",
            "null",
            "--"
        ) || normalized.contains("placeholder")
    }
}
