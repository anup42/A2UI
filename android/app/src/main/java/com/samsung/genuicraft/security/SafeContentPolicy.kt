package com.samsung.genuicraft.security

import java.net.IDN
import java.net.URI
import java.util.Locale

object SafeContentPolicy {
    enum class MediaKind {
        IMAGE,
        ICON,
        VIDEO,
        AUDIO
    }

    private val hostLabelRegex = Regex("""^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$""")
    private val bareDomainRegex = Regex(
        """(?i)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#].*)?"""
    )
    private val blockedHosts = setOf(
        "example.com",
        "example.org",
        "example.net",
        "localhost",
        "loremflickr.com",
        "picsum.photos",
        "placehold.co",
        "placeholder.com",
        "dummyimage.com",
        "via.placeholder.com",
        "placekitten.com",
        "placebear.com",
        "fakeimg.pl",
        "icon.url",
        "static.icons"
    )
    private val blockedTlds = setOf(
        "local",
        "localhost",
        "test",
        "invalid",
        "example"
    )
    private val fileLikeTlds = setOf(
        "png",
        "jpg",
        "jpeg",
        "svg",
        "webp",
        "gif",
        "bmp",
        "ico",
        "json",
        "xml",
        "txt",
        "csv",
        "md",
        "pdf",
        "zip",
        "apk"
    )
    private val allowedImageHosts = setOf(
        "upload.wikimedia.org",
        "commons.wikimedia.org",
        "raw.githubusercontent.com",
        "githubusercontent.com",
        "i.imgur.com",
        "imgur.com",
        "gstatic.com",
        "googleusercontent.com",
        "places.googleapis.com",
        "lh3.googleusercontent.com",
        "pbs.twimg.com",
        "twimg.com",
        "images.unsplash.com",
        "images.pexels.com",
        "cdn.pixabay.com"
    )
    private val allowedIconHosts = setOf(
        "cdn.jsdelivr.net",
        "unpkg.com",
        "openweathermap.org",
        "cdn.weatherapi.com",
        "weatherapi.com"
    )

    fun sanitizeActionUrl(raw: String?): String? {
        val normalized = normalizeNetworkUrl(raw, allowBareDomain = true) ?: return null
        val uri = parseUri(normalized) ?: return null
        if (uri.scheme?.lowercase(Locale.US) != "https") return null
        val host = normalizeHost(uri.host) ?: return null
        if (uri.userInfo != null) return null
        if (!isAllowedPublicHost(host)) return null
        return normalized
    }

    fun isSafeActionUrl(raw: String?): Boolean = sanitizeActionUrl(raw) != null

    fun sanitizeMediaUrl(raw: String?, kind: MediaKind): String? {
        val token = sanitizeToken(raw)
        if (token.isBlank() || isPlaceholderToken(token)) return null
        if (isLocalAssetUrl(token)) return token
        if (isGeneratedVisualUrl(token)) {
            return token.takeIf { kind == MediaKind.IMAGE || kind == MediaKind.ICON }
        }

        val normalized = normalizeNetworkUrl(token, allowBareDomain = false) ?: return null
        val uri = parseUri(normalized) ?: return null
        if (uri.scheme?.lowercase(Locale.US) != "https") return null
        val host = normalizeHost(uri.host) ?: return null
        if (!isAllowedPublicHost(host)) return null

        return when (kind) {
            MediaKind.IMAGE -> normalized.takeIf { isAllowedPhotoUrl(host, uri) }
            MediaKind.ICON -> normalized.takeIf { isAllowedIconUrl(host, uri) }
            MediaKind.VIDEO,
            MediaKind.AUDIO -> normalized.takeIf { sanitizeActionUrl(normalized) != null }
        }
    }

    fun isSafeMediaUrl(raw: String?, kind: MediaKind): Boolean =
        sanitizeMediaUrl(raw, kind) != null

    fun isIconOnlyMediaUrl(raw: String?): Boolean {
        val token = sanitizeToken(raw)
        if (token.isBlank()) return false
        val uri = parseUri(normalizeNetworkUrl(token, allowBareDomain = false) ?: token) ?: return false
        val host = normalizeHost(uri.host).orEmpty()
        val path = uri.path.orEmpty().lowercase(Locale.US)
        return isBootstrapIconUrl(host, path) ||
            isWeatherIconUrl(host, path) ||
            path.contains("/icons/") ||
            path.contains("/icon/") ||
            path.endsWith("-icon.svg") ||
            path.endsWith("_icon.svg")
    }

    fun isLocalAssetUrl(raw: String?): Boolean {
        val token = sanitizeToken(raw)
        val lower = token.lowercase(Locale.US)
        return lower.startsWith("assets/") ||
            lower.startsWith("/assets/") ||
            lower.startsWith("../assets/") ||
            lower.startsWith("./assets/") ||
            lower.startsWith("file:///android_asset/")
    }

    fun isGeneratedVisualUrl(raw: String?): Boolean {
        val token = sanitizeToken(raw)
        val uri = parseUri(token) ?: return false
        return uri.scheme.equals("genuicraft", ignoreCase = true) &&
            uri.host.equals("visual", ignoreCase = true)
    }

    fun looksLikeUrl(raw: String?): Boolean {
        val token = sanitizeToken(raw)
        val lower = token.lowercase(Locale.US)
        return lower.startsWith("http://") ||
            lower.startsWith("https://") ||
            lower.startsWith("//") ||
            lower.startsWith("javascript:") ||
            lower.startsWith("data:") ||
            lower.startsWith("file:") ||
            lower.startsWith("content:") ||
            lower.startsWith("intent:") ||
            lower.startsWith("www.") ||
            bareDomainRegex.matches(token)
    }

    fun isLikelyPublicDomainHost(rawHost: String?): Boolean =
        normalizeHost(rawHost)?.let(::isAllowedPublicHost) == true

    private fun normalizeNetworkUrl(raw: String?, allowBareDomain: Boolean): String? {
        val token = sanitizeToken(raw)
        if (token.isBlank() || token.any { it == '\r' || it == '\n' || it == '\t' }) return null
        val lower = token.lowercase(Locale.US)
        if (lower.startsWith("https://")) return token
        if (lower.startsWith("http://") ||
            lower.startsWith("javascript:") ||
            lower.startsWith("data:") ||
            lower.startsWith("file:") ||
            lower.startsWith("content:") ||
            lower.startsWith("intent:")
        ) {
            return null
        }
        if (token.startsWith("//")) return "https:$token"
        if (allowBareDomain &&
            (token.startsWith("www.", ignoreCase = true) || bareDomainRegex.matches(token))
        ) {
            return "https://$token"
        }
        return null
    }

    private fun sanitizeToken(raw: String?): String =
        raw.orEmpty()
            .trim()
            .trim('\'', '"')
            .trimEnd('.', ',', ';', ')', ']', '}')

    private fun parseUri(raw: String): URI? =
        runCatching { URI(raw.trim()) }.getOrNull()

    private fun normalizeHost(rawHost: String?): String? {
        val raw = rawHost.orEmpty().trim().trimEnd('.')
        if (raw.isBlank()) return null
        return runCatching {
            IDN.toASCII(raw, IDN.USE_STD3_ASCII_RULES).lowercase(Locale.US)
        }.getOrNull()
    }

    private fun isAllowedPublicHost(host: String): Boolean {
        if (host.isBlank() || host.contains('_')) return false
        if (host in blockedHosts || blockedHosts.any { host.endsWith(".$it") }) return false
        if (isIpLiteral(host)) return false
        val labels = host.split('.').filter { it.isNotBlank() }
        if (labels.size < 2 || labels.any { !hostLabelRegex.matches(it) }) return false
        val tld = labels.last().lowercase(Locale.US)
        if (tld in blockedTlds || tld in fileLikeTlds) return false
        if (!tld.all { it in 'a'..'z' } || tld.length !in 2..24) return false
        return true
    }

    private fun isIpLiteral(host: String): Boolean {
        if (host.contains(':')) return true
        val parts = host.split('.')
        return parts.size == 4 && parts.all { part ->
            part.toIntOrNull()?.let { it in 0..255 } == true
        }
    }

    private fun isPlaceholderToken(value: String): Boolean {
        val lower = value.lowercase(Locale.US)
        return lower in setOf(
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
        ) || lower.contains("placeholder") || lower.contains("<") || lower.contains(">")
    }

    private fun isAllowedPhotoUrl(host: String, uri: URI): Boolean {
        val path = uri.path.orEmpty().lowercase(Locale.US)
        if (isIconOnlyMediaUrl(uri.toString())) return false
        if (host == "commons.wikimedia.org") {
            return path.contains("/wiki/special:filepath/")
        }
        if (host == "places.googleapis.com") {
            return path.contains("/media")
        }
        return allowedImageHosts.any { host == it || host.endsWith(".$it") } &&
            (
                path.endsWith(".png") ||
                    path.endsWith(".jpg") ||
                    path.endsWith(".jpeg") ||
                    path.endsWith(".webp") ||
                    path.contains("/media") ||
                    path.contains("/image") ||
                    path.contains("/images/") ||
                    path.contains("/img/") ||
                    path.contains("/photo") ||
                    path.contains("/photos/") ||
                    path.contains("/thumbnail") ||
                    path.contains("/thumb/")
                )
    }

    private fun isAllowedIconUrl(host: String, uri: URI): Boolean {
        val path = uri.path.orEmpty().lowercase(Locale.US)
        return allowedIconHosts.any { host == it || host.endsWith(".$it") } &&
            (isBootstrapIconUrl(host, path) || isWeatherIconUrl(host, path))
    }

    private fun isBootstrapIconUrl(host: String, path: String): Boolean =
        (host == "cdn.jsdelivr.net" || host == "unpkg.com") &&
            path.contains("/bootstrap-icons") &&
            path.contains("/icons/") &&
            path.endsWith(".svg")

    private fun isWeatherIconUrl(host: String, path: String): Boolean =
        host.contains("weatherapi.com") ||
            (host == "openweathermap.org" && path.contains("/img/wn/")) ||
            path.contains("/weather/64x64/") ||
            path.contains("/weather/128x128/") ||
            path.contains("/weather/icons/")
}
