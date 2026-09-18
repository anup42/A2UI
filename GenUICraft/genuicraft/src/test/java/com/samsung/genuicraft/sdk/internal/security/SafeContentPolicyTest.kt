package com.samsung.genuicraft.sdk.internal.security

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SafeContentPolicyTest {

    @Test
    fun actionUrls_acceptPublicHttpAndHttpsPreservePunctuationAndNormalizeBareDomains() {
        assertEquals(
            "https://www.google.com/search?q=weather",
            SafeContentPolicy.sanitizeActionUrl("https://www.google.com/search?q=weather")
        )
        assertEquals(
            "http://www.who.int/health-topics/",
            SafeContentPolicy.sanitizeActionUrl("http://www.who.int/health-topics/")
        )
        val punctuationUrl = "https://www.who.int/reports/(weekly).?tags=health,science;"
        assertEquals(punctuationUrl, SafeContentPolicy.sanitizeActionUrl(punctuationUrl))
        assertEquals(
            "https://www.makemytrip.com/flights/",
            SafeContentPolicy.sanitizeActionUrl("www.makemytrip.com/flights/")
        )
    }

    @Test
    fun actionUrls_rejectUnsafeSchemesPrivateHostsAndPlaceholders() {
        val unsafe = listOf(
            "http://example.org/page",
            "javascript:alert(1)",
            "data:text/html,<b>x</b>",
            "file:///sdcard/secret.txt",
            "content://com.android.contacts/1",
            "intent://scan/#Intent;scheme=zxing;end",
            "https://localhost/admin",
            "https://127.0.0.1/admin",
            "https://192.168.1.12/admin",
            "https://10.0.2.2/admin",
            "https://service.local/path",
            "https://example.com/path",
            "https://placeholder.com/image.jpg",
            "not a url"
        )

        unsafe.forEach { candidate ->
            assertNull(candidate, SafeContentPolicy.sanitizeActionUrl(candidate))
            assertFalse(candidate, SafeContentPolicy.isSafeActionUrl(candidate))
        }
    }

    @Test
    fun mediaPolicy_acceptsCuratedPhotoSourcesAndGeneratedVisualsForImages() {
        val safeImages = listOf(
            "../assets/bengaluru.jpg",
            "genuicraft://visual/travel?title=Bengaluru",
            "https://commons.wikimedia.org/wiki/Special:FilePath/Vidhana_Soudha.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/1/11/Vidhana_Soudha.jpg",
            "https://places.googleapis.com/v1/places/abc/photos/def/media?key=redacted",
            "https://images.unsplash.com/photo-1234567890"
        )

        safeImages.forEach { url ->
            assertTrue(url, SafeContentPolicy.isSafeMediaUrl(url, SafeContentPolicy.MediaKind.IMAGE))
        }
    }

    @Test
    fun mediaPolicy_acceptsBoundedNewsPhotoAndSourceIconHosts() {
        val safeNewsImages = listOf(
            "https://data1.ibtimes.co.in/en/full/833180/bengaluru-news.jpg",
            "https://cache.careers360.mobi/media/article_images/2026/6/7/colleges.jpg",
            "https://img.etimg.com/thumb/msid-131569932,width-1200,height-900/articleshow.jpg",
            "https://media.assettype.com/freepressjournal/2022-07/lpg.webp",
            "https://staticprintenglish.theprint.in/wp-content/uploads/2026/06/news.jpg",
            "https://www.globalindian.com/wp-content/uploads/2026/06/story.jpg",
            "https://www.newsx.com/wp-content/uploads/2026/06/story.jpg",
            "https://n.bytvi.com/ibtimes.png"
        )

        safeNewsImages.forEach { url ->
            assertTrue(url, SafeContentPolicy.isSafeMediaUrl(url, SafeContentPolicy.MediaKind.IMAGE))
        }
    }

    @Test
    fun mediaPolicy_allowsIconHostsOnlyForIcons() {
        val bootstrapIcon = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg"
        val weatherIcon = "https://cdn.weatherapi.com/weather/64x64/day/116.png"

        assertTrue(SafeContentPolicy.isSafeMediaUrl(bootstrapIcon, SafeContentPolicy.MediaKind.ICON))
        assertTrue(SafeContentPolicy.isSafeMediaUrl(weatherIcon, SafeContentPolicy.MediaKind.ICON))
        // An icon asset must not satisfy a photo slot. FlatSpecContract and
        // PipelineMediaSanitizer both rely on this to route icon URLs to Icon
        // elements instead of Image elements.
        assertFalse(SafeContentPolicy.isSafeMediaUrl(bootstrapIcon, SafeContentPolicy.MediaKind.IMAGE))
        assertFalse(SafeContentPolicy.isSafeMediaUrl(weatherIcon, SafeContentPolicy.MediaKind.IMAGE))
        assertTrue(SafeContentPolicy.isIconOnlyMediaUrl(bootstrapIcon))
    }

    @Test
    fun mediaPolicy_rejectsPlaceholderAndMalformedImageUrls() {
        val unsafeImages = listOf(
            "http://upload.wikimedia.org/wikipedia/commons/a/a0/Test.jpg",
            "not a url"
        )

        unsafeImages.forEach { url ->
            assertFalse(url, SafeContentPolicy.isSafeMediaUrl(url, SafeContentPolicy.MediaKind.IMAGE))
        }
    }
}
