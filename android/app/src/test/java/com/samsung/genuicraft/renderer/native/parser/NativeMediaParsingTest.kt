package com.samsung.genuicraft.renderer.native.parser

import com.samsung.genuicraft.renderer.native.media.NativeMediaVisualUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeMediaParsingTest {

    @Test
    fun collectMediaEntries_treatsImageIconUrlAsIconLike() {
        val lines = listOf(
            "Media: Image=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/airplane.svg",
            "Direct flights available"
        )

        val parsed = NativeMediaParsing.collectMediaEntries(
            lines = lines,
            startIndex = 0,
            stripLeadingBulletMarker = NativeStructureParsing::stripLeadingBulletMarker,
            sanitizeUrlToken = NativeStructureParsing::sanitizeUrlToken,
            looksLikeImagePath = { value, _ ->
                NativeMediaVisualUtils.looksLikeImagePath(value)
            },
            looksLikeCompactIconUrl = NativeMediaVisualUtils::looksLikeCompactIconUrl
        )

        requireNotNull(parsed)
        val (entries, nextIndex) = parsed
        assertEquals(2, entries.size)
        assertTrue(entries.all { it.iconLike })
        assertEquals(1, nextIndex)
    }
}
