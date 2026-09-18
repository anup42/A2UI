package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.*
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.*

/**
 * Guards DoD #5 for icons: an IR-declared icon *name* must render.
 *
 * Before the catalog, `Icon.props.name` was only treated as a URL, so a bare
 * name failed `SafeContentPolicy.sanitizeMediaUrl(..., ICON)` and `RenderIcon`
 * returned early — the element drew nothing.
 */
class FlatIconCatalogTest {

    @Test
    fun bareNamesResolve() {
        listOf("calendar_today", "location_on", "restaurant", "flight_takeoff", "star")
            .forEach { name ->
                assertNotNull("Expected a vector for '$name'", FlatIconCatalog.vectorFor(name))
            }
    }

    @Test
    fun separatorAndCaseVariantsResolveToTheSameVector() {
        val expected = FlatIconCatalog.vectorFor("calendar_today")
        listOf("calendar-today", "Calendar Today", "CALENDAR_TODAY", "  calendar_today  ")
            .forEach { variant ->
                assertEquals("variant '$variant'", expected, FlatIconCatalog.vectorFor(variant))
            }
    }

    @Test
    fun bootstrapIconNamesResolve() {
        // The generator's icon URLs come from the Bootstrap set, so a spec may
        // reasonably emit the bare Bootstrap name instead of the CDN URL.
        listOf("geo-alt", "moon-stars", "signpost-split", "cup-hot", "graph-up", "check2-circle")
            .forEach { name ->
                assertNotNull("Expected a vector for '$name'", FlatIconCatalog.vectorFor(name))
            }
    }

    @Test
    fun assetPathsAreNotHijacked() {
        // This is the regression that matters: the corpus emits paths like
        // `../assets/r_000001_01_1_building.svg`. If the catalog matched a path's
        // last segment, an asset named `star.svg` would silently render a
        // Material vector instead of the intended image.
        listOf(
            "../assets/r_000001_01_1_building.svg",
            "../assets/star.svg",
            "/assets/restaurant.svg",
            "assets/calendar_today.png",
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg",
            "file:///android_asset/star.svg"
        ).forEach { path ->
            assertNull("'$path' must fall through to the URL path", FlatIconCatalog.vectorFor(path))
        }
    }

    @Test
    fun unknownAndEmptyNamesReturnNull() {
        listOf(null, "", "   ", "definitely_not_an_icon").forEach { name ->
            assertNull(FlatIconCatalog.vectorFor(name))
        }
    }

    @Test
    fun catalogCoversAMeaningfulNumberOfNames() {
        // Guards against an accidental truncation of the table.
        assertTrue(
            "Catalog shrank unexpectedly: ${FlatIconCatalog.nameCount()}",
            FlatIconCatalog.nameCount() >= 150
        )
    }
}
