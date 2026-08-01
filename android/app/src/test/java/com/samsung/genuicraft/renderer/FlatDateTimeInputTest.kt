package com.samsung.genuicraft.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.compose.*

/**
 * Guards DoD #5 for `DateTimeInput`, which was a bare `OutlinedTextField`
 * labelled "Date/Time" with no picker, no format and no min/max handling.
 */
class FlatDateTimeInputTest {

    @Test
    fun defaultsToDateOnly() {
        assertEquals(FlatDateTimeMode.DATE, flatDateTimeMode(emptyMap()))
    }

    @Test
    fun declaredModeIsHonoured() {
        assertEquals(FlatDateTimeMode.DATE, flatDateTimeMode(mapOf("mode" to "date")))
        assertEquals(FlatDateTimeMode.TIME, flatDateTimeMode(mapOf("mode" to "time")))
        assertEquals(FlatDateTimeMode.DATE_TIME, flatDateTimeMode(mapOf("mode" to "datetime")))
        assertEquals(FlatDateTimeMode.DATE_TIME, flatDateTimeMode(mapOf("mode" to "date_time")))
    }

    @Test
    fun v09EnableFlagsAreHonoured() {
        // The A2UI v0.9 catalog spells this as enableDate/enableTime, so accept
        // both rather than forcing the generator onto one spelling.
        assertEquals(
            FlatDateTimeMode.DATE_TIME,
            flatDateTimeMode(mapOf("enableDate" to true, "enableTime" to true))
        )
        assertEquals(
            FlatDateTimeMode.TIME,
            flatDateTimeMode(mapOf("enableTime" to true))
        )
        assertEquals(
            FlatDateTimeMode.DATE,
            flatDateTimeMode(mapOf("enableDate" to true))
        )
    }

    @Test
    fun modeIsReadFromAlternatePropSpellings() {
        assertEquals(FlatDateTimeMode.TIME, flatDateTimeMode(mapOf("variant" to "time")))
        assertEquals(FlatDateTimeMode.TIME, flatDateTimeMode(mapOf("inputType" to "TIME")))
        assertEquals(FlatDateTimeMode.DATE_TIME, flatDateTimeMode(mapOf("format" to "dateTime")))
    }

    @Test
    fun placeholdersDocumentTheExpectedFormat() {
        assertEquals("YYYY-MM-DD", flatDateTimePlaceholder(FlatDateTimeMode.DATE))
        assertEquals("HH:MM", flatDateTimePlaceholder(FlatDateTimeMode.TIME))
        assertEquals("YYYY-MM-DD HH:MM", flatDateTimePlaceholder(FlatDateTimeMode.DATE_TIME))
    }

    @Test
    fun pickedValuesAreIso8601SoValidationCanParseThem() {
        // FlatFormValidation compares these against declared min/max, so the
        // format has to be stable and sortable.
        assertEquals("2026-07-31", formatFlatDate(2026, 7, 31))
        assertEquals("2026-01-05", formatFlatDate(2026, 1, 5))
        assertEquals("09:05", formatFlatTime(9, 5))
        assertEquals("23:59", formatFlatTime(23, 59))
    }

    @Test
    fun existingDatePrefixIsReused() {
        // Re-opening a datetime picker should not lose the date already chosen.
        assertEquals("2026-07-31", flatDatePrefixOf("2026-07-31 14:30"))
        assertEquals("2026-07-31", flatDatePrefixOf("2026-07-31"))
        assertNull(flatDatePrefixOf("14:30"))
        assertNull(flatDatePrefixOf(""))
    }

    @Test
    fun mediaDurationLabelFormatsSecondsAndPassesThroughStrings() {
        assertEquals("2 min 5s", flatMediaDurationLabel(mapOf("duration" to 125.0)))
        assertEquals("45s", flatMediaDurationLabel(mapOf("duration" to 45.0)))
        assertEquals("3:20", flatMediaDurationLabel(mapOf("duration" to "3:20")))
        assertNull(flatMediaDurationLabel(emptyMap()))
        assertNull(flatMediaDurationLabel(mapOf("duration" to 0.0)))
    }

    @Test
    fun posterPropIsReadFromAnyDeclaredSpelling() {
        assertEquals(
            "https://example.invalid/p.jpg",
            firstNonBlankStringProp(
                mapOf("thumbnail" to "https://example.invalid/p.jpg"),
                "poster",
                "posterUrl",
                "thumbnail"
            )
        )
        assertNull(firstNonBlankStringProp(mapOf("poster" to "  "), "poster"))
    }
}
