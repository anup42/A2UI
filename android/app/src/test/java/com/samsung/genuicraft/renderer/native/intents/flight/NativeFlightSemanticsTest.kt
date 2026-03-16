package com.samsung.genuicraft.renderer.native.intents.flight

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeFlightSemanticsTest {
    @Test
    fun looksLikeFareValue_handlesInrAndDoesNotTreatTimeAsFare() {
        assertTrue(NativeFlightSemantics.looksLikeFareValue("INR 6,200"))
        assertTrue(NativeFlightSemantics.looksLikeFareValue("Rs 6200"))
        assertFalse(NativeFlightSemantics.looksLikeFareValue("06:15 AM"))
    }

    @Test
    fun splitFareDisplay_normalizesInrPrefix() {
        val (amount, suffix) = NativeFlightSemantics.splitFareDisplay("INR 6,212 /adult")
        assertEquals("\u20B96,212", amount)
        assertEquals("/adult", suffix)
    }
}
