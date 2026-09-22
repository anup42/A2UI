package com.samsung.genuicraft.sdk.provider

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiOutputRepetitionGuardTest {
    private val guard = GenUiOutputRepetitionGuard()

    @Test fun `same child reference stops exactly at twentieth occurrence`() {
        val nineteen = "<a2ui>\nroot=Column([" + List(19) { "av" }.joinToString(",")
        assertNull(guard.inspect(nineteen))

        val stop = guard.inspect("$nineteen,av")
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("'av'"))
    }

    @Test fun `sequential alphabetic id loop stops at twenty references`() {
        val nineteen = (1..19).map(::alphabeticId)
        assertNull(guard.inspect("<a2ui>\nroot=Column([${nineteen.joinToString(",")}"))

        val twenty = (1..20).map(::alphabeticId)
        val stop = guard.inspect("<a2ui>\nroot=Column([${twenty.joinToString(",")}")
        assertEquals(20, stop?.repeatLimit)
        assertTrue(stop?.detail.orEmpty().contains("'a' through 't'"))
    }

    @Test fun `sequential double-letter ids are detected as the same runaway pattern`() {
        val ids = (27..46).map(::alphabeticId)
        val stop = guard.inspect("<a2ui>\nroot=Column([${ids.joinToString(",")}")
        assertTrue(stop?.detail.orEmpty().contains("'aa' through 'at'"))
    }

    @Test fun `repeated state rows and ordinary component lists are not generation loops`() {
        val repeatedRows = List(25) { "{name:\"Same\",value:\"09:00\"}" }.joinToString(",")
        assertNull(guard.inspect("<a2ui>\n\$/={events:[$repeatedRows]}\nroot=Text(\"Ready\")"))
        val ordinaryIds = (1..30).map { "item_${it * 3}" }
        assertNull(guard.inspect("<a2ui>\nroot=Column([${ordinaryIds.joinToString(",")}"))
    }

    @Test fun `a completed document is never cancelled retroactively`() {
        val repeated = List(20) { "a" }.joinToString(",")
        assertNull(guard.inspect("<a2ui>\nroot=Column([$repeated])\n</a2ui>"))
    }

    private fun alphabeticId(value: Int): String {
        var remaining = value
        val result = StringBuilder()
        while (remaining > 0) {
            remaining -= 1
            result.append(('a'.code + remaining % 26).toChar())
            remaining /= 26
        }
        return result.reverse().toString()
    }
}
