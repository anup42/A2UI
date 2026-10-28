package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiCapturedArrayCloseRepairTest {
    @Test fun freshPhoneComparisonRetainsAllCompleteTableRows() {
        val raw = resource("/captured_20261028_phones_malformed_table.express")
        val source = JsonParser.parseString(resource("/captured_20261028_phones_source.json"))
            .asJsonObject.get("text").asString
        assertThrows(IllegalArgumentException::class.java) { GenUiCompiler.compile(raw) }

        val repaired = GenUiCompiler.compileWithRepair(
            input = raw,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )

        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, repaired.repairKind)
        assertTrue(repaired.diagnostics.any { it.contains("mismatched unquoted Table closing delimiters") })
        val graph = A2uiExpressCodec.decode(repaired.document.express)
        val tables = graph.getAsJsonObject("elements").entrySet()
            .map { it.value.asJsonObject }
            .filter { it.get("type").asString == "Table" }
        assertEquals(1, tables.size)
        val props = tables.single().getAsJsonObject("props")
        val rows = props.getAsJsonArray("rows")
        assertEquals(4, props.getAsJsonArray("columns").size())
        assertEquals(3, rows.size())
        assertTrue(rows.all { it.asJsonArray.size() == 4 })
        assertEquals(listOf("Samsung Galaxy S25", "iPhone 16", "Google Pixel 9"),
            rows.map { it.asJsonArray[0].asString })
        assertEquals("Starts around ₹79,999; some India listings show lower sale prices[12]",
            rows[2].asJsonArray[3].asString)

        assertTrue(ContentIntegrity.check(GenUiRequest(source), repaired.document).isEmpty())
        val sourceBound = GenUiCompiler.compileWithRepair(raw, source, false, true)
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, sourceBound.repairKind)
        assertEquals(repaired.document, sourceBound.document)
    }

    @Test fun capturedBatteryComparisonRetainsAllCompleteTableRows() {
        // Exact captured Fold8 BXP-047 model output. The extra unquoted `]` follows three
        // complete rows; the malformed citation in the final quoted cell is model content.
        val raw = resource("/captured_bxp047_malformed_table.express")
        val source = JsonParser.parseString(resource("/captured_bxp047_source.json"))
            .asJsonObject.get("text").asString

        val repaired = GenUiCompiler.compileWithRepair(
            input = raw,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )

        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, repaired.repairKind)
        assertTrue(repaired.diagnostics.any { it.contains("excess unquoted closing array bracket") })
        val graph = A2uiExpressCodec.decode(repaired.document.express)
        val tables = graph.getAsJsonObject("elements").entrySet()
            .map { it.value.asJsonObject }
            .filter { it.get("type").asString == "Table" }
        assertEquals(1, tables.size)
        val props = tables.single().getAsJsonObject("props")
        val columns = props.getAsJsonArray("columns")
        val rows = props.getAsJsonArray("rows")
        assertEquals(6, columns.size())
        assertEquals(3, rows.size())
        assertTrue(rows.all { it.asJsonArray.size() == columns.size() })
        assertEquals(listOf("LFP", "NMC", "Sodium-ion"), rows.map { it.asJsonArray[0].asString })
        assertEquals("Emerging; gaining traction, but still less mature than LFP and NMC[26",
            rows[2].asJsonArray[5].asString)
        assertFalse(repaired.document.express.contains("NMC[26]"))

        // The missing citation bracket is not invented. Strict conversion must still reject the
        // incomplete generated wording when the original source is supplied.
        assertTrue(ContentIntegrity.check(GenUiRequest(source), repaired.document).isNotEmpty())
        assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compileWithRepair(raw, source, false, true)
        }
    }

    @Test fun quotedClosingBracketIsVisibleContentNotARepairTarget() {
        val raw = "<a2ui>\nroot=Text(\"Keep ] in the answer\")\n</a2ui>"
        val outcome = GenUiCompiler.compileWithRepair(
            input = raw,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )

        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)
        assertTrue(outcome.document.express.contains("Keep ] in the answer"))
    }

    private fun resource(path: String): String = requireNotNull(javaClass.getResourceAsStream(path))
        .bufferedReader(Charsets.UTF_8).use { it.readText() }
}
