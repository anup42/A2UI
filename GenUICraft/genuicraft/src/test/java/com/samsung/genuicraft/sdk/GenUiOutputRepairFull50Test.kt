package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiOutputRepairFull50Test {
    @Test
    fun capturedMalformedOutputsRemainRejectedAndAllSourcesHaveLosslessFallbacks() {
        val corpus = requireNotNull(javaClass.getResourceAsStream("/genuicraft_bixby50.jsonl"))
            .bufferedReader(Charsets.UTF_8).useLines { lines ->
                lines.filter(String::isNotBlank).associate { line ->
                    val row = JsonParser.parseString(line).asJsonObject
                    row.get("id").asString to row.get("text").asString
                }
            }
        val run = File("../validation/20260921_e2b_mobile_full50/native")
        assertTrue("Missing captured full50 run: ${run.absolutePath}", run.isDirectory)
        assertEquals(50, corpus.size)

        val outcomes = corpus.map { (id, source) ->
            val raw = File(run, "$id/output.express").readText(Charsets.UTF_8)
            assertTrue(id, runCatching { GenUiCompiler.compile(raw) }.isFailure)
            assertTrue(id, runCatching { GenUiCompiler.compileWithRepair(raw) }.isFailure)
            id to GenUiCompiler.compileWithRepair(raw, source)
        }

        assertEquals(50, outcomes.size)
        assertEquals(emptyList<String>(), outcomes.filter { it.second.repairKind != GenUiRepairKind.SOURCE_TEXT_FALLBACK }.map { it.first })
        outcomes.forEach { (id, outcome) ->
            assertTrue(id, ContentIntegrity.check(GenUiRequest(corpus.getValue(id)), outcome.document).isEmpty())
            assertTrue(id, outcome.diagnostics.any { it.contains("source blocks") })
        }
    }
}
