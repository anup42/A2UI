package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import com.google.gson.GsonBuilder
import com.samsung.genuicraft.sdk.internal.renderer.GenUiNativeRenderer
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

    @Test
    fun explicitGeneratedDslSalvageCompilesAndJvmRendersWithoutSourceFallback() {
        val corpus = requireNotNull(javaClass.getResourceAsStream("/genuicraft_bixby50.jsonl"))
            .bufferedReader(Charsets.UTF_8).useLines { lines ->
                lines.filter(String::isNotBlank).associate { line ->
                    val row = JsonParser.parseString(line).asJsonObject
                    row.get("id").asString to row.get("text").asString
                }
            }
        val run = File("../validation/20260921_e2b_mobile_full50/native")
        val rows = corpus.map { (id, source) ->
            val raw = File(run, "$id/output.express").readText(Charsets.UTF_8)
            val outputOnly = runCatching {
                GenUiCompiler.compileWithRepair(
                    input = raw,
                    allowSourceTextFallback = false,
                    allowGeneratedDslRepair = true,
                )
            }
            val sourceBound = runCatching {
                GenUiCompiler.compileWithRepair(
                    input = raw,
                    sourceText = source,
                    allowSourceTextFallback = false,
                    allowGeneratedDslRepair = true,
                )
            }
            val outcome = outputOnly.getOrNull()
            val rendered = outcome?.document?.let { GenUiNativeRenderer.render(it.a2uiJson, sourceDir = null) }
            if (outcome != null) {
                assertEquals(id, GenUiRepairKind.GENERATED_DSL_REPAIR, outcome.repairKind)
                assertEquals(id, outcome.document, GenUiCompiler.compile(outcome.document.a2uiJson))
                assertTrue(id, rendered != null && rendered.errorMessage == null && rendered.surfaces.isNotEmpty())
            }
            sourceBound.getOrNull()?.let { accepted ->
                assertTrue(id, accepted.repairKind != GenUiRepairKind.SOURCE_TEXT_FALLBACK)
                assertTrue(id, ContentIntegrity.check(GenUiRequest(source), accepted.document).isEmpty())
            }
            linkedMapOf<String, Any?>(
                "id" to id,
                "generatedDslRepaired" to outputOnly.isSuccess,
                "repairKind" to outcome?.repairKind?.name,
                "jvmRendered" to (rendered?.errorMessage == null && rendered?.surfaces?.isNotEmpty() == true),
                "jvmSurfaceCount" to rendered?.surfaces?.size,
                "sourceIntegrityEvaluated" to outputOnly.isSuccess,
                "sourceIntegrityAccepted" to sourceBound.isSuccess,
                "sourceTextFallbackUsed" to (sourceBound.getOrNull()?.repairKind == GenUiRepairKind.SOURCE_TEXT_FALLBACK),
                "repairDiagnostics" to outcome?.diagnostics,
                "repairFailure" to outputOnly.exceptionOrNull()?.message,
                "sourceIntegrityFailure" to sourceBound.exceptionOrNull()?.message,
            )
        }
        val repaired = rows.filter { it["generatedDslRepaired"] == true }.map { it.getValue("id") }
        val sourceAccepted = rows.filter { it["sourceIntegrityAccepted"] == true }.map { it.getValue("id") }
        val report = linkedMapOf<String, Any?>(
            "total" to rows.size,
            "generatedDslRepaired" to repaired.size,
            "generatedDslRejected" to rows.size - repaired.size,
            "jvmRendered" to rows.count { it["jvmRendered"] == true },
            "sourceIntegrityAccepted" to sourceAccepted.size,
            "sourceIntegrityRejected" to rows.count {
                it["sourceIntegrityEvaluated"] == true && it["sourceIntegrityAccepted"] != true
            },
            "sourceIntegrityNotEvaluated" to rows.count { it["sourceIntegrityEvaluated"] != true },
            "sourceTextFallbacks" to rows.count { it["sourceTextFallbackUsed"] == true },
            "repairedCases" to repaired,
            "sourceIntegrityAcceptedCases" to sourceAccepted,
            "cases" to rows,
        )
        val reportFile = File("build/reports/generated_dsl_repair_full50.json")
        requireNotNull(reportFile.parentFile).mkdirs()
        reportFile.writeText(GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create().toJson(report))
        println("GENERATED_DSL_REPAIR_REPORT=${reportFile.absolutePath}")
        println("GENERATED_DSL_REPAIRED=${repaired.size}:${repaired.joinToString(",")}")
        println("SOURCE_INTEGRITY_ACCEPTED=${sourceAccepted.size}:${sourceAccepted.joinToString(",")}")
        assertEquals(50, rows.size)
        assertEquals(
            listOf(
                "BXP-001", "BXP-002", "BXP-003", "BXP-004", "BXP-005", "BXP-006", "BXP-007", "BXP-008",
                "BXP-010", "BXP-011", "BXP-012", "BXP-013", "BXP-014", "BXP-015", "BXP-016", "BXP-017",
                "BXP-018", "BXP-019", "BXP-020", "BXP-021", "BXP-022", "BXP-023", "BXP-024", "BXP-025",
                "BXP-026", "BXP-027", "BXP-028", "BXP-029", "BXP-030", "BXP-031", "BXP-032", "BXP-033",
                "BXP-034", "BXP-035", "BXP-036", "BXP-037", "BXP-038", "BXP-040", "BXP-041", "BXP-042",
                "BXP-043", "BXP-044", "BXP-045", "BXP-046", "BXP-047", "BXP-048", "BXP-049", "BXP-050",
            ),
            repaired,
        )
        assertEquals(emptyList<String>(), sourceAccepted)
        assertEquals(repaired.size, rows.count { it["jvmRendered"] == true })
    }
}
