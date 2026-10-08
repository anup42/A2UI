package com.samsung.genuicraft.sdk

import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiCanonicalGraph
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import java.nio.file.Files
import java.nio.file.Path
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/** Optional read-only replay of saved device output, without duplicating the corpus as fixtures. */
class GenUiExpressStreamReplayTest {
    @Test fun replaySavedBixby50CorpusWithThirtyTwoCharacterChunks() {
        val directory = setting("genuicraft.streamReplayDir", "GENUICRAFT_STREAM_REPLAY_DIR")
        assumeTrue("Set GENUICRAFT_STREAM_REPLAY_DIR to replay saved device output.", !directory.isNullOrBlank())
        val mtp = setting("genuicraft.streamReplayMtp", "GENUICRAFT_STREAM_REPLAY_MTP") ?: "on"
        require(mtp in setOf("on", "off", "any")) { "Replay MTP selection must be on, off, or any." }
        val paths = Files.walk(Path.of(directory!!)).use { stream ->
            stream.iterator().asSequence().filter { Files.isRegularFile(it) }
                .filter { it.fileName.toString() in setOf("output.express", "attempt_1_raw.express") }
                .filter { mtp == "any" || it.toString().replace('\\', '/').lowercase().contains("mtp_$mtp/") }
                .sortedBy { it.toString() }.toList()
        }
        val cases = linkedMapOf<String, Path>()
        paths.forEach { path ->
            val id = generateSequence(path.parent) { it.parent }.map { it.fileName?.toString().orEmpty() }
                .firstOrNull { Regex("BXP-\\d{3}").matches(it) }
            if (id != null) cases.putIfAbsent(id, path)
        }
        assertEquals("Select one complete saved Bixby50 MTP corpus.", 50, cases.size)
        assertEquals((1..50).map { "BXP-%03d".format(it) }.toSet(), cases.keys)
        val results = JsonArray()
        cases.toSortedMap().forEach { (id, path) ->
            val raw = path.toFile().readText(Charsets.UTF_8)
            val baseline = baseline(raw)
            val compiler = GenUiExpressStreamCompiler()
            var firstPreview: Int? = null
            var previews = 0
            var offset = 0
            while (offset < raw.length) {
                offset = (offset + 32).coerceAtMost(raw.length)
                compiler.accept(raw.substring(0, offset))?.let { preview ->
                    val graph = A2uiWireCodec.decode(JsonParser.parseString(preview.document.a2uiJson))
                    assertTrue("$id preview must validate.", A2uiCanonicalGraph.validate(graph).isValid)
                    assertEquals(graph.getAsJsonObject("elements").size(), preview.readyComponentCount)
                    if (firstPreview == null) firstPreview = offset
                    previews += 1
                }
            }
            // Preview processing cannot change the authoritative final compilation or raw bytes.
            assertEquals("$id final behavior changed during replay.", baseline, baseline(raw))
            val closing = raw.lastIndexOf("</a2ui>").let { if (it < 0) raw.length else it + 7 }
            results.add(JsonObject().apply {
                addProperty("case_id", id)
                addProperty("source", path.toString())
                addProperty("raw_characters", raw.length)
                addProperty("preview_count", previews)
                firstPreview?.let {
                    addProperty("first_preview_characters", it)
                    addProperty("first_preview_character_fraction", it.toDouble() / raw.length)
                }
                addProperty("preview_before_document_completion", firstPreview?.let { it < closing } == true)
                addProperty("baseline_accepted", baseline.accepted)
                baseline.repairKind?.let { addProperty("baseline_repair_kind", it) }
            })
        }
        val early = results.count { it.asJsonObject.get("preview_before_document_completion").asBoolean }
        val fractions = results.mapNotNull { it.asJsonObject.get("first_preview_character_fraction")?.asDouble }.sorted()
        val report = JsonObject().apply {
            addProperty("schema_version", 1)
            addProperty("evidence", "Saved raw-output replay; character readiness only, not device latency or painted-frame timing.")
            addProperty("chunk_characters", 32)
            addProperty("mtp_selection", mtp)
            addProperty("case_count", cases.size)
            addProperty("cases_with_preview_before_document_completion", early)
            addProperty("cases_without_any_preview", results.count { it.asJsonObject.get("preview_count").asInt == 0 })
            if (fractions.isNotEmpty()) {
                addProperty("mean_first_preview_character_fraction", fractions.average())
                addProperty("median_first_preview_character_fraction",
                    (fractions[(fractions.size - 1) / 2] + fractions[fractions.size / 2]) / 2)
            }
            add("cases", results)
        }
        val json = GsonBuilder().setPrettyPrinting().create().toJson(report)
        val destination = setting("genuicraft.streamReplayReport", "GENUICRAFT_STREAM_REPLAY_REPORT")
        if (!destination.isNullOrBlank()) {
            val output = Path.of(destination)
            output.toAbsolutePath().parent?.let { Files.createDirectories(it) }
            output.toFile().writeText(json, Charsets.UTF_8)
        }
        println("Saved Express replay: $early/${cases.size} cases emitted a nonempty preview before document completion; not a latency measurement.")
    }

    private data class Baseline(val accepted: Boolean, val document: GenUiDocument?, val repairKind: String?)

    private fun baseline(raw: String): Baseline = runCatching {
        GenUiCompiler.compileWithRepair(raw, allowSourceTextFallback = false, allowGeneratedDslRepair = true)
    }.fold(
        onSuccess = { Baseline(true, it.document, it.repairKind.name) },
        onFailure = { Baseline(false, null, null) },
    )

    private fun setting(property: String, environment: String): String? =
        System.getProperty(property)?.takeIf(String::isNotBlank) ?: System.getenv(environment)?.takeIf(String::isNotBlank)
}
