package com.samsung.genuicraft.renderer

import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.pipeline.FlatSpecContract
import com.samsung.genuicraft.pipeline.FlatSpecIngestMode
import com.samsung.genuicraft.pipeline.FlatSpecIngestResult
import com.samsung.genuicraft.pipeline.FlatSpecIngestor
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*

/**
 * Invariance net for renderer refactors.
 *
 * [describeFlatSpecRouting] is a deterministic, composition-free description of
 * every reachable element and the table-routing decision the renderer would
 * make. This test replays it over the bundled golden datasets at a compact and
 * an expanded width, and compares per-record hashes against a checked-in
 * golden.
 *
 * Behaviour-preserving work — moving declarations between files, splitting
 * `FlatSpecRenderer.kt`, deleting domain classifiers that `FlatSpecContract`
 * already stamps upstream — must leave every hash untouched.
 *
 * To regenerate after an intentional change:
 * ```
 * ./gradlew testDebugUnitTest --tests '*FlatSpecRoutingGoldenTest*' \
 *   -DflatRoutingGolden.record=true
 * ```
 * then copy `app/build/flat-routing-golden/flat_routing_golden.json` over
 * `app/src/test/resources/flat_routing_golden.json` and review the diff.
 * Full descriptions for every record are always written to
 * `app/build/flat-routing-golden/descriptions/` so a hash change can be
 * inspected.
 */
class FlatSpecRoutingGoldenTest {

    private companion object {
        const val GOLDEN_RESOURCE = "flat_routing_golden.json"
        const val RECORD_PROPERTY = "flatRoutingGolden.record"
        // The hybrid_r4 dataset is byte-identical to this one, so driving both
        // would double runtime for zero extra signal.
        val DATASETS = listOf(
            "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5_genui.jsonl"
        )
        val WIDTHS = listOf(360, 800)
        val WRAPPER_KEYS = listOf(
            "genui_json",
            "genui",
            "flat_spec",
            "flatSpec",
            "ir_json",
            "irJson",
            "stage3_json",
            "payload"
        )
    }

    private fun moduleDir(): File {
        var dir = File(System.getProperty("user.dir").orEmpty()).absoluteFile
        repeat(6) {
            if (File(dir, "src/main/assets").isDirectory) return dir
            dir = dir.parentFile ?: return@repeat
        }
        throw IllegalStateException(
            "Could not locate the app module from user.dir=${System.getProperty("user.dir")}"
        )
    }

    private fun sha256(value: String): String =
        MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(Charsets.UTF_8))
            .joinToString("") { byte -> "%02x".format(byte) }

    /**
     * Strips `propKeys` from a routing description.
     *
     * Canonicalization deliberately prunes props whose value already equals the
     * renderer default (`Stack.direction=vertical`, `Stack.gap=md`,
     * `Text.variant=body`, ... via `FlatSpecContract.removeDefaultProp`). That
     * changes which keys are *present* without changing what is rendered, so a
     * routing-neutrality check must compare routing, not key presence.
     */
    private fun routingOnly(description: String): String {
        val root = JsonParser.parseString(description).asJsonObject
        root.getAsJsonArray("elements").forEach { element ->
            (element as? JsonObject)?.remove("propKeys")
        }
        return GsonBuilder().setPrettyPrinting().serializeNulls().create().toJson(root)
    }

    /** Unwraps the flat spec from a dataset row, which nests it under a wrapper key. */
    private fun flatSpecFrom(row: JsonElement): JsonObject? {
        if (!row.isJsonObject) return null
        val obj = row.asJsonObject
        if (FlatSpecParser.isFlatSpec(obj)) return obj
        WRAPPER_KEYS.forEach { key ->
            val nested = obj.get(key) ?: return@forEach
            val candidate = when {
                nested.isJsonObject -> nested
                nested.isJsonPrimitive && nested.asJsonPrimitive.isString ->
                    runCatching { JsonParser.parseString(nested.asString) }.getOrNull()
                else -> null
            } ?: return@forEach
            if (candidate.isJsonObject && FlatSpecParser.isFlatSpec(candidate)) {
                return candidate.asJsonObject
            }
        }
        return null
    }

    @Test
    fun routingDescriptionMatchesCheckedInGolden() {
        val module = moduleDir()
        val assets = File(module, "src/main/assets")
        val outputDir = File(module, "build/flat-routing-golden")
        val descriptionDir = File(outputDir, "descriptions")
        outputDir.mkdirs()
        descriptionDir.mkdirs()

        val hashes = linkedMapOf<String, String>()
        var specCount = 0

        DATASETS.forEach { dataset ->
            val file = File(assets, dataset)
            assertTrue("Missing bundled dataset: ${file.path}", file.isFile)
            file.readLines()
                .filter { line -> line.isNotBlank() }
                .forEachIndexed { index, line ->
                    val row = runCatching { JsonParser.parseString(line) }.getOrNull() ?: return@forEachIndexed
                    val specJson = flatSpecFrom(row) ?: return@forEachIndexed
                    val spec = when (
                        val ingested = FlatSpecIngestor.ingest(specJson, FlatSpecIngestMode.STRICT)
                    ) {
                        is FlatSpecIngestResult.CanonicalFlatSpec -> ingested.canonicalSpec
                        is FlatSpecIngestResult.GenuineLegacyPayload,
                        is FlatSpecIngestResult.RejectedPayload ->
                            FlatSpecParser.parse(specJson) ?: return@forEachIndexed
                    }
                    specCount += 1
                    WIDTHS.forEach { width ->
                        val description = describeFlatSpecRouting(spec, width)
                        val key = "$dataset#%03d@%d".format(index, width)
                        hashes[key] = sha256(description)
                        File(descriptionDir, key.replace('#', '_').replace('@', '_') + ".json")
                            .writeText(description, Charsets.UTF_8)
                    }
                }
        }

        assertTrue(
            "No flat specs parsed from the bundled datasets; the harness would " +
                "silently pass. Check the wrapper keys.",
            specCount > 0
        )

        val actual = GsonBuilder().setPrettyPrinting().create().toJson(
            linkedMapOf<String, Any?>(
                "version" to FLAT_ROUTING_GOLDEN_VERSION,
                "specCount" to specCount,
                "widths" to WIDTHS,
                "hashes" to hashes
            )
        )
        val actualFile = File(outputDir, GOLDEN_RESOURCE)
        actualFile.writeText(actual, Charsets.UTF_8)

        if (System.getProperty(RECORD_PROPERTY) == "true") {
            println("Recorded routing golden to ${actualFile.path} ($specCount specs)")
            return
        }

        val expected = javaClass.classLoader?.getResourceAsStream(GOLDEN_RESOURCE)
            ?.bufferedReader(Charsets.UTF_8)
            ?.use { reader -> reader.readText() }
        assertTrue(
            "Missing $GOLDEN_RESOURCE. Record it with " +
                "-D$RECORD_PROPERTY=true and copy ${actualFile.path} into " +
                "src/test/resources/.",
            expected != null
        )

        val expectedHashes = JsonParser.parseString(expected)
            .asJsonObject
            .getAsJsonObject("hashes")
            .entrySet()
            .associate { (key, value) -> key to value.asString }

        val changed = hashes.filter { (key, hash) -> expectedHashes[key] != hash }.keys
        val added = hashes.keys - expectedHashes.keys
        val removed = expectedHashes.keys - hashes.keys
        assertEquals(
            "Routing changed for ${changed.size} record(s). " +
                "Added=$added Removed=$removed Changed=${changed.take(10)}. " +
                "Inspect ${descriptionDir.path} and, if intended, re-record.",
            emptySet<String>(),
            changed
        )
        assertEquals("Golden record set changed", expectedHashes.keys, hashes.keys)
    }

    /**
     * Measures the routing delta that canonicalizing at ingest (P0.4) would
     * introduce. Compatibility mode records this delta while rendering once.
     *
     * Result at the time of writing: canonicalization changes routing for 28 of
     * 50 bundled golden records (56 of 100 record/width pairs). The cause is
     * not the `props.domain` stamping the plan assumed — stored records already
     * carry `domain` 100% of the time — but `FlatSpecContract` resolving Table
     * `primaryColumn`, `highlightColumns` and `numericColumns` that the
     * renderer leaves unset when it parses the raw record. The canonicalized
     * rendering is richer, so this is an improvement rather than a regression,
     * but it moves screenshots and therefore v5.4 rendered-UX scores.
     *
     * Side-by-side descriptions for the first differing records
     * are written to `build/flat-routing-golden/canonicalization/`.
     */
    @Test
    fun compatibilityComparisonCapturesStoredRouteDeltas() {
        val assets = File(moduleDir(), "src/main/assets")
        val differing = mutableListOf<String>()
        var compared = 0

        DATASETS.forEach { dataset ->
            File(assets, dataset).readLines()
                .filter { line -> line.isNotBlank() }
                .forEachIndexed { index, line ->
                    val row = runCatching { JsonParser.parseString(line) }.getOrNull() ?: return@forEachIndexed
                    val raw = flatSpecFrom(row) ?: return@forEachIndexed
                    val rawSpec = FlatSpecParser.parse(raw) ?: return@forEachIndexed
                    val canonical = FlatSpecContract.normalizeToFlatSpec(raw).spec ?: raw
                    val canonicalSpec = FlatSpecParser.parse(canonical) ?: return@forEachIndexed
                    compared += 1
                    WIDTHS.forEach { width ->
                        val before = routingOnly(describeFlatSpecRouting(rawSpec, width))
                        val after = routingOnly(describeFlatSpecRouting(canonicalSpec, width))
                        if (before != after) {
                            val key = "$dataset#%03d@%d".format(index, width)
                            differing += key
                            if (differing.size <= 4) {
                                val dir = File(moduleDir(), "build/flat-routing-golden/canonicalization")
                                dir.mkdirs()
                                val stem = key.replace('#', '_').replace('@', '_')
                                File(dir, "$stem.raw.json").writeText(before, Charsets.UTF_8)
                                File(dir, "$stem.canonical.json").writeText(after, Charsets.UTF_8)
                            }
                        }
                    }
                }
        }

        assertTrue("No records compared", compared > 0)
        assertTrue(
            "The compatibility corpus should retain a measured raw/canonical route delta.",
            differing.isNotEmpty()
        )
    }

    @Test
    fun routingDescriptionIsDeterministic() {
        val spec = FlatSpecParser.parse(
            JsonParser.parseString(
                """
                {
                  "root": "main",
                  "state": { "rows": [ { "city": "Kyoto", "high": "24" } ] },
                  "elements": {
                    "main": {
                      "type": "Stack",
                      "props": { "direction": "vertical" },
                      "children": ["tbl"]
                    },
                    "tbl": {
                      "type": "Table",
                      "props": {
                        "domain": "weather",
                        "columns": [
                          { "key": "city", "label": "City" },
                          { "key": "high", "label": "High" }
                        ],
                        "statePath": "/rows"
                      },
                      "children": []
                    }
                  }
                }
                """.trimIndent()
            )
        )!!

        val first = describeFlatSpecRouting(spec, 360)
        val second = describeFlatSpecRouting(spec, 360)
        assertEquals(first, second)
        assertTrue("Table routing should be described", first.contains("\"directTable\""))
        // Width feeds the compact-screen breakpoint, so it must be visible.
        assertTrue(describeFlatSpecRouting(spec, 800).contains("\"compactScreen\": false"))
        assertTrue(first.contains("\"compactScreen\": true"))
    }

    @Test
    fun compactBreakpointMatchesRendererThreshold() {
        assertTrue(isFlatCompactScreenWidth(360))
        assertTrue(isFlatCompactScreenWidth(480))
        assertTrue(!isFlatCompactScreenWidth(481))
        assertTrue(!isFlatCompactScreenWidth(800))
    }
}
