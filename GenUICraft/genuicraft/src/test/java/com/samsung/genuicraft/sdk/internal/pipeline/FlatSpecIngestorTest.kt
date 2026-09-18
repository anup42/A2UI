package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.renderer.FLAT_ROUTING_GOLDEN_VERSION
import com.samsung.genuicraft.sdk.internal.renderer.buildRepeatScopes
import com.samsung.genuicraft.sdk.internal.renderer.flat.capability.GeneratedRendererCapabilities
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.FLAT_TYPE_ALIASES
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatKeyIssue
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.FlatWatchCascadeTransaction
import com.samsung.genuicraft.sdk.internal.renderer.flat.runtime.FlatWatchTrigger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatSpecIngestorTest {

    @Test
    fun generatedRegistryAndSharedManifestStayInParity() {
        assertEquals("2.0.0", GeneratedRendererCapabilities.VERSION)
        assertEquals(
            GeneratedRendererCapabilities.typeAliases
                .filter { (alias, canonical) -> alias != canonical }
                .keys +
                GeneratedRendererCapabilities.compatibilityTypeDirections.keys,
            FLAT_TYPE_ALIASES.keys
        )
        assertEquals(12, GeneratedRendererCapabilities.tableDomains.size)
        assertTrue("product" in GeneratedRendererCapabilities.tableDomains)
        assertEquals(setOf("bar", "column"), GeneratedRendererCapabilities.chartSubtypes)
        assertEquals("2.0.0", FLAT_ROUTING_GOLDEN_VERSION)
    }

    @Test
    fun strictDisablesEnglishHeaderInferenceWhileCompatibilityRetainsIt() {
        val input = JsonParser.parseString(
            """
            {"root":"root","state":{"rows":[{"day":"Mon","condition":"Sunny","high":"31°C"}]},"elements":{
              "root":{"type":"Table","props":{"columns":[{"key":"day","label":"Day"},{"key":"condition","label":"Condition"},{"key":"high","label":"High"}],"statePath":"/rows"},"children":[]}
            }}
            """.trimIndent()
        )
        val strict = FlatSpecIngestor.ingest(input, FlatSpecIngestMode.STRICT)
            as FlatSpecIngestResult.CanonicalFlatSpec
        val compatibility = FlatSpecIngestor.ingest(input, FlatSpecIngestMode.COMPATIBILITY)
            as FlatSpecIngestResult.CanonicalFlatSpec

        assertEquals("generic", strict.canonicalJson.rootProps()["domain"].asString)
        assertEquals("weather", compatibility.canonicalJson.rootProps()["domain"].asString)
        assertNotNull(compatibility.routingComparison)
    }

    @Test
    fun invalidAndUnsafeActionsAreRejectedBeforeRuntime() {
        val invalid = JsonParser.parseString(
            """
            {"root":"root","state":{},"elements":{"root":{"type":"Button","props":{"label":"Open"},"children":[],"on":{"press":{"action":"openUrl","params":{"url":"javascript:alert(1)"},"confirm":{"title":"No"}}}}}}
            """.trimIndent()
        )
        val result = FlatSpecIngestor.ingest(invalid, FlatSpecIngestMode.STRICT)
        assertTrue(result is FlatSpecIngestResult.RejectedPayload)
        val message = (result as FlatSpecIngestResult.RejectedPayload).diagnostics.single().message
        assertTrue(message.contains("unsupported action fields") || message.contains("unsafe URL"))
    }

    @Test
    fun repeatKeysUseStableValuesAndDiagnoseMissingOrDuplicateValues() {
        val scopes = buildRepeatScopes(
            RepeatConfig(statePath = "/rows", key = "id"),
            mapOf(
                "rows" to listOf(
                    mapOf("id" to "a"),
                    mapOf("id" to "a"),
                    mapOf("name" to "missing"),
                    mapOf("id" to "b")
                )
            )
        ).orEmpty()
        assertEquals(listOf("id:a", "index:1", "index:2", "id:b"), scopes.map { it.stableKey })
        assertEquals(RepeatKeyIssue.DUPLICATE, scopes[1].keyIssue)
        assertEquals(RepeatKeyIssue.MISSING, scopes[2].keyIssue)
    }

    @Test
    fun watchCascadeSharesBudgetAndRejectsVisitedFingerprints() {
        val transaction = FlatWatchCascadeTransaction(maxActions = 2, idleBoundaryNanos = Long.MAX_VALUE)
        val first = FlatWatchTrigger("first", "a", "/flag", true, emptyMap<String, Any?>())
        val second = FlatWatchTrigger("second", "b", "/other", 1, emptyMap<String, Any?>())
        transaction.beginEmission(1)
        assertTrue(transaction.enter(first))
        transaction.consume(1)
        assertFalse(transaction.enter(first))
        assertTrue(transaction.enter(second))
        transaction.consume(1)
        assertFalse(
            transaction.enter(
                FlatWatchTrigger("third", "c", "/third", 2, emptyMap<String, Any?>())
            )
        )
        assertEquals(0, transaction.remainingActions)
    }

    @Test
    fun all32FixturesAreStrictCanonicalWithoutLegacyFallback() {
        val fixture = requireNotNull(
            javaClass.classLoader?.getResourceAsStream("intent_flat_specs_v2.json")
        )
        val payload = fixture.reader(Charsets.UTF_8).use { JsonParser.parseReader(it).asJsonObject }
        val fixtures = payload.getAsJsonArray("fixtures")
        assertEquals(32, fixtures.size())
        fixtures.forEach { fixtureValue ->
            val fixture = fixtureValue.asJsonObject
            val result = FlatSpecIngestor.ingest(
                fixture.get("spec"),
                FlatSpecIngestMode.STRICT
            )
            assertTrue(
                "${fixture.get("intent").asString}: $result",
                result is FlatSpecIngestResult.CanonicalFlatSpec
            )
            assertFalse(result is FlatSpecIngestResult.GenuineLegacyPayload)
        }
    }

    private fun com.google.gson.JsonObject.rootProps() =
        getAsJsonObject("elements").getAsJsonObject(get("root").asString).getAsJsonObject("props")
}
