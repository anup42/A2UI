package com.samsung.genuicraft.sdk.internal.renderer

import com.google.gson.JsonParser
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
 * Guards the DoD #3 dispatch registry.
 *
 * The 25-way `when` in `RenderByType` had zero test coverage: every alias arm
 * (`barchart`, `pre`, `terminal`, ...) could be deleted without a single test
 * failing. Now that the aliases are data rather than control flow, they can be
 * enumerated and asserted directly.
 */
class FlatRenderRegistryTest {

    @Test
    fun everySchemaElementTypeHasARenderer() {
        // The schema is the contract; anything it permits must be renderable.
        val schemaTypes = requireNotNull(
            javaClass.classLoader?.getResourceAsStream("genui_flatspec.schema.json")
        ).reader(Charsets.UTF_8).use { JsonParser.parseReader(it).asJsonObject }
            .getAsJsonObject("\$defs")
            .getAsJsonObject("element")
            .getAsJsonObject("properties")
            .getAsJsonObject("type")
            .getAsJsonArray("enum")
            .map { entry -> entry.asString }

        val unrenderable = schemaTypes.filter { type -> flatRendererFor(type) == null }
        assertEquals(
            "Schema permits types the registry cannot render",
            emptyList<String>(),
            unrenderable
        )
    }

    @Test
    fun everyAliasResolvesToARegisteredCanonicalType() {
        FLAT_TYPE_ALIASES.forEach { (alias, canonical) ->
            assertTrue(
                "Alias '$alias' points at unregistered canonical '$canonical'",
                FLAT_ELEMENT_RENDERERS.containsKey(canonical)
            )
            assertEquals(canonical, canonicalFlatType(alias))
            assertNotNull("Alias '$alias' has no renderer", flatRendererFor(alias))
        }
    }

    @Test
    fun aliasesDoNotShadowCanonicalTypes() {
        // A canonical name appearing in the alias table would make dispatch
        // order-dependent.
        val clashes = FLAT_TYPE_ALIASES.keys.filter { it in FLAT_ELEMENT_RENDERERS }
        assertEquals(emptyList<String>(), clashes)
    }

    @Test
    fun typeLookupIsCaseAndWhitespaceInsensitive() {
        listOf("Stack", "  stack  ", "STACK", "DateTimeInput", "AudioPlayer", "BarChart")
            .forEach { raw ->
                assertNotNull("'$raw' should resolve", flatRendererFor(raw))
            }
    }

    @Test
    fun unknownTypeHasNoRenderer() {
        // The caller renders a visible placeholder for this case; see
        // FlatRendererDiagnosticsTest.
        listOf("Hologram", "", "   ", "stackk").forEach { raw ->
            assertNull("'$raw' should not resolve", flatRendererFor(raw))
            assertNull(canonicalFlatType(raw))
        }
    }

    @Test
    fun registryCoversTheDocumentedCatalogSize() {
        // 25 canonical entries: the 23 distinct schema types plus the Row/Column
        // layout aliases the renderer handles natively with a pinned direction.
        assertEquals(25, FLAT_ELEMENT_RENDERERS.size)
    }

    @Test
    fun rowAndColumnPinLayoutDirection() {
        val base = FlatRenderContext(
            elementId = "e",
            type = "Row",
            props = mapOf("gap" to "sm"),
            children = emptyList(),
            onMap = null,
            elements = emptyMap(),
            state = emptyMap(),
            repeatScope = null,
            repeatedChildScopes = null,
            onOpenUrl = {},
            onSetState = { _, _ -> },
            onAction = { _, _ -> 0 },
            activePath = emptySet()
        )
        assertEquals("horizontal", base.withProp("direction", "horizontal").props["direction"])
        // withProp must not drop existing props.
        assertEquals("sm", base.withProp("direction", "horizontal").props["gap"])
    }
}
