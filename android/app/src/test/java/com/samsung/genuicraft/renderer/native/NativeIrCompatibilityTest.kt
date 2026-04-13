package com.samsung.genuicraft.renderer.native

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class NativeIrCompatibilityTest {

    @Test
    fun extractMessages_acceptsSingleLegacyMessageObject() {
        val payload = JsonParser.parseString(
            """
            {
              "version": "v0.9",
              "updateComponents": {
                "surfaceId": "surface_live",
                "components": []
              }
            }
            """.trimIndent()
        )

        val messages = NativePayloadParser.extractMessages(payload)
        assertNotNull(messages)
        assertEquals(1, messages!!.size())
        assertTrue(messages[0].asJsonObject.has("updateComponents"))
    }

    @Test
    fun readDynamicStringList_supportsLiteralStringListWrapper() {
        val wrapper = JsonParser.parseString(
            """
            {
              "literalStringList": ["one", "two", "three"]
            }
            """.trimIndent()
        )

        val values = NativeFormComponents.readDynamicStringList(wrapper)
        assertEquals(listOf("one", "two", "three"), values)
    }

    @Test
    fun readDynamicNumber_supportsLiteralNumberWrapper() {
        val wrapper = JsonParser.parseString(
            """
            {
              "literalNumber": 42.5
            }
            """.trimIndent()
        )

        val value = NativeFormComponents.readDynamicNumber(wrapper)
        assertEquals(42.5f, value ?: -1f, 0.0001f)
    }

    @Test
    fun parseActionEnvelope_supportsEventWrappedFunctionCall() {
        val action = JsonParser.parseString(
            """
            {
              "event": {
                "onClick": {
                  "functionCall": {
                    "call": "showMessage",
                    "args": {
                      "message": "Ready"
                    }
                  }
                }
              }
            }
            """.trimIndent()
        ).asJsonObject

        val parsed = NativeActionParsing.parseActionEnvelope(action)
        assertNotNull(parsed)
        assertEquals(NativeActionParsing.ComponentActionKind.ShowMessage, parsed!!.kind)
        assertEquals("Ready", parsed.value)
    }

    @Test
    fun resolveAssetUrl_resolvesParentRelativeAssetFromSiblingAssetsDir() {
        val root = Files.createTempDirectory("a2ui_native_assets").toFile()
        val dataDir = File(root, "data").apply { mkdirs() }
        val assetFile = File(root, "assets/icon.svg").apply {
            parentFile?.mkdirs()
            writeText("<svg></svg>")
        }
        try {
            val resolved = NativePayloadParser.resolveAssetUrl("../assets/icon.svg", dataDir)
            assertEquals(assetFile.toURI().toString(), resolved)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun resolveAssetUrl_resolvesParentRelativeAssetFromRunLocalAssetsDir() {
        val root = Files.createTempDirectory("a2ui_native_assets_local").toFile()
        val runDir = File(root, "run").apply { mkdirs() }
        val assetFile = File(runDir, "assets/icon.svg").apply {
            parentFile?.mkdirs()
            writeText("<svg></svg>")
        }
        try {
            val resolved = NativePayloadParser.resolveAssetUrl("../assets/icon.svg", runDir)
            assertEquals(assetFile.toURI().toString(), resolved)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun resolveAssetUrl_fallsBackToAppAssetPathWhenLocalAssetMissing() {
        val root = Files.createTempDirectory("a2ui_native_assets_missing").toFile()
        val dataDir = File(root, "data").apply { mkdirs() }
        try {
            val resolved = NativePayloadParser.resolveAssetUrl("../assets/missing.svg", dataDir)
            assertEquals("/assets/missing.svg", resolved)
        } finally {
            root.deleteRecursively()
        }
    }
}
