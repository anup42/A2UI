package com.samsung.genuicraft.renderer.native

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

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
}
