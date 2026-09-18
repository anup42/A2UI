package com.samsung.genuicraft.sdk

import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec

/** Deterministically compiles strict A2UI Express or A2UI v0.9 JSON. */
object GenUiCompiler {
    @JvmStatic
    fun compile(input: String): GenUiDocument {
        val content = input.trim()
        require(content.isNotEmpty()) {
            "GenUI input is empty; expected one <a2ui> document or A2UI v0.9 JSON."
        }
        require(content.startsWith("<a2ui>") || content.startsWith("{") || content.startsWith("[")) {
            "GenUI input contains prose or an unsupported prefix; expected <a2ui>, {, or [."
        }
        if (content.startsWith("<a2ui>")) {
            require(content.endsWith("</a2ui>")) {
                "GenUI Express input must end exactly with </a2ui>; trailing prose or an incomplete document is not allowed."
            }
        }

        return try {
            val decoded = GenUiIrCodec.decode(JsonPrimitive(content))
            GenUiDocument(
                express = A2uiExpressCodec.encode(decoded.canonicalGraph),
                a2uiJson = A2uiWireCodec.encode(decoded.canonicalGraph).toString(),
            )
        } catch (error: RuntimeException) {
            throw IllegalArgumentException(
                "Invalid GenUI ${inputKind(content)}: ${error.message ?: "validation or compilation failed"}",
                error,
            )
        }
    }

    private fun inputKind(content: String): String = when {
        content.startsWith("<a2ui>") -> "Express"
        else -> "A2UI v0.9 JSON"
    }
}
