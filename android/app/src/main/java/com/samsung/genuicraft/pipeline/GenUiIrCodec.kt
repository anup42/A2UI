package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

/** Production detection/decoding boundary: Express in, standard A2UI out. */
internal object GenUiIrCodec {
    /** Canonical graph payload; the legacy property name is retained for migration callers. */
    data class Decoded(val sourceFormat: GenUiIrFormat, val flatSpec: JsonObject)

    fun detect(payload: JsonElement?): GenUiIrFormat {
        require(payload != null && !payload.isJsonNull) { "IR payload is null." }
        if (payload.isJsonPrimitive && payload.asJsonPrimitive.isString) {
            val text = payload.asString.trim()
            runCatching { JsonParser.parseString(text) }.getOrNull()?.let { parsed ->
                if (!(parsed.isJsonPrimitive && parsed.asJsonPrimitive.isString && parsed.asString == text)) {
                    return detect(parsed)
                }
            }
            if (A2uiExpressCodec.looksLike(text)) return GenUiIrFormat.A2UI_EXPRESS_V1
        }
        if (A2uiWireCodec.looksLike(payload)) return GenUiIrFormat.A2UI_V1_WIRE
        if (payload.isJsonObject && FlatSpecContract.looksLikeFlatSpec(payload)) {
            error("Legacy FlatSpec is accepted only by the explicit migration importer.")
        }
        if (payload.isJsonObject && payload.asJsonObject.get("v")?.asString == "gci2") {
            error("Compact IR is migration-only and is not accepted by Android inference.")
        }
        error("Unsupported or ambiguous production IR format.")
    }

    fun decode(payload: JsonElement): Decoded {
        val normalized = if (payload.isJsonPrimitive && payload.asJsonPrimitive.isString) {
            val text = payload.asString.trim()
            if (A2uiExpressCodec.looksLike(text)) payload
            else runCatching { JsonParser.parseString(text) }.getOrDefault(payload)
        } else payload
        return when (val format = detect(normalized)) {
            GenUiIrFormat.A2UI_EXPRESS_V1 -> Decoded(format, A2uiExpressCodec.decode(normalized.asString))
            GenUiIrFormat.A2UI_V1_WIRE -> Decoded(format, A2uiWireCodec.decode(normalized))
            else -> error("Non-production IR format reached the production decoder.")
        }
    }
}
