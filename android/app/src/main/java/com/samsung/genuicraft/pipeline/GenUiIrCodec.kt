package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

/** One detection/decoding boundary for all supported model-facing IR formats. */
internal object GenUiIrCodec {
    data class Decoded(val sourceFormat: GenUiIrFormat, val flatSpec: JsonObject)

    fun detect(payload: JsonElement?): GenUiIrFormat {
        require(payload != null && !payload.isJsonNull) { "IR payload is null." }
        if (payload.isJsonPrimitive && payload.asJsonPrimitive.isString) {
            val text = payload.asString.trim()
            if (A2uiExpressCodec.looksLike(text)) return GenUiIrFormat.A2UI_EXPRESS_V1
            runCatching { JsonParser.parseString(text) }.getOrNull()?.let { parsed ->
                if (!(parsed.isJsonPrimitive && parsed.asJsonPrimitive.isString && parsed.asString == text)) {
                    return detect(parsed)
                }
            }
        }
        if (CompactIrCodec.looksLike(payload)) return GenUiIrFormat.COMPACT_IR_V2
        if (A2uiWireCodec.looksLike(payload)) return GenUiIrFormat.A2UI_V1_WIRE
        if (FlatSpecContract.looksLikeFlatSpec(payload)) return GenUiIrFormat.FLAT_SPEC_V1
        error("Unsupported or ambiguous IR format.")
    }

    fun decode(payload: JsonElement): Decoded {
        val normalized = if (payload.isJsonPrimitive && payload.asJsonPrimitive.isString) {
            val text = payload.asString.trim()
            if (A2uiExpressCodec.looksLike(text)) payload
            else runCatching { JsonParser.parseString(text) }.getOrDefault(payload)
        } else payload
        return when (val format = detect(normalized)) {
            GenUiIrFormat.FLAT_SPEC_V1 -> Decoded(format, normalized.asJsonObject.deepCopy())
            GenUiIrFormat.COMPACT_IR_V2 -> Decoded(format, CompactIrCodec.decode(normalized))
            GenUiIrFormat.A2UI_EXPRESS_V1 -> Decoded(format, A2uiExpressCodec.decode(normalized.asString))
            GenUiIrFormat.A2UI_V1_WIRE -> Decoded(format, A2uiWireCodec.decode(normalized))
        }
    }
}
