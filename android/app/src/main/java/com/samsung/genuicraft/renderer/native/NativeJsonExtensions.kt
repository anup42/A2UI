package com.samsung.genuicraft.renderer.native

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

internal fun JsonElement.asJsonObjectOrNull(): JsonObject? =
    if (isJsonObject) asJsonObject else null

internal fun JsonElement.asStringOrNull(): String? =
    if (isJsonPrimitive && asJsonPrimitive.isString) asString else null

internal fun JsonElement.asBooleanOrNull(): Boolean? {
    if (!isJsonPrimitive || !asJsonPrimitive.isBoolean) return null
    return asBoolean
}

internal fun JsonObject.getAsJsonArrayOrNull(key: String): JsonArray? {
    val value = get(key) ?: return null
    return if (value.isJsonArray) value.asJsonArray else null
}

internal fun JsonObject.getAsJsonObjectOrNull(key: String): JsonObject? {
    val value = get(key) ?: return null
    return if (value.isJsonObject) value.asJsonObject else null
}

internal fun JsonObject.getString(key: String): String? {
    val value = get(key) ?: return null
    return if (value.isJsonPrimitive && value.asJsonPrimitive.isString) value.asString else null
}

internal fun JsonObject.getAsNumberOrNull(key: String): Double? {
    val value = get(key) ?: return null
    if (!value.isJsonPrimitive || !value.asJsonPrimitive.isNumber) {
        return null
    }
    return value.asDouble
}

internal fun JsonObject.hasString(key: String): Boolean {
    val value = get(key) ?: return false
    return value.isJsonPrimitive && value.asJsonPrimitive.isString
}
