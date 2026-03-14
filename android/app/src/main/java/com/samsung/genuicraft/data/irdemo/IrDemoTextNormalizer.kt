package com.samsung.genuicraft

import android.net.Uri

internal fun decodeIrDemoQueryText(raw: String): String {
    var value = raw.trim()
    if (value.isEmpty() || !value.contains('%')) {
        return value
    }

    repeat(2) {
        val decoded = runCatching { Uri.decode(value) }.getOrDefault(value).trim()
        if (decoded == value) {
            return value
        }
        value = decoded
    }

    return value
}
