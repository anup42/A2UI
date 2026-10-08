package com.samsung.genuicraft.sdk.internal.renderer

/** Keep prose at reading width; metric-sized pairs retain the compact aligned layout. */
internal fun shouldStackKeyValueRows(rows: List<List<String>>): Boolean = rows.any { row ->
    val label = row.getOrNull(0).orEmpty().trim()
    val value = row.getOrNull(1).orEmpty().trim()
    label.length > 36 || value.length > 80 || '\n' in label || '\n' in value
}
