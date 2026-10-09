package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement

/** Keep prose at reading width; metric-sized pairs retain the compact aligned layout. */
internal fun shouldStackKeyValueRows(rows: List<List<String>>): Boolean = rows.any { row ->
    val label = row.getOrNull(0).orEmpty().trim()
    val value = row.getOrNull(1).orEmpty().trim()
    label.length > 36 || value.length > 80 || '\n' in label || '\n' in value
}

private val passiveKeyValueBinding = Regex("\\$(?:/|[A-Za-z_])|@source")
private val passiveKeyValueUrl = Regex("(?i)(?:[a-z][a-z0-9+.-]*://|www\\.|mailto:|tel:|geo:|intent:)")

/** Presentation only: retain every row, and only mark equal original literal cells in passive tables. */
internal fun identicalPassiveKeyValueRowIndexes(
    table: FlatElement?,
    ancestors: List<FlatElement?> = emptyList(),
    isRepeated: Boolean = false,
): Set<Int> {
    val rows = passiveDirectLiteralTableRows(table, ancestors, isRepeated) ?: return emptySet()
    if ((table?.props?.get("columns") as? List<*>)?.size != 2) return emptySet()
    return rows.mapIndexedNotNull { index, value ->
        val row = value as List<*>
        val first = row[0] as String
        index.takeIf { first.isNotBlank() && first == row[1] }
    }.toSet()
}

/** Original, passive positional string cells; never inferred from resolved bindings. */
internal fun passiveDirectLiteralTableRows(
    table: FlatElement?,
    ancestors: List<FlatElement?> = emptyList(),
    isRepeated: Boolean = false,
): List<List<String>>? {
    if (table == null || isRepeated || !table.type.equals("table", true) || table.children.isNotEmpty()) return null
    val allowedProps = setOf("columns", "rows", "title", "subtitle", "domain", "preferredPresentation", "presentation")
    if (table.props.keys.any { it !in allowedProps }) return null
    if ((listOf(table) + ancestors).any { node ->
            node == null || node.repeat != null || node.visible != null ||
                !node.on.isNullOrEmpty() || !node.watch.isNullOrEmpty() || dynamicKeyValueContent(node.props)
        }) return null
    val columns = table.props["columns"] as? List<*> ?: return null
    if (columns.isEmpty() || columns.any { it !is String || it.isBlank() }) return null
    val rows = table.props["rows"] as? List<*> ?: return null
    if (rows.any { row -> row !is List<*> || row.size != columns.size || row.any { it !is String } }) return null
    return rows.map { row -> (row as List<*>).map { it as String } }
}

/** Recheck resolved cells while keeping the eligibility decision tied to exact source row identity. */
internal fun singleIdenticalKeyValueRowText(
    headers: List<String>,
    row: List<String>,
    rowIndex: Int,
    eligibleRows: Set<Int>,
): String? = row.getOrNull(0)?.takeIf { value ->
    rowIndex in eligibleRows && headers.size == 2 && row.size == 2 &&
        value.isNotBlank() && value == row[1] && !dynamicKeyValueContent(value)
}

private fun dynamicKeyValueContent(value: Any?): Boolean = when (value) {
    is String -> value.contains("{{") || value.contains("}}") || value.contains("\${") ||
        passiveKeyValueBinding.containsMatchIn(value) || passiveKeyValueUrl.containsMatchIn(value)
    is Map<*, *> -> value.any { (key, child) ->
        key !is String || key.startsWith("\$") || key in setOf("statePath", "rowsPath", "dataPath") || dynamicKeyValueContent(child)
    }
    is List<*> -> value.any(::dynamicKeyValueContent)
    null, is Number, is Boolean -> false
    else -> true
}
