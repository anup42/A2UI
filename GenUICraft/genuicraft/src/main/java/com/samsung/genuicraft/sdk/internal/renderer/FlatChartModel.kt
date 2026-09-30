package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.capability.GeneratedRendererCapabilities
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.*
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Locale

/** The same data contract is implemented by dataset renderer_effective_semantics_v5_4. */
internal data class FlatChartSeries(
    val key: String, val label: String, val columnIndex: Int, val mark: String,
    val axis: String, val unit: String, val values: List<Double?>, val displays: List<String>
)

internal data class FlatChartModel(
    val kind: String, val xIndex: Int, val xLabel: String, val labels: List<String>,
    val xValues: List<Double>, val xType: String, val series: List<FlatChartSeries>,
    val sizes: List<Double?>, val sizeIndex: Int?, val horizontal: Boolean,
    val dataLabels: List<String>, val dataRows: List<List<String>>,
    val diagnostics: List<String>
) {
    val complete: Boolean get() = diagnostics.isEmpty()
}

private val chartMissing = setOf("", "-", "—", "–", "n/a", "na", "null", "none")
private val chartNumeric = Regex("^[\\s$€£₹¥]*(?:[A-Z]{3}\\s+)?([+-]?(?:\\d[\\d,]*(?:\\.\\d+)?|\\.\\d+)(?:[eE][+-]?\\d+)?)(?:\\s*[%°a-zA-Zµμ/²³$€£₹¥]*)$")

internal fun parseRichChartNumber(value: String): Double? {
    val text = value.trim().replace('−', '-')
    val number = chartNumeric.matchEntire(text)?.groupValues?.get(1) ?: return null
    // Do not interpret malformed thousands grouping such as 1,2 as twelve.
    if (',' in number && !Regex("[+-]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?").matches(number)) return null
    return number.replace(",", "").toDoubleOrNull()?.takeIf { it.isFinite() && kotlin.math.abs(it) <= 1e150 }
}

private fun chartTime(value: String): Double? {
    val text = value.trim()
    if (!Regex("\\d{4}-\\d{2}-\\d{2}(?:T.*)?").matches(text)) return null
    return runCatching { Instant.parse(text).toEpochMilli().toDouble() }.getOrNull()
        ?: runCatching { OffsetDateTime.parse(text).toInstant().toEpochMilli().toDouble() }.getOrNull()
        ?: runCatching { LocalDateTime.parse(text).toInstant(ZoneOffset.UTC).toEpochMilli().toDouble() }.getOrNull()
        ?: runCatching { LocalDate.parse(text).atStartOfDay().toInstant(ZoneOffset.UTC).toEpochMilli().toDouble() }.getOrNull()
}

internal fun extractRichChartModel(props: Map<String, Any?>, state: Map<String, Any?>): FlatChartModel {
    val diagnostics = linkedSetOf<String>()
    val rawType = props["chartType"]?.toString()?.trim()?.lowercase(Locale.ROOT).orEmpty().ifBlank { "bar" }
    val kind = GeneratedRendererCapabilities.chartSubtypeAliases[rawType] ?: rawType
    if (kind !in GeneratedRendererCapabilities.chartSubtypes) diagnostics += "unsupported_chart_subtype"
    val rawRows = (props["rows"] as? List<*>) ?: (props["data"] as? List<*>)
        ?: resolveDirectTableRows(props + ("statePath" to listOf("statePath", "rowsPath", "dataPath").mapNotNull { props[it]?.toString()?.takeIf { path -> path.isNotBlank() } }.firstOrNull()), state)
    val columns = resolveDirectTableColumns(props, rawRows)
    if (columns.size < 2) diagnostics += "chart_columns_insufficient"
    if (rawRows.size > 512 || columns.size > 64) diagnostics += "chart_resource_limit"
    val rows = rawRows.take(512).map { resolveDirectTableRow(it, columns, state) }
    fun token(value: String) = value.lowercase(Locale.ROOT).replace(Regex("[^a-z0-9]"), "")
    fun column(key: Any?, fallback: Int): Int {
        val name = key?.toString()?.trim().orEmpty()
        if (name.isBlank()) return fallback
        val found = columns.indexOfFirst { token(it.key) == token(name) || token(it.label) == token(name) }
        if (found < 0) diagnostics += "chart_binding_missing"
        return found
    }
    val xIndex = column(props["xKey"], 0)
    val labels = rows.map { it.getOrNull(xIndex).orEmpty().trim() }
    if (labels.any { it.isBlank() }) diagnostics += "chart_x_missing"
    val xType = props["xType"]?.toString()?.lowercase(Locale.ROOT)
        ?: if (kind in setOf("scatter", "bubble")) "number" else "category"
    if (props.containsKey("xType") && props["xType"] !is String) diagnostics += "chart_x_type_invalid"
    if (xType !in setOf("category", "number", "time")) diagnostics += "chart_x_type_invalid"
    if (kind in setOf("scatter", "bubble") && xType != "number") diagnostics += "chart_scatter_x_not_numeric"
    val xValues = labels.mapIndexed { index, label ->
        val value = when (xType) { "number" -> parseRichChartNumber(label); "time" -> chartTime(label); else -> index.toDouble() }
        if (value == null) diagnostics += "chart_x_invalid"
        value ?: index.toDouble()
    }
    val explicitSeries = props["series"] as? List<*>
    if (props.containsKey("series") && explicitSeries == null) diagnostics += "chart_series_invalid"
    val single = kind in setOf("bar", "column", "pie", "donut", "scatter", "bubble", "funnel", "treemap")
    val boxKeys = props["boxKeys"] as? Map<*, *>
    val specs: List<Map<*, *>> = when {
        kind == "box" -> listOf("min", "q1", "median", "q3", "max").map { role ->
            if (boxKeys?.get(role) == null) diagnostics += "chart_box_keys_required"
            mapOf("yKey" to (boxKeys?.get(role) ?: role), "label" to role)
        }
        explicitSeries != null -> explicitSeries.map {
            if (it !is Map<*, *> || it["yKey"] == null || it.keys.any { key -> key !in setOf("yKey", "label", "type", "axis", "unit") }) diagnostics += "chart_series_invalid"
            (it as? Map<*, *>) ?: emptyMap<Any, Any>()
        }
        kind == "combo" -> { diagnostics += "chart_combo_series_required"; emptyList() }
        props["yKey"] != null || single -> listOf(mapOf("yKey" to (props["yKey"] ?: columns.getOrNull(1)?.key)))
        else -> columns.indices.filter { it != xIndex && rows.any { row -> parseRichChartNumber(row.getOrNull(it).orEmpty()) != null } }
            .map { mapOf("yKey" to columns[it].key) }
    }
    if (specs.size > 16) diagnostics += "chart_resource_limit"
    val series = specs.take(16).map { spec ->
        if (spec.any { (_, v) -> v !is String } || (spec["yKey"] as? String).isNullOrBlank()) diagnostics += "chart_series_invalid"
        val index = column(spec["yKey"], 1)
        if (index == xIndex) diagnostics += "chart_axes_collide"
        val displays = rows.map { it.getOrNull(index).orEmpty().trim() }
        val values = displays.map {
            val number = parseRichChartNumber(it)
            if (number == null && it.lowercase(Locale.ROOT) !in chartMissing) diagnostics += "chart_value_invalid"
            number
        }
        if (values.none { it != null }) diagnostics += "chart_series_empty"
        val mark = spec["type"]?.toString()?.lowercase(Locale.ROOT) ?: if (kind == "combo") "" else kind
        if (kind == "combo" && mark !in setOf("bar", "column", "line", "area", "scatter")) diagnostics += "chart_series_mark_invalid"
        if (kind != "combo" && spec["type"] != null && mark != kind) diagnostics += "chart_series_mark_invalid"
        val axis = spec["axis"]?.toString()?.lowercase(Locale.ROOT) ?: "left"
        if (axis !in setOf("left", "right") || (axis == "right" && kind !in setOf("combo", "line", "area", "scatter"))) diagnostics += "chart_axis_invalid"
        FlatChartSeries(columns.getOrNull(index)?.key.orEmpty(), spec["label"]?.toString() ?: columns.getOrNull(index)?.label.orEmpty(),
            index, mark, axis, spec["unit"]?.toString().orEmpty(), values, displays)
    }
    if (series.isEmpty() || rows.isEmpty()) diagnostics += "chart_points_empty"
    if (series.map { it.columnIndex }.distinct().size != series.size) diagnostics += "chart_series_duplicate"
    if (single && kind !in setOf("scatter") && series.size > 1) diagnostics += "chart_single_series_required"
    if (rawType.contains("dual") && series.map { it.axis }.toSet() != setOf("left", "right")) diagnostics += "chart_dual_axes_required"
    if (series.any { it.axis == "right" } && series.none { it.axis == "left" }) diagnostics += "chart_dual_axes_required"
    if (kind in setOf("pie", "donut", "treemap", "funnel", "radar", "box", "stackedarea", "stackedbar") && series.any { it.values.any { v -> v == null } }) diagnostics += "chart_complete_values_required"
    if (kind in setOf("pie", "donut", "treemap", "funnel", "radar") && series.any { it.values.any { v -> v != null && v < 0 } }) diagnostics += "chart_negative_weight"
    if (kind in setOf("pie", "donut", "treemap", "funnel") && (series.firstOrNull()?.values?.filterNotNull()?.sum() ?: 0.0) <= 0) diagnostics += "chart_positive_total_required"
    if (kind == "radar" && rows.size < 3) diagnostics += "chart_radar_categories_insufficient"
    if (kind == "box" && series.size == 5 && rows.indices.any { r ->
        val v = series.map { it.values[r] }; v.any { it == null } || v.filterNotNull().zipWithNext().any { it.first > it.second }
    }) diagnostics += "chart_box_order_invalid"
    val sizeIndex = if (kind == "bubble") column(props["sizeKey"], -1) else null
    if (kind == "bubble" && (sizeIndex == null || sizeIndex < 0 || sizeIndex == xIndex || series.any { it.columnIndex == sizeIndex })) diagnostics += "chart_size_key_required"
    val sizes = if (kind == "bubble") rows.map { parseRichChartNumber(it.getOrNull(sizeIndex ?: -1).orEmpty()) } else emptyList()
    if (kind == "bubble" && sizes.any { it == null || it < 0 }) diagnostics += "chart_size_invalid"
    val orientation = props["orientation"]?.toString()?.lowercase(Locale.ROOT) ?: if (kind == "bar") "horizontal" else "vertical"
    if (props.containsKey("orientation") && props["orientation"] !is String) diagnostics += "chart_orientation_invalid"
    if (orientation !in setOf("horizontal", "vertical") || (orientation == "horizontal" && kind !in setOf("bar", "column", "groupedbar", "stackedbar"))) diagnostics += "chart_orientation_invalid"
    if (orientation == "horizontal" && xType != "category") diagnostics += "chart_orientation_invalid"
    return FlatChartModel(kind, xIndex, columns.getOrNull(xIndex)?.label.orEmpty(), labels, xValues, xType, series, sizes, sizeIndex,
        orientation == "horizontal", columns.map { it.label }, rows, diagnostics.toList())
}
