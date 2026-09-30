package com.samsung.genuicraft.sdk.internal.renderer

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.*

private val chartColors = listOf(0xFF2667C9, 0xFFD55E00, 0xFF008A65, 0xFF9253B4,
    0xFFB28700, 0xFFCC4477, 0xFF007D92, 0xFF77502C, 0xFF5669A7, 0xFFAD3636,
    0xFF497C36, 0xFF836886, 0xFF746E23, 0xFFA05278, 0xFF397B80, 0xFF6C7177).map { Color(it) }

internal data class ChartRange(val min: Double, val max: Double) {
    fun fraction(value: Double): Float = ((value - min) / (max - min)).toFloat()
}

/** Signed stacks maintain independent positive/negative accumulators. */
internal fun chartValueRange(model: FlatChartModel, axis: String = "left"): ChartRange {
    val series = model.series.filter { it.axis == axis }
    val values = if (model.kind in setOf("stackedbar", "stackedarea")) model.labels.indices.flatMap { r ->
        val v = series.mapNotNull { it.values[r] }
        listOf(v.filter { it > 0 }.sum(), v.filter { it < 0 }.sum())
    } else series.flatMap { it.values.filterNotNull() }
    val lo = min(0.0, values.minOrNull() ?: 0.0)
    val hi = max(0.0, values.maxOrNull() ?: 0.0)
    return if (lo == hi) ChartRange(lo, lo + 1.0) else ChartRange(lo, hi)
}

private fun chartTick(value: Double): String = when {
    abs(value) >= 1e6 -> String.format(Locale.US, "%.2gM", value / 1e6)
    abs(value) >= 1e3 -> String.format(Locale.US, "%.3gk", value / 1e3)
    value == 0.0 -> "0"
    else -> String.format(Locale.US, "%.3g", value)
}

internal fun chartTimeTick(value: Double, span: Double): String {
    val pattern = when {
        span < 1000 -> "ss.SSS"
        span < 3600000 -> "HH:mm:ss"
        span < 86400000 -> "HH:mm"
        else -> "yyyy-MM-dd"
    }
    return java.time.Instant.ofEpochMilli(value.toLong()).atOffset(java.time.ZoneOffset.UTC)
        .format(java.time.format.DateTimeFormatter.ofPattern(pattern, Locale.US))
}

@Composable
internal fun RenderRichChart(props: Map<String, Any?>, model: FlatChartModel, modifier: Modifier) {
    val title = props["title"]?.toString().orEmpty()
    val subtitle = props["subtitle"]?.toString().orEmpty()
    val ink = MaterialTheme.colorScheme.onSurface
    val grid = MaterialTheme.colorScheme.outlineVariant
    val description = buildString {
        append(title).append(". ").append(model.kind).append(" chart. ")
        model.labels.forEachIndexed { r, label ->
            append(label).append(": ")
            model.dataLabels.forEachIndexed { c, name -> append(name).append(' ').append(model.dataRows[r][c].ifBlank { "missing" }).append("; ") }
        }
    }
    Card(modifier = modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (title.isNotBlank()) Text(title, style = MaterialTheme.typography.titleMedium)
            if (subtitle.isNotBlank()) Text(subtitle, style = MaterialTheme.typography.bodySmall)
            val weights = model.kind in setOf("pie", "donut", "treemap", "funnel")
            if (!weights) {
                model.series.forEachIndexed { i, series ->
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Box(Modifier.size(12.dp).background(chartColors[i % chartColors.size]))
                        Text(buildString {
                            append(series.label)
                            if (series.unit.isNotBlank()) append(" (").append(series.unit).append(')')
                            if (model.series.any { it.axis == "right" }) append(" · ").append(series.axis).append(" axis")
                            if (model.kind == "combo") append(" · ").append(series.mark)
                        }, style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
            val axisLabels = listOfNotNull(props["yLabel"]?.toString(), props["rightYLabel"]?.toString()).filter { it.isNotBlank() }
            if (axisLabels.isNotEmpty()) Text(axisLabels.joinToString(" / "), style = MaterialTheme.typography.labelSmall)
            Canvas(Modifier.fillMaxWidth().height(if (model.horizontal) max(240, min(480, model.labels.size * 30)).dp else 260.dp)
                .semantics { contentDescription = description }) {
                drawChartPlot(model, ink, grid)
            }
            Text((props["xLabel"]?.toString() ?: model.xLabel) + if (model.xType == "time") " (UTC)" else "", style = MaterialTheme.typography.labelSmall)
            // A bounded scroll viewport keeps every exact source value readable on mobile.
            // Geometry conveys trends; this data key also identifies small/overlapping marks.
            Column(Modifier.fillMaxWidth().heightIn(max = 200.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                model.labels.forEachIndexed { r, label ->
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (weights) Box(Modifier.size(10.dp).background(chartColors[r % chartColors.size]))
                        Text(buildString {
                            append(label).append(": ")
                            append(model.dataLabels.indices.filter { it != model.xIndex }.joinToString("; ") { c ->
                                "${model.dataLabels[c]} ${model.dataRows[r][c].ifBlank { "—" }}"
                            })
                        }, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}

internal fun DrawScope.drawChartPlot(model: FlatChartModel, ink: Color, grid: Color) {
    if (!model.complete) return
    when (model.kind) {
        "pie", "donut" -> drawWeightCircle(model, ink)
        "treemap" -> drawFlatTreemap(model, ink)
        "funnel" -> drawFunnel(model, ink)
        "radar" -> drawRadarChart(model, ink, grid)
        else -> drawCartesianChart(model, ink, grid)
    }
}

private fun DrawScope.chartText(text: String, x: Float, y: Float, ink: Color, align: Paint.Align = Paint.Align.LEFT) {
    drawContext.canvas.nativeCanvas.drawText(text, x, y, Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ink.toArgb(); textSize = 10.dp.toPx(); textAlign = align
    })
}

private fun DrawScope.drawCartesianChart(model: FlatChartModel, ink: Color, grid: Color) {
    val left = (if (model.horizontal) 78 else 48).dp.toPx()
    val right = size.width - (if (model.series.any { it.axis == "right" }) 48 else 24).dp.toPx()
    val top = 24.dp.toPx()
    val bottom = size.height - 30.dp.toPx()
    val width = (right - left).coerceAtLeast(1f)
    val height = (bottom - top).coerceAtLeast(1f)
    val leftRange = chartValueRange(model)
    val rightRange = chartValueRange(model, "right")
    fun value(v: Double, axis: String = "left"): Float {
        val f = (if (axis == "right") rightRange else leftRange).fraction(v)
        return if (model.horizontal) left + width * f else bottom - height * f
    }
    val n = model.labels.size
    val step = if (model.horizontal) height / n else width / n
    val xMin = model.xValues.minOrNull() ?: 0.0
    val xMax = model.xValues.maxOrNull() ?: 1.0
    val hasBars = model.kind == "box" || model.series.any { it.mark in setOf("bar", "column", "groupedbar", "stackedbar") }
    val xGap = model.xValues.distinct().sorted().zipWithNext().map { it.second - it.first }.minOrNull() ?: 1.0
    val domainMin = xMin - if (hasBars) xGap / 2 else 0.0
    val domainMax = xMax + if (hasBars) xGap / 2 else 0.0
    val barStep = if (model.xType == "category" || model.horizontal) step else (width * xGap / (domainMax - domainMin)).toFloat()
    fun x(r: Int): Float = if (model.horizontal) top + step * (r + .5f)
        else if (model.xType != "category") left + width * (if (domainMax == domainMin) .5f else ((model.xValues[r] - domainMin) / (domainMax - domainMin)).toFloat())
        else left + step * (r + .5f)
    for (tick in 0..4) {
        val v = leftRange.min + (leftRange.max - leftRange.min) * tick / 4
        val p = value(v)
        if (model.horizontal) {
            drawLine(grid, Offset(p, top), Offset(p, bottom)); chartText(chartTick(v), p, bottom + 16.dp.toPx(), ink, Paint.Align.CENTER)
        } else {
            drawLine(grid, Offset(left, p), Offset(right, p)); chartText(chartTick(v), left - 5.dp.toPx(), p + 3.dp.toPx(), ink, Paint.Align.RIGHT)
            if (model.series.any { it.axis == "right" }) chartText(chartTick(rightRange.min + (rightRange.max - rightRange.min) * tick / 4), right + 5.dp.toPx(), p + 3.dp.toPx(), ink)
        }
    }
    val zero = value(0.0)
    if (model.horizontal) drawLine(ink.copy(alpha = .7f), Offset(zero, top), Offset(zero, bottom))
    else drawLine(ink.copy(alpha = .7f), Offset(left, zero), Offset(right, zero))
    if (model.xType == "category") {
        val stride = max(1, ceil(n / (if (model.horizontal) 12.0 else 5.0)).toInt())
        model.labels.indices.filter { it % stride == 0 }.forEach { r ->
            val label = model.labels[r].let { if (it.length > 11) it.take(10) + "…" else it }
            if (model.horizontal) chartText(label, left - 5.dp.toPx(), x(r) + 3.dp.toPx(), ink, Paint.Align.RIGHT)
            else chartText(label, x(r), bottom + 16.dp.toPx(), ink, Paint.Align.CENTER)
        }
    } else for (tick in 0..4) {
        val v = domainMin + (domainMax - domainMin) * tick / 4
        val label = if (model.xType == "time") chartTimeTick(v, domainMax - domainMin) else chartTick(v)
        chartText(label, left + width * tick / 4, bottom + 16.dp.toPx(), ink, when (tick) { 0 -> Paint.Align.LEFT; 4 -> Paint.Align.RIGHT; else -> Paint.Align.CENTER })
    }
    if (model.kind == "box") {
        model.labels.indices.forEach { r ->
            val v = model.series.map { value(it.values[r]!!) }; val c = x(r); val w = min(barStep * .6f, 32.dp.toPx())
            drawLine(ink, Offset(c, v[0]), Offset(c, v[4]), 2.dp.toPx())
            drawLine(ink, Offset(c - w / 3, v[0]), Offset(c + w / 3, v[0]), 2.dp.toPx())
            drawLine(ink, Offset(c - w / 3, v[4]), Offset(c + w / 3, v[4]), 2.dp.toPx())
            drawRect(chartColors[0].copy(alpha = .35f), Offset(c - w / 2, v[3]), Size(w, v[1] - v[3]))
            drawRect(chartColors[0], Offset(c - w / 2, v[3]), Size(w, v[1] - v[3]), style = Stroke(2.dp.toPx()))
            drawLine(ink, Offset(c - w / 2, v[2]), Offset(c + w / 2, v[2]), 2.dp.toPx())
        }; return
    }
    val positive = DoubleArray(n); val negative = DoubleArray(n)
    val bars = model.series.count { it.mark in setOf("bar", "column", "groupedbar", "stackedbar") }.coerceAtLeast(1)
    var barIndex = 0
    model.series.forEachIndexed { s, series ->
        val color = chartColors[s % chartColors.size]
        val isStack = model.kind in setOf("stackedbar", "stackedarea")
        val lower = DoubleArray(n)
        val upper = series.values.mapIndexed { r, v ->
            if (v == null) null else if (isStack) {
                val accumulator = if (v >= 0) positive else negative
                lower[r] = accumulator[r]; accumulator[r] += v; accumulator[r]
            } else v
        }
        when (series.mark) {
            "bar", "column", "groupedbar", "stackedbar" -> {
                val count = if (isStack) 1 else bars
                val w = barStep * .75f / count
                upper.forEachIndexed { r, v -> if (v != null) {
                    val c = x(r) - barStep * .375f + (if (isStack) 0 else barIndex) * w
                    val a = value(lower[r], series.axis); val b = value(v, series.axis)
                    if (model.horizontal) drawRect(color, Offset(min(a, b), c), Size(abs(b - a), w * .94f))
                    else drawRect(color, Offset(c, min(a, b)), Size(w * .94f, abs(b - a)))
                } }; barIndex++
            }
            "scatter", "bubble" -> upper.forEachIndexed { r, v -> if (v != null) {
                val radius = if (model.kind == "bubble") 18.dp.toPx() * sqrt((model.sizes[r] ?: 0.0) / (model.sizes.filterNotNull().maxOrNull()?.takeIf { it > 0 } ?: 1.0)).toFloat() else 4.dp.toPx()
                drawCircle(color.copy(alpha = .75f), radius, Offset(x(r), value(v, series.axis)))
            } }
            else -> {
                // Separate contiguous runs: nulls never become zero or a line across a gap.
                var r = 0
                while (r < n) {
                    if (upper[r] == null) { r++; continue }
                    val start = r
                    while (r + 1 < n && upper[r + 1] != null) r++
                    val end = r
                    val path = Path().apply {
                        moveTo(x(start), value(upper[start]!!, series.axis))
                        for (p in start + 1..end) lineTo(x(p), value(upper[p]!!, series.axis))
                    }
                    if (series.mark in setOf("area", "stackedarea")) {
                        val fill = Path().apply {
                            addPath(path)
                            for (p in end downTo start) lineTo(x(p), value(lower[p], series.axis))
                            close()
                        }
                        drawPath(fill, color.copy(alpha = if (isStack) .55f else .22f))
                    }
                    drawPath(path, color, style = Stroke(2.dp.toPx()))
                    for (p in start..end) drawCircle(color, 2.5.dp.toPx(), Offset(x(p), value(upper[p]!!, series.axis)))
                    r++
                }
            }
        }
    }
}

private fun DrawScope.drawWeightCircle(model: FlatChartModel, ink: Color) {
    val values = model.series.first().values.map { it!! }
    val total = values.sum()
    val diameter = min(size.width, size.height) * (if (model.kind == "donut") .72f else .9f)
    val origin = Offset((size.width - diameter) / 2, (size.height - diameter) / 2)
    var start = -90f
    values.forEachIndexed { i, v ->
        val sweep = (v / total * 360).toFloat()
        if (model.kind == "donut") drawArc(chartColors[i % chartColors.size], start, sweep, false, origin, Size(diameter, diameter), style = Stroke(diameter * .21f))
        else drawArc(chartColors[i % chartColors.size], start, sweep, true, origin, Size(diameter, diameter))
        start += sweep
    }
    if (model.kind == "donut") chartText(chartTick(total), size.width / 2, size.height / 2, ink, Paint.Align.CENTER)
}

private fun DrawScope.drawRadarChart(model: FlatChartModel, ink: Color, grid: Color) {
    val center = Offset(size.width / 2, size.height / 2)
    val radius = min(size.width, size.height) * .34f
    val maxValue = max(1.0, model.series.flatMap { it.values.filterNotNull() }.maxOrNull() ?: 1.0)
    fun point(r: Int, fraction: Float): Offset {
        val angle = -PI / 2 + 2 * PI * r / model.labels.size
        return center + Offset(cos(angle).toFloat(), sin(angle).toFloat()) * (radius * fraction)
    }
    for (level in 1..4) {
        val path = Path().apply { model.labels.indices.forEach { r -> val p = point(r, level / 4f); if (r == 0) moveTo(p.x, p.y) else lineTo(p.x, p.y) }; close() }
        drawPath(path, grid, style = Stroke(1.dp.toPx()))
        chartText(chartTick(maxValue * level / 4), center.x + 3.dp.toPx(), center.y - radius * level / 4, ink)
    }
    model.labels.forEachIndexed { r, label ->
        val p = point(r, 1f); drawLine(grid, center, p)
        val text = point(r, 1.17f); chartText(label.take(12), text.x, text.y, ink, Paint.Align.CENTER)
    }
    model.series.forEachIndexed { i, s ->
        val path = Path().apply { s.values.forEachIndexed { r, v -> val p = point(r, (v!! / maxValue).toFloat()); if (r == 0) moveTo(p.x, p.y) else lineTo(p.x, p.y) }; close() }
        drawPath(path, chartColors[i].copy(alpha = .12f)); drawPath(path, chartColors[i], style = Stroke(2.dp.toPx()))
    }
}

private fun DrawScope.drawFunnel(model: FlatChartModel, ink: Color) {
    val values = model.series.first().values.map { it!! }
    val maxValue = values.maxOrNull()!!.coerceAtLeast(1e-20)
    val step = size.height / values.size
    values.forEachIndexed { i, v ->
        val topWidth = size.width * (v / maxValue).toFloat() * .92f
        val bottomWidth = size.width * ((values.getOrNull(i + 1) ?: v) / maxValue).toFloat() * .92f
        val y = i * step
        drawPath(Path().apply {
            moveTo((size.width - topWidth) / 2, y); lineTo((size.width + topWidth) / 2, y)
            lineTo((size.width + bottomWidth) / 2, y + step - 2.dp.toPx()); lineTo((size.width - bottomWidth) / 2, y + step - 2.dp.toPx()); close()
        }, chartColors[i % chartColors.size].copy(alpha = .65f))
        chartText(model.labels[i].take(22), size.width / 2, y + step / 2, ink, Paint.Align.CENTER)
    }
}

private fun DrawScope.drawFlatTreemap(model: FlatChartModel, ink: Color) {
    val values = model.series.first().values.map { it!! }
    // A binary treemap gives area proportional to each weight, preserving input order.
    fun tile(indices: List<Int>, x: Float, y: Float, w: Float, h: Float) {
        if (indices.isEmpty()) return
        if (indices.size == 1) {
            val i = indices.first()
            drawRect(chartColors[i % chartColors.size].copy(alpha = .65f), Offset(x, y), Size(w, h))
            drawRect(ink.copy(alpha = .3f), Offset(x, y), Size(w, h), style = Stroke(1.dp.toPx()))
            if (w > 40.dp.toPx() && h > 18.dp.toPx()) chartText(model.labels[i].take(max(1, (w / 7.dp.toPx()).toInt())), x + 3.dp.toPx(), y + 14.dp.toPx(), ink)
            return
        }
        val total = indices.sumOf { values[it] }
        var split = 1; var subtotal = values[indices.first()]
        while (split < indices.lastIndex && subtotal < total / 2) { subtotal += values[indices[split]]; split++ }
        val fraction = (subtotal / total).toFloat()
        if (w >= h) { tile(indices.take(split), x, y, w * fraction, h); tile(indices.drop(split), x + w * fraction, y, w * (1 - fraction), h) }
        else { tile(indices.take(split), x, y, w, h * fraction); tile(indices.drop(split), x, y + h * fraction, w, h * (1 - fraction)) }
    }
    tile(values.indices.filter { values[it] > 0 }, 0f, 0f, size.width, size.height)
}
