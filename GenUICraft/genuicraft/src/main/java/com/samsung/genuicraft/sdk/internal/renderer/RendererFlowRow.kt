package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.layout.ParentDataModifier
import androidx.compose.ui.layout.Placeable
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.constrainHeight
import androidx.compose.ui.unit.constrainWidth
import kotlin.math.roundToInt

/** The subset of row weight used by the renderer's metric cards. */
internal interface RendererFlowRowScope {
    fun Modifier.weight(weight: Float, fill: Boolean = true): Modifier
}

private data class FlowWeight(val value: Float, val fill: Boolean) : ParentDataModifier {
    override fun Density.modifyParentData(parentData: Any?): Any = this@FlowWeight
}

private object RendererFlowRowScopeImpl : RendererFlowRowScope {
    override fun Modifier.weight(weight: Float, fill: Boolean): Modifier {
        require(weight > 0f && weight.isFinite()) { "Flow row weight must be positive and finite" }
        return this.then(FlowWeight(weight, fill))
    }
}

/** A row of child indexes and its natural width before row-local weight is distributed. */
internal data class FlowRowLine(val indices: IntRange, val width: Int)

/** Keeps an oversized child on its own row and never drops a child at a wrap boundary. */
internal fun planFlowRowLines(
    widths: List<Int>,
    maxWidth: Int,
    spacing: Int
): List<FlowRowLine> {
    if (widths.isEmpty()) return emptyList()
    require(widths.all { it >= 0 })
    val lines = mutableListOf<FlowRowLine>()
    var first = 0
    var lineWidth = 0L
    widths.forEachIndexed { index, childWidth ->
        val proposed = lineWidth + if (index == first) 0 else spacing.toLong() + childWidth
        if (index > first && proposed > maxWidth.toLong()) {
            lines += FlowRowLine(first until index, lineWidth.coerceIn(0, Int.MAX_VALUE.toLong()).toInt())
            first = index
            lineWidth = childWidth.toLong()
        } else {
            lineWidth = if (index == first) childWidth.toLong() else proposed
        }
    }
    lines += FlowRowLine(first until widths.size, lineWidth.coerceIn(0, Int.MAX_VALUE.toLong()).toInt())
    return lines
}

internal fun Density.flowRowItemPositions(
    width: Int,
    itemWidths: IntArray,
    arrangement: Arrangement.Horizontal,
    layoutDirection: LayoutDirection
): IntArray = IntArray(itemWidths.size).also { positions ->
    with(arrangement) { arrange(width, itemWidths, layoutDirection, positions) }
}

internal fun Density.flowRowLinePositions(
    height: Int,
    rowHeights: IntArray,
    arrangement: Arrangement.Vertical
): IntArray = IntArray(rowHeights.size).also { positions ->
    with(arrangement) { arrange(height, rowHeights, positions) }
}

/** Gives each weighted child the remaining width in its own line, as FlowRow's weight does. */
internal fun flowRowWeightWidths(
    line: FlowRowLine,
    naturalWidths: List<Int>,
    weights: List<Float?>,
    maxWidth: Int
): Map<Int, Int> {
    val weighted = line.indices.filter { weights[it] != null }
    if (weighted.isEmpty()) return emptyMap()
    if (maxWidth == Constraints.Infinity) return weighted.associateWith { naturalWidths[it] }
    val totalWeight = weighted.sumOf { weights[it]!!.toDouble() }
    val weightedNaturalWidth = weighted.sumOf { naturalWidths[it].toLong() }
    val spaceForWeights = (maxWidth.toLong() - (line.width.toLong() - weightedNaturalWidth))
        .coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
    var distributed = 0
    return weighted.mapIndexed { position, index ->
        val share = if (position == weighted.lastIndex) spaceForWeights - distributed else
            (spaceForWeights.toDouble() * weights[index]!! / totalWeight)
                .roundToInt().coerceIn(0, spaceForWeights - distributed)
        distributed += share
        index to share
    }.toMap()
}

/**
 * Renderer-owned wrapping layout. Compose Foundation changed FlowRow's binary signature between
 * the SDK and host versions, so SDK content must not call that experimental layout at runtime.
 */
@Composable
internal fun RendererFlowRow(
    modifier: Modifier = Modifier,
    horizontalArrangement: Arrangement.Horizontal = Arrangement.Start,
    verticalArrangement: Arrangement.Vertical = Arrangement.Top,
    content: @Composable RendererFlowRowScope.() -> Unit
) {
    Layout(
        content = { RendererFlowRowScopeImpl.content() },
        modifier = modifier
    ) { measurables, constraints ->
        val childConstraints = constraints.copy(minWidth = 0, minHeight = 0)
        val weights = measurables.map { it.parentData as? FlowWeight }
        val placeables = arrayOfNulls<Placeable>(measurables.size)
        val naturalWidths = measurables.indices.map { index ->
            if (weights[index] == null) {
                measurables[index].measure(childConstraints).also { placeables[index] = it }.width
            } else {
                // A weighted child is measured once, after its row's width has been decided.
                measurables[index].minIntrinsicWidth(0).coerceIn(0, constraints.maxWidth)
            }
        }
        val horizontalGap = horizontalArrangement.spacing.roundToPx()
        val lines = planFlowRowLines(naturalWidths, constraints.maxWidth, horizontalGap)

        lines.forEach { line ->
            val targets = flowRowWeightWidths(line, naturalWidths, weights.map { it?.value }, constraints.maxWidth)
            targets.forEach { (index, targetWidth) ->
                val weight = weights[index]!!
                val weightedConstraints = childConstraints.copy(
                    minWidth = if (weight.fill) targetWidth else 0,
                    maxWidth = targetWidth
                )
                placeables[index] = measurables[index].measure(weightedConstraints)
            }
        }

        val rowHeights = lines.map { line -> line.indices.maxOf { placeables[it]!!.height } }.toIntArray()
        val verticalGap = verticalArrangement.spacing.roundToPx()
        val naturalHeight = rowHeights.foldIndexed(0L) { index, total, rowHeight ->
            total + rowHeight + if (index == 0) 0 else verticalGap
        }.coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
        val naturalWidth = lines.maxOfOrNull { line ->
            line.indices.foldIndexed(0L) { position, total, index ->
                total + placeables[index]!!.width + if (position == 0) 0 else horizontalGap
            }.coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
        } ?: 0
        val width = constraints.constrainWidth(naturalWidth)
        val height = constraints.constrainHeight(naturalHeight)
        val rowYs = flowRowLinePositions(height, rowHeights, verticalArrangement)
        val itemXs = lines.map { line ->
            val itemWidths = line.indices.map { placeables[it]!!.width }.toIntArray()
            flowRowItemPositions(width, itemWidths, horizontalArrangement, layoutDirection)
        }

        layout(width, height) {
            lines.forEachIndexed { rowIndex, line ->
                line.indices.forEachIndexed { position, index ->
                    // arrange has already applied layoutDirection; placeRelative would mirror twice.
                    placeables[index]!!.place(itemXs[rowIndex][position], rowYs[rowIndex])
                }
            }
        }
    }
}
