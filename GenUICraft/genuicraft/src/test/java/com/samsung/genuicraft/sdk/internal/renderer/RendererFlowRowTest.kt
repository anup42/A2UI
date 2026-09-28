package com.samsung.genuicraft.sdk.internal.renderer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class RendererFlowRowTest {
    @Test
    fun exactFitStaysOnOneLineAndNextChildWraps() {
        val lines = planFlowRowLines(widths = listOf(40, 50, 20), maxWidth = 100, spacing = 10)

        assertEquals(listOf(FlowRowLine(0..1, 100), FlowRowLine(2..2, 20)), lines)
    }

    @Test
    fun oversizedChildKeepsItsOwnLineWithoutHidingFollowingChildren() {
        val lines = planFlowRowLines(widths = listOf(130, 20, 20), maxWidth = 100, spacing = 10)

        assertEquals(listOf(FlowRowLine(0..0, 130), FlowRowLine(1..2, 50)), lines)
        assertEquals((0..2).toList(), lines.flatMap { it.indices.toList() })
    }

    @Test
    fun emptyAndUnboundedRowsDoNotCreatePhantomLines() {
        assertEquals(emptyList<FlowRowLine>(), planFlowRowLines(emptyList(), 100, 8))
        assertEquals(
            listOf(FlowRowLine(0..2, 179)),
            planFlowRowLines(listOf(40, 50, 65), Int.MAX_VALUE, 12)
        )
    }

    @Test
    fun metricCardsShareWidthWithinEachLineOnly() {
        val naturalWidths = listOf(142, 142, 142)
        val lines = planFlowRowLines(naturalWidths, maxWidth = 300, spacing = 8)
        val weights = listOf(1f, 1f, 1f)

        assertEquals(listOf(FlowRowLine(0..1, 292), FlowRowLine(2..2, 142)), lines)
        assertEquals(mapOf(0 to 146, 1 to 146), flowRowWeightWidths(lines[0], naturalWidths, weights, 300))
        assertEquals(mapOf(2 to 300), flowRowWeightWidths(lines[1], naturalWidths, weights, 300))
    }

    @Test
    fun equalWeightsRemainEqualWhenIntrinsicWidthsDiffer() {
        val naturalWidths = listOf(142, 200)
        val line = planFlowRowLines(naturalWidths, maxWidth = 400, spacing = 8).single()

        assertEquals(mapOf(0 to 196, 1 to 196),
            flowRowWeightWidths(line, naturalWidths, listOf(1f, 1f), 400))
    }

    @Test
    fun fixedChildKeepsItsWidthWhileWeightedChildrenShareTheRemainder() {
        val naturalWidths = listOf(60, 142, 200)
        val line = planFlowRowLines(naturalWidths, maxWidth = 450, spacing = 8).single()

        assertEquals(mapOf(1 to 187, 2 to 187),
            flowRowWeightWidths(line, naturalWidths, listOf(null, 1f, 1f), 450))
    }

    @Test
    fun rtlStartSpacingPlacesFirstChildAtTheRightEdge() {
        val positions = with(Density(1f)) {
            flowRowItemPositions(
                width = 100,
                itemWidths = intArrayOf(20, 30),
                arrangement = Arrangement.spacedBy(10.dp),
                layoutDirection = LayoutDirection.Rtl
            )
        }

        assertArrayEquals(intArrayOf(80, 40), positions)
    }

    @Test
    fun centerAndVerticalSpacingUseTheAvailableLineSpace() {
        val itemPositions = with(Density(1f)) {
            flowRowItemPositions(100, intArrayOf(20, 30), Arrangement.Center, LayoutDirection.Ltr)
        }
        val rowPositions = with(Density(1f)) {
            flowRowLinePositions(50, intArrayOf(10, 15), Arrangement.spacedBy(6.dp))
        }

        assertArrayEquals(intArrayOf(25, 45), itemPositions)
        assertArrayEquals(intArrayOf(0, 16), rowPositions)
    }
}
