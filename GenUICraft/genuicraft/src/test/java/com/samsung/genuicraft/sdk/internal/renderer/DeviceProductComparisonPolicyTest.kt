package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.deviceProductComparisonPrimaryIndex
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class DeviceProductComparisonPolicyTest {
    private val headers = listOf("Phone", "Battery", "Camera", "Price in India")
    private val rows = listOf(
        listOf(
            "Samsung Galaxy S25", "4,000 mAh [21][31]",
            "50 MP main + 12 MP ultra-wide + 10 MP telephoto, 12 MP front [15][8]",
            "Starts around ₹58,999 to ₹84,999 depending on listing and variant[23]",
        ),
        listOf(
            "iPhone 16", "3,561 mAh [27][35]",
            "48 MP main + 12 MP ultra-wide, 12 MP front [13][32]",
            "Starts at ₹79,900 officially; some listings show ₹76,999",
        ),
        listOf(
            "Google Pixel 9", "4,700 mAh [1][10]",
            "50 MP main + 48 MP ultra-wide, 10.5 MP front [17]",
            "Starts around ₹79,999; some India listings show lower sale prices[12]",
        ),
    )

    @Test fun `repaired phone comparison uses row cards and retains every original cell`() {
        val props = mapOf<String, Any?>("columns" to headers, "rows" to rows)
        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), compactScreen = true))

        assertEquals(0, deviceProductComparisonPrimaryIndex(
            headers, model.rows, props, model.shape, model.domain,
        ))
        model.rows.forEachIndexed { index, row ->
            val card = entityCardRowContent(headers, row, index, 0, emptyList())
            assertEquals(row.first(), card.title)
            assertEquals(headers.zip(row), card.representedSourceCells.map { it.label to it.value })
        }
    }

    @Test fun `feature matrix stays a table`() {
        val matrixHeaders = listOf("Feature", "Samsung Galaxy S25", "iPhone 16", "Google Pixel 9")
        val matrixRows = listOf(
            listOf("Battery", "4,000 mAh", "3,561 mAh", "4,700 mAh"),
            listOf("Camera", "50 MP", "48 MP", "50 MP"),
        )
        val props = mapOf<String, Any?>("columns" to matrixHeaders, "rows" to matrixRows)
        val model = requireNotNull(extractDirectTableModel(props, emptyMap(), compactScreen = true))

        assertEquals(FlatTableShape.FEATURE_MATRIX, model.shape)
        assertNull(deviceProductComparisonPrimaryIndex(
            matrixHeaders, model.rows, props, model.shape, model.domain,
        ))
    }

    @Test fun `explicit table and weak row identities keep the grid`() {
        val tableProps = mapOf<String, Any?>(
            "columns" to headers, "rows" to rows, "preferredPresentation" to "table",
        )
        val model = requireNotNull(extractDirectTableModel(tableProps, emptyMap(), compactScreen = true))
        assertNull(deviceProductComparisonPrimaryIndex(
            headers, model.rows, tableProps, model.shape, model.domain,
        ))

        val weakRows = listOf(rows[0], rows[0])
        assertNull(deviceProductComparisonPrimaryIndex(
            headers, weakRows, emptyMap(), model.shape, "generic",
        ))
    }
}
