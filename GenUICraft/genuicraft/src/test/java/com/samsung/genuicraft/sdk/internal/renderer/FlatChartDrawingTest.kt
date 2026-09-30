package com.samsung.genuicraft.sdk.internal.renderer

import android.graphics.Bitmap
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Canvas
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.CanvasDrawScope
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.LayoutDirection
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class FlatChartDrawingTest {
    @Test fun everySupportedFamilyDrawsWithTheProductionCanvasPath() {
        val fixtures: List<Map<String, Any?>> = Gson().fromJson(File("../../dataset/tests/fixtures/chart_contract_v2_1.json").readText(), object : TypeToken<List<Map<String, Any?>>>() {}.type)
        val output = File("build/reports/chart-rendering").apply { mkdirs() }
        val families = mutableSetOf<String>()
        val images = mutableSetOf<Int>()
        fixtures.forEach { fixture ->
            @Suppress("UNCHECKED_CAST")
            val model = extractRichChartModel(fixture["props"] as Map<String, Any?>, fixture["state"] as Map<String, Any?>)
            if (!model.complete) return@forEach
            families += model.kind
            val bitmap = Bitmap.createBitmap(720, 520, Bitmap.Config.ARGB_8888)
            bitmap.eraseColor(android.graphics.Color.WHITE)
            CanvasDrawScope().draw(Density(2f), LayoutDirection.Ltr, Canvas(bitmap.asImageBitmap()), Size(720f, 520f)) {
                drawChartPlot(model, Color.Black, Color.LightGray)
            }
            val pixels = IntArray(720 * 520)
            bitmap.getPixels(pixels, 0, 720, 0, 0, 720, 520)
            assertTrue("${fixture["name"]} produced an empty chart", pixels.count { it != android.graphics.Color.WHITE } > 100)
            images += pixels.contentHashCode()
            File(output, "${fixture["name"]}.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            bitmap.recycle()
        }
        assertEquals(16, families.size)
        assertTrue("Different chart families must not all fall back to bars", images.size >= 16)
    }
}
