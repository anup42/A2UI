package com.samsung.genuicraft.sdk.internal.renderer

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import java.io.File
import org.junit.Assert.*
import org.junit.Test

class FlatRichChartTest {
    @Test fun timeAxisKeepsIntradayPrecision() {
        assertEquals("00:00:00", chartTimeTick(1767225600123.0, 1000.0))
        assertEquals("00:00:01", chartTimeTick(1767225601123.0, 1000.0))
        assertEquals("2026-01-01", chartTimeTick(1767225600000.0, 172800000.0))
    }

    @Test fun expressRoundTripKeepsSeriesAndAxisBindings() {
        val program = """<a2ui>
root=Chart("combo",["week","hours","quiz"],rows=[["W1",8,78],["W2",10,90]],xKey="week",series=[{yKey:"hours",type:"column",axis:"left",unit:"h"},{yKey:"quiz",type:"line",axis:"right",unit:"%"}])
</a2ui>"""
        val graph = A2uiExpressCodec.decode(program)
        val reparsed = A2uiExpressCodec.decode(A2uiExpressCodec.encode(graph))
        assertEquals(graph, reparsed)
        val propsJson = reparsed.getAsJsonObject("elements").getAsJsonObject(reparsed["root"].asString).getAsJsonObject("props")
        val props: Map<String, Any?> = Gson().fromJson(propsJson, object : TypeToken<Map<String, Any?>>() {}.type)
        val chart = extractRichChartModel(props, emptyMap())
        assertTrue(chart.diagnostics.toString(), chart.complete)
        assertEquals(listOf("left", "right"), chart.series.map { it.axis })
        assertEquals(listOf("h", "%"), chart.series.map { it.unit })
    }

    @Test fun sharedChartFixturesMatchAndroid() {
        val file = File("../../dataset/tests/fixtures/chart_contract_v2_1.json")
        val fixtures: List<Map<String, Any?>> = Gson().fromJson(file.readText(), object : TypeToken<List<Map<String, Any?>>>() {}.type)
        fixtures.forEach { fixture ->
            @Suppress("UNCHECKED_CAST")
            val model = extractRichChartModel(fixture["props"] as Map<String, Any?>, fixture["state"] as Map<String, Any?>)
            @Suppress("UNCHECKED_CAST")
            val expected = fixture["expected"] as Map<String, Any?>
            val name = "${fixture["name"]}: ${model.diagnostics}"
            assertEquals(name, expected["complete"], model.complete)
            expected["diagnostic"]?.let { assertTrue(name, it in model.diagnostics) }
            expected["series_values"]?.let { assertEquals(name, it, model.series.map { s -> s.values }) }
            expected["x_values"]?.let { assertEquals(name, it, model.xValues) }
            expected["sizes"]?.let { assertEquals(name, it, model.sizes) }
            expected["axes"]?.let { assertEquals(name, it, model.series.map { s -> s.axis }) }
            expected["range"]?.let { assertEquals(name, it, chartValueRange(model).let { r -> listOf(r.min, r.max) }) }
        }
    }

    @Test fun zeroAndNegativeValuesKeepTheirActualGeometry() {
        val model = extractRichChartModel(mapOf("chartType" to "bar", "rows" to listOf(listOf("loss", -8), listOf("zero", 0), listOf("profit", 2))), emptyMap())
        val range = chartValueRange(model)
        assertEquals(0f, range.fraction(-8.0))
        assertEquals(.8f, range.fraction(0.0))
        assertEquals(1f, range.fraction(2.0))
    }
}
