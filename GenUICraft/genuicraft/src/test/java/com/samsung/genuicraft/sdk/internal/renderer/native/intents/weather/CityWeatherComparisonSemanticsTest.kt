package com.samsung.genuicraft.sdk.internal.renderer.native.intents.weather

import com.samsung.genuicraft.sdk.internal.renderer.FlatTableShape
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CityWeatherComparisonSemanticsTest {
    private val headers = listOf("City", "Current weather", "Today’s high", "Today’s low", "Rain chance / note")
    private val rows = listOf(
        listOf(
            "Bengaluru", "Mostly cloudy, about 21.8°C at IMD observation time",
            "About 29.0°C to 30.0°C", "About 16.0°C to 18.0°C",
            "Low chance of rain; cloudy/partly cloudy conditions[7][11][10]",
        ),
        listOf(
            "Mysore", "Partly sunny, about 27°C to 29°C range in the morning",
            "About 29.0°C to 29.6°C", "About 17.0°C to 18.3°C",
            "Mostly dry; sunny spells or cloudy/overcast[8][12]",
        ),
        listOf(
            "Ooty", "Cool and cloudy, about 14.9°C to 15.9°C in the day forecast",
            "About 23.9°C to 24.4°C", "About 14.9°C to 17.0°C",
            "Higher rain chance, including patchy rain nearby or showers[13]",
        ),
    )

    @Test fun `captured city weather table and card domains keep every field and citation`() {
        val plainRows = rows.map { row -> row.map { it.replace(Regex("""\[\d+\]"""), "") } }
        listOf(
            Triple("comparison", "table", rows),
            Triple("weather", "cards", plainRows),
        ).forEach { (domain, presentation, sourceRows) ->
            val model = requireNotNull(extractDirectTableModel(
                mapOf("columns" to headers, "rows" to sourceRows, "domain" to domain,
                    "preferredPresentation" to presentation),
                emptyMap(), compactScreen = true,
            ))
            assertEquals(FlatTableShape.ENTITY_ROW, model.shape)
            assertEquals(headers, model.columns.map { it.label })
            assertEquals(sourceRows, model.rows)
            val cards = requireNotNull(CityWeatherComparisonSemantics.buildRows(
                model.columns.map { it.label }, model.rows, model.domain,
            ))
            assertEquals(sourceRows.map { it[0] }, cards.map { it.city })
            cards.forEachIndexed { index, card ->
                assertEquals(headers.zip(sourceRows[index]),
                    listOf(card.cityLabel to card.city) + card.fields.map { it.label to it.value })
                assertEquals(sourceRows[index][1], card.iconCondition)
                assertEquals(CityWeatherFieldKind.CURRENT, card.fields[0].kind)
                assertEquals(CityWeatherFieldKind.HIGH, card.fields[1].kind)
                assertEquals(CityWeatherFieldKind.LOW, card.fields[2].kind)
                assertEquals(CityWeatherFieldKind.RAIN, card.fields[3].kind)
            }
        }
    }

    @Test fun `source condition drives icon while current temperature stays separately labelled`() {
        val result = requireNotNull(CityWeatherComparisonSemantics.buildRows(
            listOf("City", "Condition", "Current temperature", "High", "Low", "Humidity"),
            listOf(
                listOf("Delhi", "Hazy", "27°C", "31°C", "22°C", "63%"),
                listOf("Mumbai", "Showers", "29°C", "32°C", "26°C", "78%"),
            ),
            "comparison",
        ))
        assertEquals("Hazy", result[0].iconCondition)
        assertEquals("Showers", result[1].iconCondition)
        assertEquals("27°C", result[0].fields[1].value)
        assertEquals(CityWeatherFieldKind.OTHER, result[0].fields.last().kind)
    }

    @Test fun `ordinary tables and weather forecasts do not take the city route`() {
        assertNull(CityWeatherComparisonSemantics.buildRows(
            listOf("Feature", "Bengaluru", "Mysore"),
            listOf(listOf("High", "30°C", "29°C"), listOf("Low", "18°C", "17°C")),
            "comparison",
        ))
        assertNull(CityWeatherComparisonSemantics.buildRows(
            listOf("Date", "High", "Low", "Rain"),
            listOf(listOf("Wed", "31°C", "20°C", "57%"), listOf("Thu", "30°C", "19°C", "20%")),
            "weather",
        ))
        assertNull(CityWeatherComparisonSemantics.buildRows(headers, rows, "generic"))
        assertNull(CityWeatherComparisonSemantics.buildRows(headers, rows.take(1), "comparison"))
        assertNull(CityWeatherComparisonSemantics.buildRows(headers, rows + listOf(rows.first()), "comparison"))
    }
}
