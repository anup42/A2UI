package com.samsung.genuicraft.renderer.native

internal data class ParsedButton(
    val label: String,
    val url: String
)

internal data class ParsedLogo(
    val label: String,
    val url: String
)

internal data class ParsedMediaEntry(
    val label: String,
    val url: String,
    val iconLike: Boolean
)

internal data class BookingOption(
    val title: String,
    val details: String,
    val button: ParsedButton?,
    val logo: ParsedLogo? = null
)

internal data class TableCell(
    val text: String,
    val variant: String,
    val weight: Float
)

internal data class TableSpec(
    val header: List<TableCell>?,
    val rows: List<List<TableCell>>
)

internal data class WeatherTableColumns(
    val period: Int,
    val date: Int?,
    val condition: Int?,
    val high: Int?,
    val low: Int?,
    val temp: Int?,
    val precip: Int?,
    val wind: Int?,
    val humidity: Int?,
    val uv: Int?
)

internal data class WeatherRow(
    val period: String,
    val date: String?,
    val condition: String?,
    val high: String?,
    val low: String?,
    val temp: String?,
    val metrics: List<Pair<String, String>>
)

internal data class FlightTableColumns(
    val airline: Int,
    val depart: Int?,
    val arrive: Int?,
    val duration: Int?,
    val stops: Int?,
    val fare: Int?,
    val status: Int?
)

internal data class FlightRow(
    val airline: String,
    val depart: String?,
    val originCode: String?,
    val arrive: String?,
    val destinationCode: String?,
    val duration: String?,
    val stops: String?,
    val fare: String?,
    val status: String?
)

internal data class FlightPoint(
    val time: String?,
    val code: String?
)

internal data class StepEntry(
    val title: String,
    val details: String
)

internal data class LeadingLabelValue(
    val prefix: String,
    val label: String,
    val delimiter: String,
    val value: String
)

internal data class InlineBulletRow(
    val bullet: String,
    val text: String,
    val variant: String
)

internal data class WeatherCurrentDetails(
    val iconUrl: String?,
    val condition: String?,
    val temperature: String?,
    val feelsLike: String?,
    val humidity: String?,
    val wind: String?,
    val rainChance: String?,
    val uvIndex: String?,
    val summary: String?
)

internal sealed interface TextBlock {
    data class Title(val text: String) : TextBlock
    data class Heading(val text: String) : TextBlock
    data class Paragraph(val text: String) : TextBlock
    data class Bullets(val items: List<String>) : TextBlock
    data class NumberedSteps(val items: List<StepEntry>) : TextBlock
    data class Table(val rows: List<List<String>>) : TextBlock
    data class Sources(val links: List<ParsedButton>) : TextBlock
    data class Actions(val actions: List<ParsedButton>) : TextBlock
    data class BookingCards(val options: List<BookingOption>) : TextBlock
    data class MediaCards(val entries: List<ParsedMediaEntry>) : TextBlock
}
