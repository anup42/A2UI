package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

internal data class DayActivityScheduleCardContent(
    val day: String,
    val activityLabel: String,
    val activity: String
)

private val literalScheduleDay = Regex("Day [1-9][0-9]*")
private val scheduleUrl = Regex("(?i)(?:[a-z][a-z0-9+.-]*://|www\\.|mailto:|tel:|geo:|intent:)")
private val scheduleBinding = Regex("\\$(?:\\{|/|[A-Za-z_])|@source")

/** Only the complete literal two-field schedule qualifies; other rows keep their existing route. */
internal fun dayActivityScheduleCardContent(
    headers: List<String>,
    row: List<String>
): DayActivityScheduleCardContent? {
    if (headers.size != 2 || row.size != 2 || headers[0] != "Day" ||
        headers[1] !in setOf("Activity", "Task") || !literalScheduleDay.matches(row[0]) ||
        row[1].isBlank()) return null
    if (row.any { value ->
            value.contains("{{") || value.contains("}}") ||
                scheduleBinding.containsMatchIn(value) || scheduleUrl.containsMatchIn(value)
        }) return null
    return DayActivityScheduleCardContent(row[0], headers[1], row[1])
}

internal fun shouldStackDayActivitySchedule(availableWidthDp: Float, fontScale: Float): Boolean =
    !availableWidthDp.isFinite() || !fontScale.isFinite() || fontScale <= 0f ||
        availableWidthDp < 320f || fontScale > 1.2f
