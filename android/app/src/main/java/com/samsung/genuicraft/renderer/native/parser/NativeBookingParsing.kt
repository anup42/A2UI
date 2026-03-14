package com.samsung.genuicraft.renderer.native.parser

import com.samsung.genuicraft.renderer.native.BookingOption
import com.samsung.genuicraft.renderer.native.ParsedButton
import com.samsung.genuicraft.renderer.native.ParsedLogo

internal object NativeBookingParsing {
    fun collectBookingOptions(
        lines: List<String>,
        startIndex: Int,
        parseOptionLine: (String) -> Pair<String, String>?,
        parseButtonLine: (String) -> ParsedButton?,
        isBulletListLine: (String) -> Boolean,
        extractBulletListItem: (String) -> String?,
        looksLikeImagePath: (String, String?) -> Boolean,
        normalizeMatchText: (String) -> String
    ): Pair<List<BookingOption>, Int>? {
        val firstLine = lines[startIndex].trim()
        if (parseOptionLine(firstLine) == null) {
            return null
        }

        val options = mutableListOf<BookingOption>()
        var cursor = startIndex

        while (cursor < lines.size) {
            val current = lines[cursor].trim()
            if (current.isEmpty()) {
                cursor++
                continue
            }

            val parsed = parseOptionLine(current) ?: break
            var next = cursor + 1
            while (next < lines.size && lines[next].trim().isEmpty()) {
                next++
            }

            val button = if (next < lines.size) parseButtonLine(lines[next].trim()) else null
            options += BookingOption(
                title = parsed.first,
                details = parsed.second,
                button = button
            )

            cursor = if (button != null) next + 1 else cursor + 1
            while (cursor < lines.size && lines[cursor].trim().isEmpty()) {
                cursor++
            }

            if (cursor >= lines.size || parseOptionLine(lines[cursor].trim()) == null) {
                break
            }
        }

        val logoBlock = collectTrailingLogoBlock(
            lines = lines,
            startIndex = cursor,
            isBulletListLine = isBulletListLine,
            extractBulletListItem = extractBulletListItem,
            looksLikeImagePath = looksLikeImagePath
        )
        if (logoBlock != null) {
            val (logos, nextCursor) = logoBlock
            if (logos.isNotEmpty()) {
                return attachLogosToOptions(options, logos, normalizeMatchText) to nextCursor
            }
        }

        return options to cursor
    }

    private fun collectTrailingLogoBlock(
        lines: List<String>,
        startIndex: Int,
        isBulletListLine: (String) -> Boolean,
        extractBulletListItem: (String) -> String?,
        looksLikeImagePath: (String, String?) -> Boolean
    ): Pair<List<ParsedLogo>, Int>? {
        var cursor = startIndex
        while (cursor < lines.size && lines[cursor].trim().isEmpty()) {
            cursor++
        }
        if (cursor >= lines.size) {
            return null
        }

        val heading = lines[cursor].trim()
        if (!heading.startsWith("Airline Logos", ignoreCase = true)) {
            return null
        }
        cursor++

        val logos = mutableListOf<ParsedLogo>()
        while (cursor < lines.size) {
            val line = lines[cursor].trim()
            if (line.isEmpty()) {
                cursor++
                continue
            }
            if (!isBulletListLine(line)) {
                break
            }

            val item = extractBulletListItem(line) ?: break
            val parts = item.split(":", limit = 2)
            if (parts.size == 2) {
                val label = parts[0].trim()
                val value = parts[1].trim()
                if (label.isNotEmpty() && looksLikeImagePath(value, label)) {
                    logos += ParsedLogo(label = label, url = value)
                }
            }
            cursor++
        }

        if (logos.isEmpty()) {
            return null
        }
        return logos to cursor
    }

    private fun attachLogosToOptions(
        options: List<BookingOption>,
        logos: List<ParsedLogo>,
        normalizeMatchText: (String) -> String
    ): List<BookingOption> {
        if (options.isEmpty() || logos.isEmpty()) {
            return options
        }

        val remaining = logos.toMutableList()
        return options.map { option ->
            val matched = matchLogoForOption(option, remaining, normalizeMatchText)
            if (matched != null) {
                remaining.remove(matched)
                option.copy(logo = matched)
            } else {
                option
            }
        }
    }

    private fun matchLogoForOption(
        option: BookingOption,
        logos: List<ParsedLogo>,
        normalizeMatchText: (String) -> String
    ): ParsedLogo? {
        if (logos.isEmpty()) {
            return null
        }

        val haystack = normalizeMatchText(option.title + " " + (option.button?.label ?: ""))
        if (haystack.isBlank()) {
            return null
        }

        return logos.firstOrNull { logo ->
            val needle = normalizeMatchText(logo.label)
            needle.isNotBlank() && (haystack.contains(needle) || needle.contains(haystack))
        }
    }
}
