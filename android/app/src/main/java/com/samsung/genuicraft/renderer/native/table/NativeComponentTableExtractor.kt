package com.samsung.genuicraft.renderer.native.table

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.renderer.native.FlightTableColumns
import com.samsung.genuicraft.renderer.native.TableCell
import com.samsung.genuicraft.renderer.native.TableSpec
import com.samsung.genuicraft.renderer.native.WeatherTableColumns
import java.util.Locale

internal object NativeComponentTableExtractor {
    fun maybeExtractTable(
        children: List<String>,
        index: Map<String, JsonObject>,
        readChildren: (JsonObject) -> List<String>,
        getString: (JsonObject, String) -> String?,
        readDynamicString: (JsonElement?) -> String,
        getAsNumberOrNull: (JsonObject, String) -> Double?,
        detectWeatherColumns: (List<String>) -> WeatherTableColumns?,
        detectFlightColumns: (List<String>) -> FlightTableColumns?,
        isPlaceholderTableCellRow: (List<TableCell>) -> Boolean
    ): TableSpec? {
        val extracted = maybeExtractTableRun(
            children = children,
            startIndex = 0,
            index = index,
            readChildren = readChildren,
            getString = getString,
            readDynamicString = readDynamicString,
            getAsNumberOrNull = getAsNumberOrNull,
            detectWeatherColumns = detectWeatherColumns,
            detectFlightColumns = detectFlightColumns,
            isPlaceholderTableCellRow = isPlaceholderTableCellRow
        ) ?: return null
        val (tableSpec, consumed) = extracted
        if (consumed != children.size) {
            return null
        }
        return tableSpec
    }

    fun maybeExtractTableRun(
        children: List<String>,
        startIndex: Int,
        index: Map<String, JsonObject>,
        readChildren: (JsonObject) -> List<String>,
        getString: (JsonObject, String) -> String?,
        readDynamicString: (JsonElement?) -> String,
        getAsNumberOrNull: (JsonObject, String) -> Double?,
        detectWeatherColumns: (List<String>) -> WeatherTableColumns?,
        detectFlightColumns: (List<String>) -> FlightTableColumns?,
        isPlaceholderTableCellRow: (List<TableCell>) -> Boolean
    ): Pair<TableSpec, Int>? {
        if (startIndex !in children.indices) {
            return null
        }

        val rowComponents = mutableListOf<JsonObject>()
        var cursor = startIndex
        while (cursor < children.size) {
            val child = index[children[cursor]] ?: return null
            if (getString(child, "component") != "Row") {
                break
            }
            rowComponents += child
            cursor++
        }
        if (rowComponents.size < 2) {
            return null
        }

        val rowCells = rowComponents.map(readChildren)
        if (rowCells.any { it.isEmpty() }) {
            return null
        }

        val columnCount = rowCells.first().size
        if (columnCount < 2 || rowCells.any { it.size != columnCount }) {
            return null
        }

        var hasWeight = false
        val textRows = mutableListOf<List<TableCell>>()
        for (cellIds in rowCells) {
            val cells = mutableListOf<TableCell>()
            for (cellId in cellIds) {
                val cell = index[cellId] ?: return null
                if (getString(cell, "component") != "Text") {
                    return null
                }

                val text = readDynamicString(cell.get("text"))
                val variant = (getString(cell, "variant") ?: "body").lowercase(Locale.US)
                val weight = (getAsNumberOrNull(cell, "weight") ?: 1.0).toFloat().coerceAtLeast(0.6f)
                if (weight > 1f) hasWeight = true
                cells += TableCell(text, variant, weight)
            }
            textRows += cells
        }

        val headerLikeVariants = setOf("h1", "h2", "h3", "h4", "h5", "h6")
        val headerLike = textRows.first().all { it.variant in headerLikeVariants } ||
            rowCells.first().all { it.startsWith("th-", ignoreCase = true) || it.startsWith("th_", ignoreCase = true) }
        val weatherHeaderLike = detectWeatherColumns(textRows.first().map { it.text }) != null
        val flightHeaderLike = detectFlightColumns(textRows.first().map { it.text }) != null
        if (!hasWeight && !headerLike && !weatherHeaderLike && !flightHeaderLike) {
            return null
        }

        val header = if (headerLike || weatherHeaderLike || flightHeaderLike) textRows.first() else null
        val body = if (header != null) textRows.drop(1) else textRows
        val filteredBody = body.filterNot(isPlaceholderTableCellRow)
        if (filteredBody.isEmpty()) {
            return null
        }
        return TableSpec(header = header, rows = filteredBody) to (cursor - startIndex)
    }

    fun maybeExtractCardRowTable(
        children: List<String>,
        index: Map<String, JsonObject>,
        readChildren: (JsonObject) -> List<String>,
        getString: (JsonObject, String) -> String?,
        readDynamicString: (JsonElement?) -> String,
        getAsNumberOrNull: (JsonObject, String) -> Double?,
        detectWeatherColumns: (List<String>) -> WeatherTableColumns?,
        detectFlightColumns: (List<String>) -> FlightTableColumns?,
        isPlaceholderTableCellRow: (List<TableCell>) -> Boolean
    ): TableSpec? {
        if (children.size < 2) {
            return null
        }

        val headerRowId = children.first()
        val headerCells = extractTableCellsFromRow(
            rowId = headerRowId,
            index = index,
            readChildren = readChildren,
            getString = getString,
            readDynamicString = readDynamicString,
            getAsNumberOrNull = getAsNumberOrNull
        ) ?: return null
        if (headerCells.size < 2) {
            return null
        }

        val body = mutableListOf<List<TableCell>>()
        for (childId in children.drop(1)) {
            val child = index[childId] ?: return null
            val rowCells = when (getString(child, "component")) {
                "Row" -> extractTableCellsFromRow(
                    rowId = childId,
                    index = index,
                    readChildren = readChildren,
                    getString = getString,
                    readDynamicString = readDynamicString,
                    getAsNumberOrNull = getAsNumberOrNull
                )

                "Card" -> {
                    val directChild = getString(child, "child")
                    when {
                        !directChild.isNullOrBlank() -> extractTableCellsFromRow(
                            rowId = directChild,
                            index = index,
                            readChildren = readChildren,
                            getString = getString,
                            readDynamicString = readDynamicString,
                            getAsNumberOrNull = getAsNumberOrNull
                        )

                        else -> {
                            val nested = readChildren(child)
                            if (nested.size == 1) {
                                extractTableCellsFromRow(
                                    rowId = nested.first(),
                                    index = index,
                                    readChildren = readChildren,
                                    getString = getString,
                                    readDynamicString = readDynamicString,
                                    getAsNumberOrNull = getAsNumberOrNull
                                )
                            } else {
                                null
                            }
                        }
                    }
                }

                else -> null
            } ?: return null

            if (rowCells.size != headerCells.size) {
                return null
            }
            body += rowCells
        }

        val filteredBody = body.filterNot(isPlaceholderTableCellRow)
        if (filteredBody.isEmpty()) {
            return null
        }

        val headerText = headerCells.map { it.text }
        val weatherHeaderLike = detectWeatherColumns(headerText) != null
        val flightHeaderLike = detectFlightColumns(headerText) != null
        if (!weatherHeaderLike && !flightHeaderLike) {
            return null
        }

        return TableSpec(
            header = headerCells,
            rows = filteredBody
        )
    }

    fun maybeExtractTextList(
        children: List<String>,
        index: Map<String, JsonObject>,
        getString: (JsonObject, String) -> String?,
        readDynamicString: (JsonElement?) -> String
    ): List<String>? {
        if (children.size < 2) {
            return null
        }

        val items = mutableListOf<String>()
        for (childId in children) {
            val child = index[childId] ?: return null
            if (getString(child, "component") != "Text") {
                return null
            }
            items += readDynamicString(child.get("text"))
        }
        return items
    }

    private fun extractTableCellsFromRow(
        rowId: String,
        index: Map<String, JsonObject>,
        readChildren: (JsonObject) -> List<String>,
        getString: (JsonObject, String) -> String?,
        readDynamicString: (JsonElement?) -> String,
        getAsNumberOrNull: (JsonObject, String) -> Double?
    ): List<TableCell>? {
        val row = index[rowId] ?: return null
        if (getString(row, "component") != "Row") {
            return null
        }

        val cellIds = readChildren(row)
        if (cellIds.isEmpty()) {
            return null
        }

        val cells = mutableListOf<TableCell>()
        cellIds.forEach { cellId ->
            val cell = index[cellId] ?: return null
            if (getString(cell, "component") != "Text") {
                return null
            }

            cells += TableCell(
                text = readDynamicString(cell.get("text")),
                variant = (getString(cell, "variant") ?: "body").lowercase(Locale.US),
                weight = (getAsNumberOrNull(cell, "weight") ?: 1.0).toFloat().coerceAtLeast(0.6f)
            )
        }
        return cells
    }
}
