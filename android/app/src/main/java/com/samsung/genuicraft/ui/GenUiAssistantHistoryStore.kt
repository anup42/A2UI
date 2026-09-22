package com.samsung.genuicraft

import android.content.Context
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import java.io.File

internal data class GenUiAssistantHistoryItem(
    val id: String,
    val query: String,
    val responseText: String,
    val genUiJson: String,
    val createdAtMs: Long,
    val usedFallback: Boolean = false,
    val warnings: List<String> = emptyList(),
    val stage3InputTokens: Int? = null,
    val stage3OutputTokens: Int? = null,
    val stage3OutputTokensPerSecond: Double? = null,
    val stage3GenerationDurationMs: Long? = null,
)

internal object GenUiAssistantHistoryStore {
    private const val FILE_NAME = "genui_assistant_history.json"
    private const val MAX_ITEMS = 30
    private val gson = GsonBuilder()
        .disableHtmlEscaping()
        .create()
    private val listType = object : TypeToken<List<GenUiAssistantHistoryItem>>() {}.type

    @Synchronized
    fun load(context: Context): List<GenUiAssistantHistoryItem> {
        val file = historyFile(context)
        if (!file.exists()) return emptyList()
        return runCatching {
            gson.fromJson<List<GenUiAssistantHistoryItem>>(file.readText(), listType)
                .orEmpty()
                .filter { it.query.isNotBlank() && it.genUiJson.isNotBlank() }
                .take(MAX_ITEMS)
        }.getOrElse { emptyList() }
    }

    @Synchronized
    fun saveCompleted(
        context: Context,
        item: GenUiAssistantHistoryItem
    ): List<GenUiAssistantHistoryItem> {
        if (item.query.isBlank() || item.responseText.isBlank() || item.genUiJson.isBlank()) {
            return load(context)
        }
        val next = buildList {
            add(item)
            load(context)
                .asSequence()
                .filterNot { existing ->
                    existing.id == item.id ||
                        (existing.query == item.query && existing.genUiJson == item.genUiJson)
                }
                .take(MAX_ITEMS - 1)
                .forEach(::add)
        }
        val file = historyFile(context)
        runCatching {
            file.parentFile?.mkdirs()
            file.writeText(gson.toJson(next))
        }
        return next
    }

    private fun historyFile(context: Context): File {
        return File(context.applicationContext.filesDir, FILE_NAME)
    }
}
