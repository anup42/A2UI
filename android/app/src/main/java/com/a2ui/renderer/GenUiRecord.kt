package com.samsung.genuicraft

import java.io.File

data class GenUiRecord(
    val title: String,
    val rawJson: String,
    val sourceDir: File?,
    val sourceLabel: String,
    val uiId: String?,
    val summary: String?,
    val queryId: String?,
    val responseId: String?
)

