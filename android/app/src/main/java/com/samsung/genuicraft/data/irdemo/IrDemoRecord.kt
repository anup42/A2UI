package com.samsung.genuicraft

data class IrDemoRecord(
    val queryId: String,
    val responseId: String?,
    val queryText: String,
    val responseText: String,
    val genUiJson: String? = null,
    val assetsJson: String? = null,
    val sourceDirPath: String? = null
) {
    val hasSavedIr: Boolean
        get() = !genUiJson.isNullOrBlank()
}

