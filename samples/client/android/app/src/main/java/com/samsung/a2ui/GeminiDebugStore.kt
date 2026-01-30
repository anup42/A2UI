package com.samsung.a2ui

data class GeminiDebugSnapshot(
  val requestJson: String?,
  val rawResponse: String?,
  val uiJson: String?
)

object GeminiDebugStore {
  @Volatile
  private var requestJson: String? = null
  @Volatile
  private var rawResponse: String? = null
  @Volatile
  private var uiJson: String? = null

  fun update(requestJson: String?, rawResponse: String?, uiJson: String?) {
    this.requestJson = requestJson
    this.rawResponse = rawResponse
    this.uiJson = uiJson
  }

  fun snapshot(): GeminiDebugSnapshot {
    return GeminiDebugSnapshot(requestJson, rawResponse, uiJson)
  }
}
