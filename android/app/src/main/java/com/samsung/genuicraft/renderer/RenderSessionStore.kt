package com.samsung.genuicraft

object RenderSessionStore {
    data class Session(
        val sourceLabel: String,
        val records: List<GenUiRecord>,
        val renderMode: RenderMode,
        val loadedAtEpochMs: Long
    )

    @Volatile
    private var currentSession: Session? = null

    fun update(sourceLabel: String, records: List<GenUiRecord>, renderMode: RenderMode) {
        currentSession = Session(
            sourceLabel = sourceLabel,
            records = records,
            renderMode = renderMode,
            loadedAtEpochMs = System.currentTimeMillis()
        )
    }

    fun current(): Session? = currentSession

    fun recordAt(index: Int): GenUiRecord? = currentSession?.records?.getOrNull(index)
}

