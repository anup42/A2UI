package com.samsung.genuicraft

object RenderSessionStore {
    data class Session(
        val sourceLabel: String,
        val records: List<GenUiRecord>,
        val loadedAtEpochMs: Long
    )

    @Volatile
    private var currentSession: Session? = null

    fun update(sourceLabel: String, records: List<GenUiRecord>) {
        currentSession = Session(
            sourceLabel = sourceLabel,
            records = records,
            loadedAtEpochMs = System.currentTimeMillis()
        )
    }

    fun current(): Session? = currentSession

    fun recordAt(index: Int): GenUiRecord? = currentSession?.records?.getOrNull(index)
}

