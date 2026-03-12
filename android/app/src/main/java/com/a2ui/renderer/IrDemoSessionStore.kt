package com.samsung.genuicraft

object IrDemoSessionStore {
    data class Session(
        val sourceLabel: String,
        val records: List<IrDemoRecord>,
        val loadedAtEpochMs: Long
    )

    @Volatile
    private var currentSession: Session? = null

    fun update(sourceLabel: String, records: List<IrDemoRecord>) {
        currentSession = Session(
            sourceLabel = sourceLabel,
            records = records,
            loadedAtEpochMs = System.currentTimeMillis()
        )
    }

    fun current(): Session? = currentSession

    fun recordAt(index: Int): IrDemoRecord? = currentSession?.records?.getOrNull(index)
}

