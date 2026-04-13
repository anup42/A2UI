package com.samsung.genuicraft

data class SampleDataset(
    val id: String,
    val title: String,
    val description: String,
    val assetFileName: String,
    val sourceLabel: String,
    val expectedRecords: Int
)

object SampleDatasets {
    val all: List<SampleDataset> = listOf(
        SampleDataset(
            id = "subset10",
            title = "Subset 10 Sample",
            description = "10 curated IR items with mixed tables, media, sources, and actions.",
            assetFileName = "sample_genui.jsonl",
            sourceLabel = "sample_genui.jsonl",
            expectedRecords = 10
        ),
        SampleDataset(
            id = "golden50_g25pro_20260309_204033",
            title = "Golden 50 (g25pro)",
            description = "50-item benchmark run from March 9, 2026 with full assets bundle.",
            assetFileName = "golden50_g25pro_20260309_204033_genui.jsonl",
            sourceLabel = "golden50_g25pro_20260309_204033",
            expectedRecords = 50
        ),
        SampleDataset(
            id = "subset10_android_promptsync_20260412_v10",
            title = "Subset10 Android Prompt Sync v10",
            description = "10-item Stage3 run using Android v10 flat-spec prompt sync (Stack/Table canonical output).",
            assetFileName = "subset10_g3pro_iconcatalog_headings_20260216_091411_android_promptsync_20260412_v10_genui.jsonl",
            sourceLabel = "subset10_android_promptsync_20260412_v10",
            expectedRecords = 10
        )
    )

    fun byId(id: String?): SampleDataset? = all.firstOrNull { it.id == id }
}
