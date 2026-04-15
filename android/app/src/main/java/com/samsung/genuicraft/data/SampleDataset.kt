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
            id = "golden50_g25pro_20260309_204033_stitch_compare_20260415",
            title = "Golden 50 Stitch Compare (Latest)",
            description = "Latest stitch-guided 50-item IR run with improved fallback handling and table extraction.",
            assetFileName = "golden50_g25pro_20260309_204033_stitch_compare_20260415_genui.jsonl",
            sourceLabel = "golden50_g25pro_20260309_204033_stitch_compare_20260415",
            expectedRecords = 50
        ),
        SampleDataset(
            id = "subset10_latest_ir",
            title = "Subset 10 Latest IR",
            description = "Latest 10-item subset run synced from Android prompt updates.",
            assetFileName = "subset10_g3pro_iconcatalog_headings_20260216_091411_latest_ir_genui.jsonl",
            sourceLabel = "subset10_latest_ir",
            expectedRecords = 10
        )
    )

    fun byId(id: String?): SampleDataset? = all.firstOrNull { it.id == id }
}
