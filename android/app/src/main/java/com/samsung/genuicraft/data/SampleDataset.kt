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
            id = "golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4",
            title = "Golden 50 Stitch Hybrid (Latest)",
            description = "Latest stitch-guided 50-item IR set with renderer fixes, Android captures, and compact hybrid selection.",
            assetFileName = "golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4_genui.jsonl",
            sourceLabel = "golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4",
            expectedRecords = 50
        )
    )

    fun byId(id: String?): SampleDataset? = all.firstOrNull { it.id == id }
}
