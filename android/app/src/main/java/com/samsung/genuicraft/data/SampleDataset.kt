package com.samsung.genuicraft

data class SampleDataset(
    val id: String,
    val title: String,
    val description: String,
    val assetFileName: String,
    val sourceLabel: String,
    val expectedRecords: Int,
    val forceWebRendering: Boolean = false
)

object SampleDatasets {
    val all: List<SampleDataset> = listOf(
        SampleDataset(
            id = "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5_html",
            title = "Golden 50 R5 HTML Mobile Previews",
            description = "Response-derived mobile HTML previews with enriched local images/icons, separate from the native IR renderer.",
            assetFileName = "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5_html_index.jsonl",
            sourceLabel = "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5_html",
            expectedRecords = 50,
            forceWebRendering = true
        ),
        SampleDataset(
            id = "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5",
            title = "Golden 50 Stitch Hybrid R5",
            description = "Copied from the latest Golden50 hybrid run for renderer experiments without changing the original run.",
            assetFileName = "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5_genui.jsonl",
            sourceLabel = "golden50_g25pro_20260309_204033_stitch_compare_20260606_shellcopy_r5",
            expectedRecords = 50
        ),
        SampleDataset(
            id = "golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4",
            title = "Golden 50 Stitch Hybrid R4",
            description = "Latest stitch-guided 50-item IR set with renderer fixes, Android captures, and compact hybrid selection.",
            assetFileName = "golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4_genui.jsonl",
            sourceLabel = "golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4",
            expectedRecords = 50
        )
    )

    fun byId(id: String?): SampleDataset? = all.firstOrNull { it.id == id }
}
