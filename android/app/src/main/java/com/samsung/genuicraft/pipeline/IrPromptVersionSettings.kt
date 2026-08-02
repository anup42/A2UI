package com.samsung.genuicraft.pipeline

import android.content.Context

/** Model-output format selection. Legacy FlatSpec assets remain in the repo for migration only. */
internal object IrPromptVersionSettings {
    private const val PREFS_NAME = "ir_prompt_version_settings"
    private const val KEY_SELECTED_VERSION_ID = "selected_version_id"

    data class Option(
        val id: String,
        val title: String,
        val description: String,
        val stage3PromptAssetPath: String,
        val outputFormat: GenUiIrFormat,
    )

    private val options: List<Option> = listOf(
        Option(
            id = "compact_ir_v2",
            title = "Compact IR v2",
            description = "Default JSON-constrained format. Preserves full UI richness while omitting empty/default syntax.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_compact_ir_v2.md",
            outputFormat = GenUiIrFormat.COMPACT_IR_V2,
        ),
        Option(
            id = "a2ui_express_v1",
            title = "A2UI Express v1",
            description = "Pinned compact DSL with the strongest measured token reduction.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_a2ui_express_v1.md",
            outputFormat = GenUiIrFormat.A2UI_EXPRESS_V1,
        ),
    )

    private val optionById: Map<String, Option> = options.associateBy { it.id }
    fun options(): List<Option> = options
    fun defaultOption(): Option = options.first()

    fun getSelectedVersionId(context: Context): String {
        val stored = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_SELECTED_VERSION_ID, null)?.trim()
        return stored?.takeIf(optionById::containsKey) ?: defaultOption().id
    }

    fun getSelectedOption(context: Context): Option = optionById[getSelectedVersionId(context)] ?: defaultOption()

    fun setSelectedVersionId(context: Context, versionId: String) {
        val resolved = optionById[versionId.trim()]?.id ?: defaultOption().id
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .putString(KEY_SELECTED_VERSION_ID, resolved).apply()
    }

    fun stage3PromptAssetPath(context: Context): String = getSelectedOption(context).stage3PromptAssetPath
    fun outputFormat(context: Context): GenUiIrFormat = getSelectedOption(context).outputFormat
}
