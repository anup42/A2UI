package com.samsung.genuicraft.pipeline

import android.content.Context

/** The production model-output contract. Legacy formats have no production setting. */
internal object IrPromptVersionSettings {
    private const val PREFS_NAME = "ir_prompt_version_settings"
    private const val KEY_SELECTED_VERSION_ID = "selected_version_id"
    private const val DEFAULT_VERSION_ID = "a2ui_express_v1"

    data class Option(
        val id: String,
        val title: String,
        val description: String,
        val stage3PromptAssetPath: String,
        val outputFormat: GenUiIrFormat,
    )

    private val options: List<Option> = listOf(
        Option(
            id = "a2ui_express_v1",
            title = "A2UI Express v1",
            description = "The only supported model-output DSL; it compiles to standard A2UI.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_a2ui_express_v1.md",
            outputFormat = GenUiIrFormat.A2UI_EXPRESS_V1,
        ),
    )

    private val optionById: Map<String, Option> = options.associateBy { it.id }
    fun options(): List<Option> = options
    fun defaultOption(): Option = requireNotNull(optionById[DEFAULT_VERSION_ID])

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
