package com.samsung.genuicraft.pipeline

import android.content.Context

/** The sole production model-output contract. Legacy formats have no production setting. */
internal object IrPromptVersionSettings {
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

    fun options(): List<Option> = options
    fun defaultOption(): Option = options.single { it.id == DEFAULT_VERSION_ID }

    fun getSelectedVersionId(context: Context): String = defaultOption().id

    fun getSelectedOption(context: Context): Option = defaultOption()

    /** Retained for migration/test callers; no alternate production format can be selected. */
    fun setSelectedVersionId(context: Context, versionId: String) = Unit

    fun stage3PromptAssetPath(context: Context): String = getSelectedOption(context).stage3PromptAssetPath
    fun outputFormat(context: Context): GenUiIrFormat = getSelectedOption(context).outputFormat
}
