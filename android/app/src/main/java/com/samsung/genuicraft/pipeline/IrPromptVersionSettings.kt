package com.samsung.genuicraft.pipeline

import android.content.Context

internal object IrPromptVersionSettings {

    private const val PREFS_NAME = "ir_prompt_version_settings"
    private const val KEY_SELECTED_VERSION_ID = "selected_version_id"

    data class Option(
        val id: String,
        val title: String,
        val description: String,
        val stage3PromptAssetPath: String
    )

    private val options: List<Option> = listOf(
        Option(
            id = "step1_v3_baseline",
            title = "Step 1 - v3 baseline",
            description = "Schema-safe baseline with local-asset-only constraints.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_v3.md"
        ),
        Option(
            id = "step2_v5_tuned",
            title = "Step 2 - v5 tuned",
            description = "Adds stronger sectioning, CTA, and table decomposition rules.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_v5_tuned.md"
        ),
        Option(
            id = "step3_v6_structured_local",
            title = "Step 3 - v6 structured local",
            description = "Adds strict validation checklist and richer structural quality constraints.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_v6_structured_local.md"
        ),
        Option(
            id = "step4_v6_structured_remote_policy",
            title = "Step 4 - v6 structured remote-aware",
            description = "Adds explicit local-vs-remote asset policy and stronger media colocation.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen_v6_structured_remote.md"
        ),
        Option(
            id = "step5_v7_essential",
            title = "Step 5 - v7 essential",
            description = "Production prompt with strict message contract and interaction constraints.",
            stage3PromptAssetPath = "pipeline_prompts/genui_gen.md"
        )
    )

    private val optionById: Map<String, Option> = options.associateBy { it.id }

    fun options(): List<Option> = options

    fun defaultOption(): Option = options.last()

    fun getSelectedVersionId(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_SELECTED_VERSION_ID, null)?.trim()
        if (!stored.isNullOrBlank() && optionById.containsKey(stored)) {
            return stored
        }
        return defaultOption().id
    }

    fun getSelectedOption(context: Context): Option {
        val id = getSelectedVersionId(context)
        return optionById[id] ?: defaultOption()
    }

    fun setSelectedVersionId(context: Context, versionId: String) {
        val normalized = versionId.trim()
        val resolved = optionById[normalized]?.id ?: defaultOption().id
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SELECTED_VERSION_ID, resolved)
            .apply()
    }

    fun stage3PromptAssetPath(context: Context): String = getSelectedOption(context).stage3PromptAssetPath
}
