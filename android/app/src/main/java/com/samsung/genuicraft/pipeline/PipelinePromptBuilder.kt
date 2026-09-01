package com.samsung.genuicraft.pipeline

import android.content.SharedPreferences
import android.content.res.AssetManager
import com.samsung.genuicraft.inference.InferenceBackend

internal object PipelinePromptBuilder {

    const val STAGE2_PROMPT_ASSET = "pipeline_prompts/response_gen.md"
    const val STAGE3_PROMPT_ASSET = "pipeline_prompts/genui_gen.md"
    const val STAGE3_A2UI_EXPRESS_PROMPT_ASSET = "pipeline_prompts/genui_gen_a2ui_express_v1.md"
    const val STAGE3_A2UI_EXPRESS_TRAINING_PROMPT_ASSET =
        "pipeline_prompts/genui_gen_a2ui_express_training_v1.md"
    const val STAGE3_GEMMA_PROMPT_ASSET = "pipeline_prompts/genui_gen_gemma_litert.md"

    data class PromptContext(
        val systemPrompt: String?,
        val userTemplate: String
    )

    data class Stage2PromptContext(
        val systemPrompt: String?,
        val userTemplate: String
    )

    data class Stage3PromptContext(
        val systemPrompt: String?,
        val userTemplate: String,
        val initialMessages: List<InferenceBackend.ConversationMessage> = emptyList(),
    )

    data class AssetMapping(
        val url: String,
        val localPath: String
    )

    fun loadPromptAsset(assets: AssetManager, path: String): String {
        return assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    fun prepareStage2PromptContext(template: String): Stage2PromptContext {
        return preparePromptContext(
            template = template,
            placeholder = "{query_text}",
            sentinel = "[QUERY_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
            fallbackUserTemplate =
                "Answer the following user query with a complete, user-ready response:\n" +
                    "{query_text}"
        ).let { context ->
            Stage2PromptContext(
                systemPrompt = context.systemPrompt,
                userTemplate = context.userTemplate
            )
        }
    }

    fun prepareStage3PromptContext(
        template: String,
        rawResponseOnly: Boolean = false,
        rawResponsePrefix: String? = null,
        trainingCompatible: Boolean = false,
    ): Stage3PromptContext {
        if (rawResponseOnly) {
            return Stage3PromptContext(
                systemPrompt = null,
                userTemplate = "${rawResponsePrefix.orEmpty()}{response_text}",
            )
        }
        val prepared = preparePromptContext(
            template = template,
            placeholder = "{response_text}",
            sentinel = "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
            fallbackUserTemplate =
                "Convert the response text into the selected GenUICraft IR format described " +
                    "by the system instructions.\n" +
                    "Return ONLY the requested payload (no prose and no markdown fences).\n\n" +
                    "Response:\n{response_text}"
        )
        if (trainingCompatible) {
            return Stage3PromptContext(
                systemPrompt = prepared.systemPrompt,
                userTemplate = "Create A2UI Express v1 GenUI IR for this response:\n\n{response_text}",
                initialMessages = listOf(
                    InferenceBackend.ConversationMessage(
                        role = InferenceBackend.ConversationRole.USER,
                        content =
                            "Create A2UI Express v1 GenUI IR for this response:\n\n" +
                                "A small travel checklist with a title and two items.",
                    ),
                    InferenceBackend.ConversationMessage(
                        role = InferenceBackend.ConversationRole.MODEL,
                        content =
                            "<a2ui>\n" +
                                "root=Column([a,b])\n" +
                                "a=Text(\"Travel Checklist\",\"h1\")\n" +
                                "b=List([c,d])\n" +
                                "c=Text(\"Passport\")\n" +
                                "d=Text(\"Charger\")\n" +
                                "</a2ui>",
                    ),
                ),
            )
        }
        return prepared.let { context ->
            Stage3PromptContext(
                systemPrompt = context.systemPrompt,
                userTemplate = context.userTemplate
            )
        }
    }

    private fun preparePromptContext(
        template: String,
        placeholder: String,
        sentinel: String,
        fallbackUserTemplate: String
    ): PromptContext {
        if (!template.contains(placeholder)) {
            return PromptContext(
                systemPrompt = template.trim(),
                userTemplate = fallbackUserTemplate
            )
        }
        val split = template.split(placeholder, limit = 2)
        val systemPrompt = "${split[0]}$sentinel${split[1]}".trim()
        return PromptContext(
            systemPrompt = systemPrompt,
            userTemplate = fallbackUserTemplate
        )
    }

    @Suppress("UNUSED_PARAMETER")
    fun buildStage3UserPrompt(
        userTemplate: String,
        stage2Response: String,
        catalogId: String,
        assets: List<AssetMapping>,
        outputFormat: GenUiIrFormat = GenUiIrFormat.A2UI_EXPRESS_V1,
        appendRequestPolicies: Boolean = true,
    ): String {
        if (!appendRequestPolicies) {
            return renderPrompt(
                template = userTemplate,
                values = mapOf("response_text" to stage2Response.trim()),
            )
        }
        val assetPolicy = if (assets.isEmpty()) {
            "Asset URL policy for this request:\n" +
                "- No local asset mapping is provided.\n" +
                "- Preserve reference placeholders from the response exactly as written.\n" +
                "- Placeholders like {{u1}}, {{u2}} can represent URLs or local asset paths. Preserve them exactly in media/action/source fields; they will be restored after compilation.\n" +
                "- Do not invent local media paths that are absent from the response."
        } else {
            "Asset URL policy for this request:\n" +
                "- Use only local media paths from the provided Assets mapping.\n" +
                "- Placeholders like {{u1}}, {{u2}} can represent URLs or local asset paths. Preserve them exactly in media/action/source fields; they will be restored after compilation.\n" +
                "- Do not emit remote media URLs for images/icons.\n" +
                "- Do not invent local placeholder paths not present in the mapping."
        }

        val assetContext = if (assets.isEmpty()) {
            ""
        } else {
            val rows = assets.joinToString("\n") { "- ${it.url} -> ${it.localPath}" }
            "Assets (local copies of any URLs in the response; use ONLY these local paths):\n$rows"
        }

        val formatPolicy = when (outputFormat) {
            GenUiIrFormat.A2UI_EXPRESS_V1 ->
                "A2UI Express policy for this request: follow the generated " +
                    "pinned Express contract in the system prompt exactly; " +
                    "preserve all semantics and return no prose."
            else -> error("Only A2UI Express is a production model-output format.")
        }
        val responseWithPolicy = "${stage2Response.trim()}\n\n$formatPolicy\n\n$assetPolicy"
        val responseText = if (assetContext.isBlank()) {
            responseWithPolicy
        } else {
            "$responseWithPolicy\n\n$assetContext"
        }

        return renderPrompt(template = userTemplate, values = mapOf("response_text" to responseText))
    }

    fun renderPrompt(template: String, values: Map<String, String>): String {
        var rendered = template
        values.forEach { (key, value) ->
            rendered = rendered.replace("{$key}", value)
        }
        return rendered
    }

    fun stripStage3TrainingPromptLeak(
        responseText: String,
        trainingPromptPrefix: String?,
    ): String {
        val leakedInstruction = trainingPromptPrefix?.trim().orEmpty()
        return if (leakedInstruction.isBlank()) {
            responseText
        } else {
            responseText.replace(leakedInstruction, "")
        }
    }
}
