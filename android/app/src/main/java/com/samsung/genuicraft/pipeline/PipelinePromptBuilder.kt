package com.samsung.genuicraft.pipeline

import android.content.SharedPreferences
import android.content.res.AssetManager

internal object PipelinePromptBuilder {

    const val STAGE2_PROMPT_ASSET = "pipeline_prompts/response_gen.md"
    const val STAGE3_PROMPT_ASSET = "pipeline_prompts/genui_gen.md"
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
        val userTemplate: String
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
    ): Stage3PromptContext {
        if (rawResponseOnly) {
            return Stage3PromptContext(
                systemPrompt = null,
                userTemplate = "${rawResponsePrefix.orEmpty()}{response_text}",
            )
        }
        return preparePromptContext(
            template = template,
            placeholder = "{response_text}",
            sentinel = "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
            fallbackUserTemplate =
                "Convert the response text into a GenUICraft flat-spec JSON object.\n" +
                    "Return ONLY the JSON object (no prose, no fences).\n\n" +
                    "Response:\n{response_text}"
        ).let { context ->
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
        outputFormat: GenUiIrFormat = GenUiIrFormat.COMPACT_IR_V2,
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
                "- Preserve media URLs from the response exactly as written.\n" +
                "- If the response contains compact URL placeholders like {{u1}}, {{u2}}, preserve them exactly in media/action/source fields; they represent verified URLs and will be restored after JSON generation.\n" +
                "- Do not invent local placeholder paths such as /image.jpg or /asset/foo.png."
        } else {
            "Asset URL policy for this request:\n" +
                "- Use only local media paths from the provided Assets mapping.\n" +
                "- If the response contains compact URL placeholders like {{u1}}, {{u2}}, preserve them exactly in action/source fields; they represent verified URLs and will be restored after JSON generation.\n" +
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
            GenUiIrFormat.A2UI_EXPRESS_V1 -> "A2UI Express policy for this request:\n" +
                "- Return one <a2ui>...</a2ui> block and no prose.\n" +
                "- Assign the root component to reserved variable root.\n" +
                "- Preserve rich UI structure and all requested interactions.\n" +
                "- Use _props/_children/_repeat/_visible/_on/_watch when required.\n" +
                "- Every component reference must resolve."
            else -> "Compact IR v2 policy for this request:\n" +
                "- Return ONE JSON object with v=\"gci2\", r, optional s, and e.\n" +
                "- Elements use t and optional p/c/x/z/o/w.\n" +
                "- Omit empty fields but preserve all semantic UI components.\n" +
                "- Every referenced id must exist in e.\n" +
                "- Return JSON only (no prose, markdown, or fences)."
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
