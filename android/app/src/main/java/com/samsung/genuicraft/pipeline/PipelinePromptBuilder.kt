package com.samsung.genuicraft.pipeline

import android.content.SharedPreferences
import android.content.res.AssetManager

internal object PipelinePromptBuilder {

    const val STAGE2_PROMPT_ASSET = "pipeline_prompts/response_gen.md"
    const val STAGE3_PROMPT_ASSET = "pipeline_prompts/genui_gen.md"

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

    fun prepareStage3PromptContext(template: String): Stage3PromptContext {
        return preparePromptContext(
            template = template,
            placeholder = "{response_text}",
            sentinel = "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
            fallbackUserTemplate =
                "Convert the response text into valid GenUICraft JSON.\n" +
                    "Return ONLY the JSON message array.\n\n" +
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

    fun buildStage3UserPrompt(
        userTemplate: String,
        stage2Response: String,
        catalogId: String,
        assets: List<AssetMapping>
    ): String {
        val assetPolicy = if (assets.isEmpty()) {
            "Asset URL policy for this request:\n" +
                "- No local asset mapping is provided.\n" +
                "- Preserve media URLs from the response exactly as written.\n" +
                "- Do not invent local placeholder paths such as /image.jpg or /asset/foo.png."
        } else {
            "Asset URL policy for this request:\n" +
                "- Use only local media paths from the provided Assets mapping.\n" +
                "- Do not emit remote media URLs for images/icons.\n" +
                "- Do not invent local placeholder paths not present in the mapping."
        }

        val assetContext = if (assets.isEmpty()) {
            ""
        } else {
            val rows = assets.joinToString("\n") { "- ${it.url} -> ${it.localPath}" }
            "Assets (local copies of any URLs in the response; use ONLY these local paths):\n$rows"
        }

        val catalogPolicy = "Catalog policy for this request:\n" +
            "- Use this catalogId in createSurface unless an explicit catalog is provided in context:\n" +
            "  - $catalogId"
        val responseWithPolicy = "${stage2Response.trim()}\n\n$catalogPolicy\n\n$assetPolicy"
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
}
