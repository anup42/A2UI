package com.samsung.genuicraft.sdk

import android.content.Context
import java.security.MessageDigest

/** The short training-compatible prompt used by the app's 270M Stage 3 route. */
internal data class Gemma270PromptContract private constructor(
    val system: String,
) : TrainedPrompt {
    override fun prompt(response: String): GenUiPrompt = GenUiPrompt(
        system = system,
        user = TASK_PREFIX + response.trim(),
        maxOutputTokens = GenUiModelProfiles.GEMMA270M_OUTPUT_TOKENS,
        temperature = 0.0,
        initialMessages = listOf(
            GenUiPromptMessage(GenUiPromptRole.USER, EXAMPLE_USER),
            GenUiPromptMessage(GenUiPromptRole.MODEL, EXAMPLE_MODEL),
        ),
        // A null override deliberately selects the Gemma 3 package's native turn template.
        chatTemplateOverride = null,
        expectedRenderedPrompt = null,
        stopAtCompleteExpress = true,
    )

    companion object {
        const val PROFILE = "trained_gemma3_270m"
        const val PROMPT_ASSET = "genuicraft/prompts/gemma3_270m_a2ui_express_training_v1.md"
        private const val SNAPSHOT_SHA256 = "9564619542d60c930a5dfedd43a186040449883db7155b5202ebbc6bf225ee3f"
        const val TASK_PREFIX = "Create A2UI Express v1 GenUI IR for this response:\n\n"
        const val EXAMPLE_USER = TASK_PREFIX + "A small travel checklist with a title and two items."
        const val EXAMPLE_MODEL = "<a2ui>\n" +
            "root=Column([a,b])\n" +
            "a=Text(\"Travel Checklist\",\"h1\")\n" +
            "b=List([c,d])\n" +
            "c=Text(\"Passport\")\n" +
            "d=Text(\"Charger\")\n" +
            "</a2ui>"

        fun load(context: Context): Gemma270PromptContract = parse(
            context.applicationContext.assets.open(PROMPT_ASSET)
                .bufferedReader(Charsets.UTF_8).use { it.readText() },
        )

        fun parse(snapshot: String): Gemma270PromptContract {
            val normalized = snapshot.replace("\r\n", "\n")
            val digest = MessageDigest.getInstance("SHA-256")
                .digest(normalized.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
            require(digest == SNAPSHOT_SHA256) {
                "Gemma 3 270M training prompt differs from the pinned app Stage 3 contract."
            }
            return Gemma270PromptContract(normalized.trim())
        }
    }
}
