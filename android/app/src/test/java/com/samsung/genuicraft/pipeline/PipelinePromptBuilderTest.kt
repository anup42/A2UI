package com.samsung.genuicraft.pipeline

import com.samsung.genuicraft.mcp.McpUrlShortener
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelinePromptBuilderTest {
    @Test
    fun stage3UserPromptDoesNotOverrideSelectedFormatWithLegacyFlatSpec() {
        val context = PipelinePromptBuilder.prepareStage3PromptContext(
            template = "Generate A2UI Express. [RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
        )
        val prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = context.userTemplate,
            stage2Response = "Order A-1042 is out for delivery.",
            catalogId = "unused",
            assets = emptyList(),
            outputFormat = GenUiIrFormat.A2UI_EXPRESS_V1,
        )

        assertFalse(prompt.contains("flat-spec JSON object"))
        assertTrue(prompt.contains("selected GenUICraft IR format"))
        assertTrue(prompt.contains("A2UI Express policy"))
        assertTrue(prompt.contains("Order A-1042 is out for delivery."))
        assertTrue(prompt.contains("follow the generated pinned Express contract"))
    }

    @Test
    fun stage3PromptReferencesCanBeFullyMaskedAndRestored() {
        val remoteUrl = "https://example.org/media/icon.svg"
        val localPath = "../assets/icon.svg"
        val rawPrompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = "Response:\n{response_text}",
            stage2Response = "Media: Icon=$remoteUrl Local=$localPath",
            catalogId = "unused",
            assets = listOf(PipelinePromptBuilder.AssetMapping(remoteUrl, localPath)),
        )

        val masked = McpUrlShortener.shorten(rawPrompt)

        assertFalse(masked.shortenedText.contains(remoteUrl))
        assertFalse(masked.shortenedText.contains(localPath))
        assertTrue(masked.shortenedText.contains("{{u1}}"))
        assertTrue(masked.shortenedText.contains("{{u2}}"))
        assertEquals(rawPrompt, McpUrlShortener.restore(masked.shortenedText, masked.urlMap))
    }
}
