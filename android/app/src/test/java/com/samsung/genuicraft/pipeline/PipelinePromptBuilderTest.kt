package com.samsung.genuicraft.pipeline

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
        assertTrue(prompt.contains("openUrl(\"https://...\")"))
        assertTrue(prompt.contains("bare component type such as Icon"))
        assertTrue(prompt.contains("never expose assignment ids or snake_case names"))
        assertTrue(prompt.contains("every carrier, status, ETA"))
        assertTrue(prompt.contains("never an unassigned generic name such as action"))
    }
}
