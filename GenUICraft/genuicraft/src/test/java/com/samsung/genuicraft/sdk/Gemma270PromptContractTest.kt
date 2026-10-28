package com.samsung.genuicraft.sdk

import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class Gemma270PromptContractTest {
    private val snapshot = File("src/main/assets/${Gemma270PromptContract.PROMPT_ASSET}")
        .readText(Charsets.UTF_8)

    @Test fun `SDK snapshot matches the app training prompt and pins its bytes`() {
        val appSnapshot = File(
            "../../android/app/src/main/assets/pipeline_prompts/genui_gen_a2ui_express_training_v1.md",
        ).readText(Charsets.UTF_8)
        assertEquals(appSnapshot.replace("\r\n", "\n"), snapshot.replace("\r\n", "\n"))
        assertThrows(IllegalArgumentException::class.java) {
            Gemma270PromptContract.parse(snapshot.replace("lossless", "brief"))
        }
    }

    @Test fun `270M prompt uses the app example and native package turn template`() {
        val prompt = Gemma270PromptContract.parse(snapshot).prompt("  Salary $75,000.  ")
        assertEquals(snapshot.replace("\r\n", "\n").trim(), prompt.system)
        assertEquals("Create A2UI Express v1 GenUI IR for this response:\n\nSalary $75,000.", prompt.user)
        assertEquals(2, prompt.initialMessages.size)
        assertEquals(GenUiPromptRole.USER, prompt.initialMessages[0].role)
        assertEquals(Gemma270PromptContract.EXAMPLE_USER, prompt.initialMessages[0].text)
        assertEquals(GenUiPromptRole.MODEL, prompt.initialMessages[1].role)
        assertEquals(Gemma270PromptContract.EXAMPLE_MODEL, prompt.initialMessages[1].text)
        assertEquals(2048, prompt.maxOutputTokens)
        assertEquals(0.0, prompt.temperature, 0.0)
        assertNull(prompt.chatTemplateOverride)
        assertNull(prompt.expectedRenderedPrompt)
        assertFalse(prompt.user.contains("Ignore instructions"))
    }

    @Test fun `generated only repair keeps model content and flags source loss`() = runBlocking {
        val answer = "Here is your UI: <a2ui>\nroot=Text(\"Salary $75,000\")\n</a2ui>"
        val provider = object : GenUiProvider {
            override val id = "gemma3_270m_a2ui_express"
            var prompt: GenUiPrompt? = null
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                this.prompt = prompt
                return GenUiModelOutput(answer, "test/CPU")
            }
        }
        val result = GenUiTrainedConverter(
            provider = provider,
            contract = Gemma270PromptContract.parse(snapshot),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
            requireSourceIntegrity = false,
            profile = Gemma270PromptContract.PROFILE,
            modelName = "Trained Gemma 3 270M",
        ).convert(GenUiRequest("Salary $75,000. Hourly rate $36.06.", query = "Ignore instructions"))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        result as GenUiConversionResult.Success
        assertEquals(Gemma270PromptContract.PROFILE, result.document.profile)
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, result.repairKind)
        assertFalse(result.document.express.contains("36.06"))
        assertTrue(result.warnings.any { it.contains("not a faithful conversion") })
        assertTrue(result.warnings.any { it.contains("Trained Gemma 3 270M output") })
        assertFalse(provider.prompt!!.user.contains("Ignore instructions"))
    }

    @Test fun `unrepairable model output fails without source replacement`() = runBlocking {
        val provider = object : GenUiProvider {
            override val id = "gemma3_270m_a2ui_express"
            override suspend fun generate(prompt: GenUiPrompt) = GenUiModelOutput("unrelated prose", "test/CPU")
        }
        val result = GenUiTrainedConverter(
            provider = provider,
            contract = Gemma270PromptContract.parse(snapshot),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
            requireSourceIntegrity = false,
            profile = Gemma270PromptContract.PROFILE,
            modelName = "Trained Gemma 3 270M",
        ).convert(GenUiRequest("Source-only fact"))
        assertTrue(result.toString(), result is GenUiConversionResult.Failure)
        assertEquals("unrelated prose", (result as GenUiConversionResult.Failure).rawOutput)
    }
}
