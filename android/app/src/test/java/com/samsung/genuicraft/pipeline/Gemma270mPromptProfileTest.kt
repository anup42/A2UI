package com.samsung.genuicraft.pipeline

import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class Gemma270mPromptProfileTest {
    @Test
    fun gemma270mCatalogEntryRequiresGpuAndInt8() {
        val entry = OnDeviceModelCatalog.entryForModelPath(
            "/sdcard/Android/data/com.samsung.genuicraft/files/on_device_models/gemma-3-270m-ir-int8.litertlm"
        )

        requireNotNull(entry)
        assertEquals("gemma3_270m_ir_int8", entry.id)
        assertEquals("INT8 per-channel", entry.quantization)
        assertEquals(4_096, entry.maxContextTokens)
        assertEquals(1_024, entry.maxOutputTokens)
        assertTrue(entry.requireGpu)
        assertTrue(entry.rawStage3Response)
        assertTrue(entry.useRawTrainingWrapper)
        assertFalse(entry.isDownloadable)
    }

    @Test
    fun unknownModelPathDoesNotInheritGpuOnlyProfile() {
        assertNull(OnDeviceModelCatalog.entryForModelPath("/tmp/another-model.litertlm"))
    }

    @Test
    fun rawResponseProfileDoesNotAppendGenericPolicies() {
        val context = PipelinePromptBuilder.prepareStage3PromptContext(
            template = "ignored {response_text} template",
            rawResponseOnly = true,
        )
        val prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = context.userTemplate,
            stage2Response = "  compact response  ",
            catalogId = "ignored",
            assets = emptyList(),
            appendRequestPolicies = false,
        )

        assertNull(context.systemPrompt)
        assertEquals("compact response", prompt)
    }
}
