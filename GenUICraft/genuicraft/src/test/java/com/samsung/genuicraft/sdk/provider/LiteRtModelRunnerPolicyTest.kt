package com.samsung.genuicraft.sdk.provider

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtModelRunnerPolicyTest {
    @Test
    fun `MTP runs only on GPU when requested`() {
        assertTrue(LiteRtModelPolicy.mtpEnabledForBackend("GPU", requested = true))
        assertFalse(LiteRtModelPolicy.mtpEnabledForBackend("GPU", requested = false))
        assertFalse(LiteRtModelPolicy.mtpEnabledForBackend("CPU", requested = true))
        assertFalse(LiteRtModelPolicy.mtpEnabledForBackend("NPU", requested = true))
    }

    @Test
    fun `automatic backend order preserves GPU then CPU fallback`() {
        assertEquals(
            listOf("GPU", "CPU"),
            LiteRtModelPolicy.backendOrder(forceCpu = false, requireGpu = false),
        )
        assertEquals(
            listOf("GPU"),
            LiteRtModelPolicy.backendOrder(forceCpu = false, requireGpu = true),
        )
        assertEquals(
            listOf("NPU"),
            LiteRtModelPolicy.backendOrder(
                forceCpu = false,
                requireGpu = true,
                accelerator = LiteRtAccelerator.NPU,
            ),
        )
    }

    @Test
    fun `raw training wrapper combines system and user without chat history`() {
        val prepared = LiteRtModelPolicy.preparePrompt(
            request = LiteRtGenerationRequest(
                prompt = "agent response",
                systemPrompt = "stage 3 instruction",
                initialMessages = listOf(
                    LiteRtConversationMessage(LiteRtConversationRole.USER, "ignored example"),
                ),
            ),
            useRawTrainingWrapper = true,
        )

        assertNull(prepared.system)
        assertTrue(prepared.initialMessages.isEmpty())
        assertEquals(
            "<|im_start|>user\nstage 3 instruction\n\nagent response\n" +
                "<|im_start|>assistant\n",
            prepared.user,
        )
    }

    @Test
    fun `stream merge accepts delta and cumulative callback forms`() {
        assertEquals(
            "{\"root\": \"screen\"}",
            LiteRtModelPolicy.mergeStreamText("{\"root\"", ": \"screen\"}"),
        )
        assertEquals(
            "{\"root\": \"screen\"}",
            LiteRtModelPolicy.mergeStreamText("{\"root\"", "{\"root\": \"screen\"}"),
        )
        assertEquals(
            "{\"root\": \"screen\"}",
            LiteRtModelPolicy.mergeStreamText("{\"root\": \"screen\"", "}"),
        )
    }

    @Test
    fun `health applies host model size policy without native initialization`() {
        val directory = Files.createTempDirectory("genuicraft-litert-policy").toFile()
        try {
            val model = File(directory, "model.litertlm").apply { writeBytes(byteArrayOf(1, 2, 3)) }
            val health = LiteRtModelPolicy.checkModelHealth(
                LiteRtModelConfig(
                    modelPath = model.absolutePath,
                    minimumFileSizeBytes = 4,
                    modelDisplayName = "Test model",
                ),
            )

            assertFalse(health.healthy)
            assertEquals(
                "On-device model is incomplete: 3 bytes; expected at least 4 bytes for Test model.",
                health.errorMessage,
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun `runtime label and invalid GPU output policy remain observable`() {
        assertEquals("GPU+MTP", LiteRtModelPolicy.runtimeBackendLabel("GPU", true))
        assertEquals("CPU", LiteRtModelPolicy.runtimeBackendLabel("CPU", false))
        assertTrue(LiteRtModelPolicy.isInvalidGpuOutput("<pad>"))
        assertTrue(LiteRtModelPolicy.isInvalidGpuOutput("<unused0><eos>"))
        assertFalse(LiteRtModelPolicy.isInvalidGpuOutput("{ root = screen }"))
    }
}
