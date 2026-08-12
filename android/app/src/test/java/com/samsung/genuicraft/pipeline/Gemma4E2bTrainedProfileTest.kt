package com.samsung.genuicraft.pipeline

import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.GenUiNativeRenderer
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class Gemma4E2bTrainedProfileTest {
    @Test
    fun trainedExpressCatalogEntryUsesTheMobileQatPromptAndMtpProfile() {
        val entry = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath(
                "/sdcard/Android/data/com.samsung.genuicraft/files/on_device_models/" +
                    "gemma-4-e2b-trained-express-int4.litertlm"
            )
        )

        assertEquals("gemma4_e2b_trained_express", entry.id)
        assertEquals("Gemma 4 E2B Trained Express", entry.displayName)
        assertEquals("Mixed W2/W4/W8-A8 mobile topology", entry.quantization)
        assertEquals(4_096, entry.maxContextTokens)
        assertEquals(2_048, entry.maxOutputTokens)
        assertTrue(entry.requireGpu)
        assertTrue(entry.enableSpeculativeDecoding)
        assertTrue(entry.trainingCompatiblePrompt)
        assertFalse(entry.rawStage3Response)
        assertFalse(entry.useRawTrainingWrapper)
        assertNull(entry.stage3TrainingPromptPrefix)
        assertEquals(2_500_000_000L, entry.minimumFileSizeBytes)
        assertFalse(entry.isDownloadable)
    }

    @Test
    fun trainedExpressPromptMatchesTheMobileQatTrainingMessages() {
        val entry = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath(
                "/tmp/gemma-4-e2b-trained-express-int4.litertlm"
            )
        )
        val context = PipelinePromptBuilder.prepareStage3PromptContext(
            template =
                "You convert response text into A2UI Express v1. " +
                    "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
            trainingCompatible = entry.trainingCompatiblePrompt,
        )
        val prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = context.userTemplate,
            stage2Response = "Order A-1042 is out for delivery.",
            catalogId = "ignored",
            assets = emptyList(),
            appendRequestPolicies = false,
        )

        assertEquals(
            "Create A2UI Express v1 GenUI IR for this response:\n\n" +
                "Order A-1042 is out for delivery.",
            prompt,
        )
        assertEquals(2, context.initialMessages.size)
        assertTrue(context.initialMessages.first().content.contains("travel checklist"))
        assertTrue(context.initialMessages.last().content.startsWith("<a2ui>"))
        assertTrue(context.systemPrompt.orEmpty().contains("A2UI Express v1"))
    }

    @Test
    fun catalogEntryUsesRawResponseWithNativeGemmaChatTemplate() {
        val entry = OnDeviceModelCatalog.entryForModelPath(
            "/sdcard/Android/data/com.samsung.genuicraft/files/on_device_models/gemma-4-e2b-ir-trained-int4.litertlm"
        )

        requireNotNull(entry)
        assertEquals("gemma4_e2b_ir_trained_int4", entry.id)
        assertEquals("INT4/INT8 attention-protected blockwise-32", entry.quantization)
        assertEquals(4_096, entry.maxContextTokens)
        assertEquals(2_048, entry.maxOutputTokens)
        assertTrue(entry.requireGpu)
        assertFalse(entry.enableSpeculativeDecoding)
        assertTrue(entry.rawStage3Response)
        assertFalse(entry.useRawTrainingWrapper)
        assertEquals(
            "Given an agent response you have to generate a structured intermediate representation. ",
            entry.stage3TrainingPromptPrefix,
        )
        assertFalse(entry.isDownloadable)
    }

    @Test
    fun pretrainedGemma4E2bEnablesMtpSpeculativeDecoding() {
        val entry = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath("/tmp/gemma-4-E2B-it.litertlm")
        )

        assertEquals("gemma4_e2b_it_litert", entry.id)
        assertTrue(entry.enableSpeculativeDecoding)
        assertFalse(entry.rawStage3Response)
        assertNull(entry.stage3TrainingPromptPrefix)
    }

    @Test
    fun v6A2uiExpressEntryMatchesThePackagedAndroidArtifact() {
        val entry = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath(
                "/sdcard/Android/data/com.samsung.genuicraft/files/on_device_models/gemma-4-e2b-a2ui-express-v6.litertlm"
            )
        )

        assertEquals("gemma4_e2b_a2ui_express_v6_litert", entry.id)
        assertEquals("INT4 weights / FP32 activations, blockwise-32", entry.quantization)
        assertEquals(4_096, entry.maxContextTokens)
        assertEquals(2_048, entry.maxOutputTokens)
        assertTrue(entry.requireGpu)
        assertFalse(entry.enableSpeculativeDecoding)
        assertFalse(entry.rawStage3Response)
        assertTrue(entry.trainingCompatiblePrompt)
        assertEquals(2_500_000_000L, entry.minimumFileSizeBytes)
        assertFalse(entry.isDownloadable)
    }

    @Test
    fun stage3PromptMatchesTrainingInferenceFormat() {
        val entry = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath("/tmp/gemma-4-e2b-ir-trained-int4.litertlm")
        )
        val context = PipelinePromptBuilder.prepareStage3PromptContext(
            template = "ignored {response_text} template",
            rawResponseOnly = entry.rawStage3Response,
            rawResponsePrefix = entry.stage3TrainingPromptPrefix,
        )
        val prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = context.userTemplate,
            stage2Response = "  ## Service Status\nThe checkout service is online and healthy.  ",
            catalogId = "ignored",
            assets = emptyList(),
            appendRequestPolicies = false,
        )

        assertNull(context.systemPrompt)
        assertEquals(
            "Given an agent response you have to generate a structured intermediate representation. " +
                "## Service Status\nThe checkout service is online and healthy.",
            prompt,
        )
    }

    @Test
    fun exactTrainingInstructionLeakIsRemovedFromGeneratedIr() {
        val prefix =
            "Given an agent response you have to generate a structured intermediate representation. "
        val raw =
            "{\"props\":{\"text\":\"Given an agent response you have to generate a structured intermediate representation.\"}}"

        val cleaned = PipelinePromptBuilder.stripStage3TrainingPromptLeak(raw, prefix)

        assertFalse(cleaned.contains(prefix.trim()))
        assertEquals("{\"props\":{\"text\":\"\"}}", cleaned)
    }

    @Test
    fun expressGpuGeneratedIrPassesContractAndRenderer() {
        val generatedIr = """
            {
              "root": "main_stack",
              "state": {},
              "elements": {
                "main_stack": {
                  "type": "Stack",
                  "props": { "direction": "vertical", "gap": "md", "padding": 16 },
                  "children": ["status_card", "checkout_card"]
                },
                "status_card": {
                  "type": "Card",
                  "props": {},
                  "children": ["status_text"]
                },
                "status_text": {
                  "type": "Text",
                  "props": { "text": "The checkout service is online and healthy.", "variant": "body" },
                  "children": []
                },
                "checkout_card": {
                  "type": "Card",
                  "props": {},
                  "children": []
                }
              }
            }
        """.trimIndent()

        val express = A2uiExpressCodec.encode(JsonParser.parseString(generatedIr).asJsonObject)
        val ingested = FlatSpecIngestor.ingest(JsonPrimitive(express), FlatSpecIngestMode.STRICT)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)

        val rendered = GenUiNativeRenderer.render(express, sourceDir = null)
        assertNull(rendered.errorMessage)
        assertEquals(1, rendered.surfaces.size)
        assertEquals("root", rendered.surfaces.single().rootId)
        assertTrue(
            rendered.surfaces.single().flatSpec?.elements?.values?.any { element ->
                (element.props["text"] as? String)?.contains("checkout service", ignoreCase = true) == true
            } == true,
        )
    }
}
