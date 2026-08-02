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
