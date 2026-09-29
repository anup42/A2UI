package com.samsung.genuicraft.sdk.provider

import com.google.gson.JsonObject
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GpuFp16CorrectionTest {
    @get:Rule val temporary = TemporaryFolder()

    private fun fixture(drafter: Boolean = true): Pair<File, JsonObject> {
        val model = temporary.newFile("model-${System.nanoTime()}.litertlm").apply { writeText("immutable trained weights") }
        val record = JsonObject().apply {
            addProperty("schema_version", 1)
            addProperty("source_sha256", GpuFp16Correction.SOURCE_SHA256)
            addProperty("transform_version", "pinned-r64-rope-target-mtp-8192-v1")
            addProperty("precision_policy", GpuFp16Correction.POLICY)
            addProperty("target_rope_corrected", true)
            addProperty("drafter_rope_corrected", drafter)
            addProperty("max_context_tokens", 8192)
            addProperty("model_bytes", model.length())
            addProperty("model_sha256", MessageDigest.getInstance("SHA-256").digest(model.readBytes()).joinToString("") { "%02x".format(it) })
        }
        File(model.path + ".fp16.json").writeText(record.toString())
        return model to record
    }

    private fun config(file: File, mtp: Boolean = true, context: Int = 8192) = ValidatedGemma4Config(
        modelPath = file.path, accelerator = Gemma4Accelerator.GPU,
        maxContextTokens = context, maxOutputTokens = 2048, cpuThreads = 4,
        cacheDir = null, enableThinking = false, thinkingTokenBudget = 0,
        enableSpeculativeDecoding = mtp, enableMetrics = true,
        gpuPrecision = Gemma4GpuPrecision.FP16_CORRECTED,
    )

    private fun validateFixture(model: File, mtp: Boolean = true, context: Int = 8192): File =
        GpuFp16Correction.validateModel(config(model, mtp, context),
            MessageDigest.getInstance("SHA-256").digest("immutable trained weights".toByteArray())
                .joinToString("") { "%02x".format(it) })

    @Test fun unpinnedPayloadCannotOptIntoCorrectedFp16() {
        val (model, _) = fixture()
        assertThrows(IllegalArgumentException::class.java) { GpuFp16Correction.validateModel(config(model)) }
    }

    @Test fun validatesTheActualModelPayload() {
        val (model, _) = fixture()
        assertEquals(model.canonicalFile, validateFixture(model))
        val modified = model.lastModified()
        model.writeText("different trained weights")
        model.setLastModified(modified + 2000)
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model) }
    }

    @Test fun rejectsUnverifiedDrafterWhenMtpRequested() {
        val (model, _) = fixture(drafter = false)
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model) }
        assertEquals(model.canonicalFile, validateFixture(model, mtp = false))
    }

    @Test fun enforcesLookupBoundsAndManifestVersion() {
        val (model, record) = fixture()
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model, context = 16384) }
        record.addProperty("precision_policy", "unverified-half-mode")
        File(model.path + ".fp16.json").writeText(record.toString())
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model) }
    }

    @Test fun rejectsTamperedHashEvenWhenSizeMatches() {
        val (model, record) = fixture()
        record.addProperty("model_sha256", "a".repeat(64))
        File(model.path + ".fp16.json").writeText(record.toString())
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model) }
    }
}
