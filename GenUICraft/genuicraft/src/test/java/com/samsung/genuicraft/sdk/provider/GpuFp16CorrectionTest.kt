package com.samsung.genuicraft.sdk.provider

import com.google.gson.JsonObject
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GpuFp16CorrectionTest {
    @get:Rule val temporary = TemporaryFolder()
    private val fixtureSourceBytes = 12345L
    private val legacyTransform = "pinned-r64-rope-target-mtp-8192-v1"
    private val newTransform = "e2b-rope-target-mtp-8192-v2"

    private fun sha256(value: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(value).joinToString("") { "%02x".format(it) }

    private fun testProfile(sourceSha: String = GpuFp16Correction.SOURCE_SHA256,
                            transform: String = legacyTransform) = GpuFp16Correction.VerifiedModelProfile(
        sourceSha256 = sourceSha,
        sourceBytes = fixtureSourceBytes,
        transformVersion = transform,
        modelSha256 = sha256("immutable trained weights".toByteArray()),
    )

    private fun fixture(drafter: Boolean = true): Pair<File, JsonObject> {
        val model = temporary.newFile("model-${System.nanoTime()}.litertlm").apply { writeText("immutable trained weights") }
        val record = JsonObject().apply {
            addProperty("schema_version", 1)
            addProperty("source_sha256", GpuFp16Correction.SOURCE_SHA256)
            addProperty("source_bytes", fixtureSourceBytes)
            addProperty("transform_version", legacyTransform)
            addProperty("precision_policy", GpuFp16Correction.POLICY)
            addProperty("target_rope_corrected", true)
            addProperty("drafter_rope_corrected", drafter)
            addProperty("max_context_tokens", 8192)
            addProperty("model_bytes", model.length())
            addProperty("model_sha256", sha256(model.readBytes()))
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
        GpuFp16Correction.validateModelAgainstProfiles(
            config(model, mtp, context), listOf(testProfile()),
        )

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

    @Test fun rejectsAnotherSourceOrTransformDespiteMatchingPreparedPayload() {
        val (model, record) = fixture()
        record.addProperty("source_sha256", GpuFp16Correction.NEW_SOURCE_SHA256)
        File(model.path + ".fp16.json").writeText(record.toString())
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model) }

        record.addProperty("source_sha256", GpuFp16Correction.SOURCE_SHA256)
        record.addProperty("transform_version", newTransform)
        File(model.path + ".fp16.json").writeText(record.toString())
        assertThrows(IllegalArgumentException::class.java) { validateFixture(model) }
    }

    @Test fun secondProfileRequiresItsOwnSourceTransformAndPreparedHash() {
        val (model, record) = fixture()
        record.addProperty("source_sha256", GpuFp16Correction.NEW_SOURCE_SHA256)
        record.addProperty("transform_version", newTransform)
        File(model.path + ".fp16.json").writeText(record.toString())
        assertEquals(model.canonicalFile, GpuFp16Correction.validateModelAgainstProfiles(
            config(model), listOf(testProfile(), testProfile(GpuFp16Correction.NEW_SOURCE_SHA256, newTransform)),
        ))

        record.addProperty("model_sha256", "a".repeat(64))
        File(model.path + ".fp16.json").writeText(record.toString())
        assertThrows(IllegalArgumentException::class.java) {
            GpuFp16Correction.validateModelAgainstProfiles(
                config(model), listOf(testProfile(), testProfile(GpuFp16Correction.NEW_SOURCE_SHA256, newTransform)),
            )
        }
    }

    @Test fun verifiedGrpoProfileReachesPayloadHashCheck() {
        val (model, record) = fixture()
        record.addProperty("source_sha256", GpuFp16Correction.NEW_SOURCE_SHA256)
        record.addProperty("source_bytes", 2_588_147_712L)
        record.addProperty("transform_version", newTransform)
        record.addProperty("model_sha256", GpuFp16Correction.NEW_MODEL_SHA256)
        File(model.path + ".fp16.json").writeText(record.toString())

        val failure = assertThrows(IllegalArgumentException::class.java) {
            GpuFp16Correction.validateModel(config(model))
        }
        assertTrue(failure.message.orEmpty().contains("model hash differs from its preparation manifest"))
    }
}
