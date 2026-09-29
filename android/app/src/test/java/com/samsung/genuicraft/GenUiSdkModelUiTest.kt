package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GenUiSdkModelUiTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun cancelledReplacementWaitsForOldProviderCleanupAndDoesNotContinue() = runBlocking {
        val closeStarted = CompletableDeferred<Unit>()
        val finishClose = CompletableDeferred<Unit>()
        val continued = AtomicBoolean(false)
        val provider = object : GenUiProvider {
            override val id = "closing"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                error("Generation is not part of this test.")

            override suspend fun closeAndAwait() {
                closeStarted.complete(Unit)
                finishClose.await()
            }
        }
        val replacement = launch {
            closeProviderBeforeReplacement(provider)
            continued.set(true)
        }

        closeStarted.await()
        replacement.cancel()
        yield()
        assertFalse("Cancelled replacement finished before old-provider cleanup", replacement.isCompleted)

        finishClose.complete(Unit)
        replacement.cancelAndJoin()
        assertTrue(replacement.isCancelled)
        assertFalse("Cancelled replacement proceeded after cleanup", continued.get())
    }

    @Test
    fun modelChoiceRestoresKnownValueAndDefaultsUnknownValueToOfficial() {
        assertEquals(
            E2bModelChoice.TRAINED_E2B_V10_W4,
            E2bModelChoice.fromPreference("trained_e2b_v10_w4"),
        )
        assertEquals(E2bModelChoice.OFFICIAL_E2B, E2bModelChoice.fromPreference(null))
        assertEquals(E2bModelChoice.OFFICIAL_E2B, E2bModelChoice.fromPreference("future"))
        assertEquals(
            InferenceBackendSettings.TrainedE2bGpuPrecision.FP32,
            InferenceBackendSettings.TrainedE2bGpuPrecision.fromRawValue(null),
        )
        assertEquals(
            InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            InferenceBackendSettings.TrainedE2bGpuPrecision.fromRawValue("FP16_CORRECTED"),
        )
        assertEquals(
            InferenceBackendSettings.TrainedE2bGpuPrecision.FP32,
            InferenceBackendSettings.TrainedE2bGpuPrecision.fromRawValue("model_default"),
        )
    }

    @Test
    fun trainedAndOfficialPathsAndProviderProfilesRemainIsolated() {
        assertEquals("model_path", PREFERENCE_OFFICIAL_E2B_MODEL_PATH)
        assertEquals(
            "trained_e2b_v10_w4_model_path",
            PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
        )
        assertNotEquals(
            PREFERENCE_OFFICIAL_E2B_MODEL_PATH,
            PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
        )
        assertNotEquals(PREFERENCE_E2B_MODEL_CHOICE, PREFERENCE_OFFICIAL_E2B_MODEL_PATH)
        assertNotEquals(PREFERENCE_E2B_MODEL_CHOICE, PREFERENCE_TRAINED_E2B_W4_MODEL_PATH)

        val officialKey = sdkDemoProviderKey(
            e2bModelChoice = E2bModelChoice.OFFICIAL_E2B,
            modelPath = "/models/e2b.litertlm",
            enableMetrics = true,
        )
        val trainedKey = sdkDemoProviderKey(
            e2bModelChoice = E2bModelChoice.TRAINED_E2B_V10_W4,
            modelPath = "/models/e2b.litertlm",
            enableMetrics = true,
        )

        assertNotEquals(officialKey, trainedKey)
        assertTrue(officialKey.contains("profile=official_e2b"))
        assertTrue(trainedKey.contains("profile=trained_e2b_v10_w4"))
        assertTrue(officialKey.contains("path=/models/e2b.litertlm"))
        assertTrue(trainedKey.contains("path=/models/e2b.litertlm"))
    }

    @Test
    fun trainedModelMustBeAReadableNonEmptyLiteRtLmFile() {
        assertFalse(trainedE2bW4Readiness("").usable)
        assertFalse(trainedE2bW4Readiness("relative.litertlm").usable)

        val missing = temporaryFolder.root.resolve("missing.litertlm")
        assertFalse(trainedE2bW4Readiness(missing.absolutePath).usable)

        val wrongFormat = temporaryFolder.newFile("model.bin").apply {
            writeBytes(byteArrayOf(1))
        }
        assertFalse(trainedE2bW4Readiness(wrongFormat.absolutePath).usable)

        val empty = temporaryFolder.newFile("empty.litertlm")
        assertFalse(trainedE2bW4Readiness(empty.absolutePath).usable)

        val ready = temporaryFolder.newFile("e2b_v10_w4.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3))
        }
        val readiness = trainedE2bW4Readiness(ready.absolutePath)
        assertTrue(readiness.usable)
        assertTrue(readiness.message.contains("GPU"))
    }

    @Test
    fun correctedFp16RequiresPreparedSiblingAndMatchingManifest() {
        val original = temporaryFolder.newFile("gemma4_e2b_a2ui_mobile.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3))
        }
        val fp32 = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP32,
        )
        assertEquals(original.absolutePath, fp32.modelPath)
        assertEquals(Gemma4GpuPrecision.FP32, fp32.gpuPrecision)
        assertTrue(fp32.readiness.usable)

        val corrected = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
        )
        assertFalse(corrected.readiness.usable)
        assertEquals(CORRECTED_FP16_MODEL_FILE_NAME, java.io.File(corrected.modelPath).name)

        val correctedFile = temporaryFolder.newFile(CORRECTED_FP16_MODEL_FILE_NAME).apply {
            writeBytes(byteArrayOf(4, 5, 6, 7))
        }
        assertFalse(selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
        ).readiness.usable)

        val manifest = java.io.File("${correctedFile.absolutePath}.fp16.json")
        manifest.writeText(correctedManifest(modelBytes = correctedFile.length(), drafterCorrected = false))
        val ready = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
        )
        assertEquals(correctedFile.absolutePath, ready.modelPath)
        assertEquals(Gemma4GpuPrecision.FP16_CORRECTED, ready.gpuPrecision)
        assertTrue(ready.readiness.message, ready.readiness.usable)
        assertFalse(ready.mtpSupported)
        assertTrue(ready.readiness.message.contains("MTP unavailable"))

        manifest.writeText(correctedManifest(modelBytes = correctedFile.length() + 1L))
        assertFalse(selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
        ).readiness.usable)

        manifest.writeText(correctedManifest(modelBytes = correctedFile.length(), drafterCorrected = true))
        assertTrue(selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
        ).mtpSupported)
    }

    @Test
    fun convertReadinessFollowsOnlyTheSelectedModel() {
        assertFalse(
            sdkDemoActionAvailability(
                working = false,
                e2bModelChoice = E2bModelChoice.TRAINED_E2B_V10_W4,
                officialModelSource = GemmaModelSource.MANAGED_DOWNLOAD,
                managedOfficialModelReady = true,
                trainedModelReady = false,
            ).convertEnabled
        )
        assertTrue(
            sdkDemoActionAvailability(
                working = false,
                e2bModelChoice = E2bModelChoice.TRAINED_E2B_V10_W4,
                officialModelSource = GemmaModelSource.MANAGED_DOWNLOAD,
                managedOfficialModelReady = false,
                trainedModelReady = true,
            ).convertEnabled
        )
        assertTrue(
            sdkDemoActionAvailability(
                working = false,
                e2bModelChoice = E2bModelChoice.OFFICIAL_E2B,
                officialModelSource = GemmaModelSource.MANAGED_DOWNLOAD,
                managedOfficialModelReady = true,
                trainedModelReady = false,
            ).convertEnabled
        )
        assertTrue(
            sdkDemoActionAvailability(
                working = false,
                e2bModelChoice = E2bModelChoice.OFFICIAL_E2B,
                officialModelSource = GemmaModelSource.LOCAL_FILE,
                managedOfficialModelReady = false,
                trainedModelReady = false,
            ).convertEnabled
        )
        assertFalse(
            sdkDemoActionAvailability(
                working = true,
                e2bModelChoice = E2bModelChoice.OFFICIAL_E2B,
                officialModelSource = GemmaModelSource.LOCAL_FILE,
                managedOfficialModelReady = true,
                trainedModelReady = true,
            ).renderEnabled
        )
    }

    @Test
    fun mtpSettingReplacesEachOnDeviceProvider() {
        E2bModelChoice.entries.forEach { choice ->
            val on = sdkDemoProviderKey(choice, "/models/e2b.litertlm", true, true)
            val off = sdkDemoProviderKey(choice, "/models/e2b.litertlm", true, false)
            assertNotEquals("MTP changes must recreate the engine for $choice", on, off)
        }
        val enabled = trainedE2bW4Config("/models/e2b.litertlm", true, enableMtp = true)
        val disabled = trainedE2bW4Config("/models/e2b.litertlm", true, enableMtp = false)
        assertTrue(enabled.enableSpeculativeDecoding)
        assertFalse(disabled.enableSpeculativeDecoding)
        assertEquals(disabled, enabled.copy(enableSpeculativeDecoding = false))
        assertFalse("MTP toggle must not change the trained prompt's thinking mode", enabled.enableThinking)
        assertNotEquals(
            sdkDemoProviderKey(E2bModelChoice.TRAINED_E2B_V10_W4, "/models/e2b.litertlm", true, false),
            sdkDemoProviderKey(
                E2bModelChoice.TRAINED_E2B_V10_W4,
                "/models/e2b.litertlm",
                true,
                false,
                Gemma4GpuPrecision.FP16_CORRECTED,
            ),
        )
    }

    @Test
    fun trainedProfileMatchesExportRuntimeContract() {
        val config = trainedE2bW4Config(
            modelPath = temporaryFolder.newFile("profile.litertlm").absolutePath,
            enableMetrics = true,
        )

        assertEquals("GPU", config.accelerator)
        assertEquals(com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision.FP32, config.gpuPrecision)
        assertEquals(8_192, config.maxContextTokens)
        assertEquals(2_048, config.maxOutputTokens)
        assertFalse(config.enableThinking)
        assertFalse(config.enableSpeculativeDecoding)
        assertTrue(config.enableMetrics)
    }

    private fun correctedManifest(modelBytes: Long, drafterCorrected: Boolean = false): String = """
        {
          "schema_version": 1,
          "precision_policy": "genuicraft-fp16-rope-qdq-v1",
          "max_context_tokens": 8192,
          "target_rope_corrected": true,
          "drafter_rope_corrected": $drafterCorrected,
          "source_sha256": "${"a".repeat(64)}",
          "model_sha256": "${"b".repeat(64)}",
          "model_bytes": $modelBytes
        }
    """.trimIndent()
}
