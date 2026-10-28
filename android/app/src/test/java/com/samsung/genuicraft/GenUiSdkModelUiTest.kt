package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import java.io.File
import java.security.MessageDigest
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
        assertEquals(
            E2bModelChoice.TRAINED_E2B_PREVIOUS_R64,
            E2bModelChoice.fromPreference("trained_e2b_previous_r64"),
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
        assertEquals(
            "trained_e2b_previous_r64_model_path",
            PREFERENCE_TRAINED_E2B_PREVIOUS_R64_MODEL_PATH,
        )
        assertNotEquals(
            PREFERENCE_OFFICIAL_E2B_MODEL_PATH,
            PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
        )
        assertNotEquals(PREFERENCE_E2B_MODEL_CHOICE, PREFERENCE_OFFICIAL_E2B_MODEL_PATH)
        assertNotEquals(PREFERENCE_E2B_MODEL_CHOICE, PREFERENCE_TRAINED_E2B_W4_MODEL_PATH)
        assertNotEquals(PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
            PREFERENCE_TRAINED_E2B_PREVIOUS_R64_MODEL_PATH)
        assertNotEquals(PREFERENCE_OFFICIAL_E2B_MODEL_PATH,
            PREFERENCE_TRAINED_E2B_PREVIOUS_R64_MODEL_PATH)

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
    fun previousR64UsesTheSameTrainedSdkProfileWithItsOwnPathAndReadiness() {
        val latest = E2bModelChoice.TRAINED_E2B_V10_W4
        val previous = E2bModelChoice.TRAINED_E2B_PREVIOUS_R64
        assertTrue(latest.isTrainedE2b)
        assertTrue(previous.isTrainedE2b)
        assertFalse(E2bModelChoice.OFFICIAL_E2B.isTrainedE2b)
        assertEquals(GenUiTrainedConverter.PROFILE, latest.profile)
        assertEquals(latest.profile, previous.profile)
        assertTrue(previous.supportsMtp)
        assertNotEquals(latest.displayName, previous.displayName)
        assertNotEquals(
            sdkDemoProviderKey(latest, "/models/latest.litertlm", true, true,
                Gemma4GpuPrecision.FP16_CORRECTED),
            sdkDemoProviderKey(previous, "/models/previous/model-fp16-corrected.litertlm", true, true,
                Gemma4GpuPrecision.FP16_CORRECTED),
        )
        assertFalse(sdkDemoActionAvailability(false, previous, GemmaModelSource.MANAGED_DOWNLOAD,
            managedOfficialModelReady = true, trainedModelReady = false).convertEnabled)
        assertTrue(sdkDemoActionAvailability(false, previous, GemmaModelSource.MANAGED_DOWNLOAD,
            managedOfficialModelReady = false, trainedModelReady = true).convertEnabled)
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
        val preparedBytes = byteArrayOf(4, 5, 6, 7)
        val profile = correctedProfile(original, preparedBytes)
        val fp32 = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP32,
        )
        assertEquals(original.absolutePath, fp32.modelPath)
        assertEquals(Gemma4GpuPrecision.FP32, fp32.gpuPrecision)
        assertTrue(fp32.readiness.usable)

        val corrected = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(profile),
        )
        assertFalse(corrected.readiness.usable)
        assertEquals(CORRECTED_FP16_MODEL_FILE_NAME, java.io.File(corrected.modelPath).name)

        val correctedFile = temporaryFolder.newFile(CORRECTED_FP16_MODEL_FILE_NAME).apply {
            writeBytes(preparedBytes)
        }
        assertFalse(selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(profile),
        ).readiness.usable)

        val manifest = File("${correctedFile.absolutePath}.fp16.json")
        manifest.writeText(correctedManifest(profile, correctedFile.length(), drafterCorrected = false))
        val ready = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(profile),
        )
        assertEquals(correctedFile.absolutePath, ready.modelPath)
        assertEquals(Gemma4GpuPrecision.FP16_CORRECTED, ready.gpuPrecision)
        assertTrue(ready.readiness.message, ready.readiness.usable)
        assertFalse(ready.mtpSupported)
        assertTrue(ready.readiness.message.contains("MTP unavailable"))

        manifest.writeText(correctedManifest(profile, correctedFile.length() + 1L))
        assertFalse(selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(profile),
        ).readiness.usable)

        manifest.writeText(correctedManifest(profile, correctedFile.length(), drafterCorrected = true))
        assertTrue(selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(profile),
        ).mtpSupported)
    }

    @Test
    fun correctedFp16RejectsLegacySiblingForNewOriginalOfSameSize() {
        val oldOriginal = temporaryFolder.newFile("gemma4_e2b_a2ui_mobile.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3))
        }
        val newOriginal = temporaryFolder.newFile("e2b_qat_grpo_20261009.litertlm").apply {
            writeBytes(byteArrayOf(8, 9, 0))
        }
        val preparedBytes = byteArrayOf(4, 5, 6, 7)
        val oldProfile = correctedProfile(oldOriginal, preparedBytes)
        val newProfile = correctedProfile(
            newOriginal, byteArrayOf(10, 11, 12), transform = "e2b-rope-target-mtp-8192-v2",
        )
        val prepared = temporaryFolder.newFile(CORRECTED_FP16_MODEL_FILE_NAME).apply { writeBytes(preparedBytes) }
        File("${prepared.absolutePath}.fp16.json").writeText(correctedManifest(oldProfile, prepared.length()))

        assertTrue(selectTrainedE2bModel(
            oldOriginal.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(oldProfile, newProfile),
        ).readiness.usable)
        assertFalse(selectTrainedE2bModel(
            newOriginal.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(oldProfile, newProfile),
        ).readiness.usable)
    }

    @Test
    fun correctedFp16DirectPackageNeedsOnlyVerifiedSidecarAndRejectsFp32() {
        val original = temporaryFolder.newFile("gemma4_e2b_a2ui_previous_r64.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3))
        }
        val preparedBytes = byteArrayOf(4, 5, 6, 7)
        val profile = correctedProfile(original, preparedBytes)
        val unrelatedProfile = correctedProfile(original, byteArrayOf(8, 9), "other-transform")
        val prepared = File(temporaryFolder.newFolder("previous_r64"), CORRECTED_FP16_MODEL_FILE_NAME)
            .apply { writeBytes(preparedBytes) }
        assertTrue(original.delete())
        val sidecar = File("${prepared.absolutePath}.fp16.json")
        fun select(precision: InferenceBackendSettings.TrainedE2bGpuPrecision) = selectTrainedE2bModel(
            prepared.absolutePath, precision, listOf(unrelatedProfile, profile),
        )

        val fp32 = select(InferenceBackendSettings.TrainedE2bGpuPrecision.FP32)
        assertFalse(fp32.readiness.usable)
        assertTrue(fp32.readiness.message.contains("Use FP16 (corrected)"))
        assertFalse(select(InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED).readiness.usable)

        sidecar.writeText(correctedManifest(profile, prepared.length(), drafterCorrected = true))
        val direct = select(InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED)
        assertEquals(prepared.absolutePath, direct.modelPath)
        assertEquals(Gemma4GpuPrecision.FP16_CORRECTED, direct.gpuPrecision)
        assertTrue(direct.readiness.message, direct.readiness.usable)
        assertTrue(direct.mtpSupported)
    }

    @Test
    fun correctedFp16DirectPackageRejectsUnverifiedOrMalformedSidecar() {
        val original = temporaryFolder.newFile("previous-original.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3))
        }
        val preparedBytes = byteArrayOf(4, 5, 6, 7)
        val profile = correctedProfile(original, preparedBytes)
        val prepared = File(temporaryFolder.newFolder("direct-only"), CORRECTED_FP16_MODEL_FILE_NAME)
            .apply { writeBytes(preparedBytes) }
        val sidecar = File("${prepared.absolutePath}.fp16.json")
        val valid = correctedManifest(profile, prepared.length(), drafterCorrected = false)
        fun usable(profiles: List<CorrectedFp16Profile> = listOf(profile)) = selectTrainedE2bModel(
            prepared.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            profiles,
        ).readiness.usable

        sidecar.writeText(valid)
        assertTrue(usable())
        assertFalse(usable(listOf(profile.copy(modelSha256 = "b".repeat(64)))))
        assertFalse(usable(listOf(profile.copy(sourceSha256 = "a".repeat(64)))))
        val invalidManifests = listOf(
            valid.replace("\"source_sha256\": \"${profile.sourceSha256}\"", "\"source_sha256\": \"${"a".repeat(64)}\""),
            valid.replace("\"source_bytes\": ${profile.sourceBytes}", "\"source_bytes\": ${profile.sourceBytes + 1}"),
            valid.replace("\"model_sha256\": \"${profile.modelSha256}\"", "\"model_sha256\": \"${"b".repeat(64)}\""),
            valid.replace("\"model_bytes\": ${prepared.length()}", "\"model_bytes\": ${prepared.length() + 1}"),
            valid.replace("\"transform_version\": \"${profile.transformVersion}\"", "\"transform_version\": \"wrong\""),
            valid.replace("genuicraft-fp16-rope-qdq-v1", "wrong-policy"),
            valid.replace("\"max_context_tokens\": 8192", "\"max_context_tokens\": 4096"),
            valid.replace("\"target_rope_corrected\": true", "\"target_rope_corrected\": false"),
            valid.replace("\"target_rope_corrected\": true", "\"target_rope_corrected\": \"true\""),
            valid.replace("\"drafter_rope_corrected\": false", "\"drafter_rope_corrected\": \"false\""),
            "not-json",
        )
        invalidManifests.forEachIndexed { index, invalid ->
            sidecar.writeText(invalid)
            assertFalse("Invalid direct sidecar #$index must not select a model", usable())
        }
    }

    @Test
    fun correctedFp16RejectsManifestIdentityChangesAndUnverifiedProfile() {
        val original = temporaryFolder.newFile("e2b_qat_grpo_20261009.litertlm").apply {
            writeBytes(byteArrayOf(8, 9, 0))
        }
        val prepared = temporaryFolder.newFile(CORRECTED_FP16_MODEL_FILE_NAME).apply {
            writeBytes(byteArrayOf(10, 11, 12))
        }
        val profile = correctedProfile(original, prepared.readBytes(), "e2b-rope-target-mtp-8192-v2")
        val manifest = File("${prepared.absolutePath}.fp16.json")
        manifest.writeText(correctedManifest(profile, prepared.length()))
        fun usable(candidate: CorrectedFp16Profile = profile) = selectTrainedE2bModel(
            original.absolutePath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(candidate),
        ).readiness.usable

        assertTrue(usable())
        manifest.writeText(correctedManifest(profile.copy(sourceSha256 = "a".repeat(64)), prepared.length()))
        assertFalse(usable())
        manifest.writeText(correctedManifest(profile.copy(modelSha256 = "b".repeat(64)), prepared.length()))
        assertFalse(usable())
        manifest.writeText(correctedManifest(profile.copy(transformVersion = "wrong-transform"), prepared.length()))
        assertFalse(usable())
        manifest.writeText(correctedManifest(profile, prepared.length()))
        assertFalse(usable(profile.copy(modelSha256 = "")))
        assertFalse(selectTrainedE2bModel(
            File(temporaryFolder.root, "missing.litertlm").absolutePath,
            InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
            listOf(profile),
        ).readiness.usable)
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
        E2bModelChoice.entries.filter { it.supportsMtp }.forEach { choice ->
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

    @Test
    fun gemma270ChoiceIsIndependentOfE2bReadinessAndMtpSettings() {
        val choice = E2bModelChoice.TRAINED_GEMMA3_270M
        assertEquals(choice, E2bModelChoice.fromPreference("trained_gemma3_270m"))
        assertFalse(choice.supportsMtp)
        assertFalse(sdkDemoActionAvailability(false, choice, GemmaModelSource.LOCAL_FILE,
            true, true, gemma270ModelReady = false).convertEnabled)
        assertTrue(sdkDemoActionAvailability(false, choice, GemmaModelSource.MANAGED_DOWNLOAD,
            false, false, gemma270ModelReady = true).convertEnabled)
        assertFalse(sdkDemoActionAvailability(true, choice, GemmaModelSource.LOCAL_FILE,
            false, false, gemma270ModelReady = true).convertEnabled)
        assertEquals(sdkDemoProviderKey(choice, "/models/270.litertlm", true, true),
            sdkDemoProviderKey(choice, "/models/270.litertlm", true, false, Gemma4GpuPrecision.FP16_CORRECTED))
        assertNotEquals(sdkDemoProviderKey(choice, "/models/270.litertlm", true),
            sdkDemoProviderKey(choice, "/models/270.litertlm", true, gemma270Accelerator = "GPU"))
        assertNotEquals(sdkDemoProviderKey(choice, "/models/270.litertlm", true),
            sdkDemoProviderKey(choice, "/models/other.litertlm", true))
    }

    @Test
    fun gemma270ReadinessRejectsMissingOrIncompletePackage() {
        assertFalse(gemma270ModelReadiness("/missing/270.litertlm").usable)
        val model = temporaryFolder.newFile("gemma-3-270m-a2ui-express-int8.litertlm")
        assertFalse(gemma270ModelReadiness(model.absolutePath).usable)
        java.io.RandomAccessFile(model, "rw").use { it.setLength(200_000_000L) }
        assertTrue(gemma270ModelReadiness(model.absolutePath).usable)
        val wrongExtension = temporaryFolder.newFile("270.bin")
        java.io.RandomAccessFile(wrongExtension, "rw").use { it.setLength(200_000_000L) }
        assertFalse(gemma270ModelReadiness(wrongExtension.absolutePath).usable)
        val differentModel = temporaryFolder.newFile("e2b_v10_w4.litertlm")
        java.io.RandomAccessFile(differentModel, "rw").use { it.setLength(200_000_000L) }
        assertFalse(gemma270ModelReadiness(differentModel.absolutePath).usable)
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private fun correctedProfile(
        original: File,
        preparedBytes: ByteArray,
        transform: String = "pinned-r64-rope-target-mtp-8192-v1",
    ) = CorrectedFp16Profile(
        originalFileNames = setOf(original.name),
        sourceSha256 = sha256(original.readBytes()),
        sourceBytes = original.length(),
        transformVersion = transform,
        modelSha256 = sha256(preparedBytes),
    )

    private fun correctedManifest(
        profile: CorrectedFp16Profile,
        modelBytes: Long,
        drafterCorrected: Boolean = false,
    ): String = """
        {
          "schema_version": 1,
          "precision_policy": "genuicraft-fp16-rope-qdq-v1",
          "max_context_tokens": 8192,
          "target_rope_corrected": true,
          "drafter_rope_corrected": $drafterCorrected,
          "source_sha256": "${profile.sourceSha256}",
          "source_bytes": ${profile.sourceBytes},
          "transform_version": "${profile.transformVersion}",
          "model_sha256": "${profile.modelSha256}",
          "model_bytes": $modelBytes
        }
    """.trimIndent()
}
