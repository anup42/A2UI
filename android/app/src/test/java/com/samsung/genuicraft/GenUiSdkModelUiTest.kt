package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
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
    }

    @Test
    fun trainedProfileMatchesExportRuntimeContract() {
        val config = trainedE2bW4Config(
            modelPath = temporaryFolder.newFile("profile.litertlm").absolutePath,
            enableMetrics = true,
        )

        assertEquals("GPU", config.accelerator)
        assertEquals(8_192, config.maxContextTokens)
        assertEquals(2_048, config.maxOutputTokens)
        assertFalse(config.enableThinking)
        assertFalse(config.enableSpeculativeDecoding)
        assertTrue(config.enableMetrics)
    }
}
