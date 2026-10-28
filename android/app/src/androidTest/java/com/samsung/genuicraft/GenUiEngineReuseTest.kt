package com.samsung.genuicraft

import android.os.Build
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.*
import com.samsung.genuicraft.sdk.provider.*
import java.io.File
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Real GPU lifecycle probe. Uses existing device models and leaves user settings untouched. */
@RunWith(AndroidJUnit4::class)
class GenUiEngineReuseTest {
    @Test fun retainedEnginePrewarmMetricsAndIdleRelease() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val args = InstrumentationRegistry.getArguments()
        val model = args.getString("modelPath") ?: File(context.getExternalFilesDir(null),
            "sdk_models/qat_grpo_20261009/model-fp16-corrected.litertlm").absolutePath
        val out = File(context.getExternalFilesDir(null), "result/engine_reuse/${System.currentTimeMillis()}")
        check(out.mkdirs())
        val rows = context.assets.open("genuicraft_bixby50.jsonl").bufferedReader().useLines {
            it.map { line -> JsonParser.parseString(line).asJsonObject }.associateBy { row -> row["id"].asString }
        }
        val gson = GsonBuilder().setPrettyPrinting().serializeNulls().create()
        val records = mutableListOf<Map<String, Any?>>()
        val config = GenUiModelProfiles.trainedE2b(model, enableMtp = true, enableMetrics = true,
            gpuPrecision = Gemma4GpuPrecision.FP16_CORRECTED)
        GenUiEngineManager.install(context)
        GenUiEngineManager.retire("probe_start")

        fun session(stopComplete: Boolean = true): GenUiSession {
            val native = Gemma4Provider(config)
            val provider = if (stopComplete) native else object : GenUiProvider by native {
                override suspend fun generate(prompt: GenUiPrompt) = native.generate(prompt.copy(stopAtCompleteExpress = false))
                override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit) =
                    native.generate(prompt.copy(stopAtCompleteExpress = false), onPartialText)
            }
            return GenUiSession(context, provider, GenUiConversionProfile.TRAINED_E2B_V10_W4)
        }

        suspend fun convert(label: String, caseId: String, stopComplete: Boolean = true): GenUiModelOutput {
            val sdk = session(stopComplete)
            val row = requireNotNull(rows[caseId])
            var chunks = 0
            val started = SystemClock.elapsedRealtime()
            try {
                val result = withTimeout(180_000) {
                    sdk.convert(GenUiRequest(row["text"].asString, row["query"].asString),
                        GenUiGenerationObserver(onPartialText = { _, _ -> chunks++ }),
                        enableStreamingRendering = false)
                }
                val wall = SystemClock.elapsedRealtime() - started
                val output = requireNotNull(sdk.attemptSnapshots.firstOrNull()?.output)
                File(out, "$label.express").writeText(output.text)
                val record = linkedMapOf<String, Any?>("label" to label, "caseId" to caseId,
                    "wallMs" to wall, "chunks" to chunks, "output" to output,
                    "sessionMetrics" to sdk.generationSessionMetrics, "conversion" to result)
                records += record
                File(out, "requests.json").writeText(gson.toJson(records))
                assertTrue("Native output missing", output.text.isNotBlank())
                assertTrue("No token metrics", (output.metrics?.outputTokens ?: 0) > 0)
                assertTrue("No runtime resource samples", (output.metrics?.resources?.sampleCount ?: 0) > 0)
                assertTrue("Unexpected runtime ${output.runtime}", output.runtime.contains("GPU+FP16_CORRECTED+MTP"))
                assertTrue("MTP must remain pending", sdk.generationSessionMetrics?.pendingRetirement == true)
                assertTrue("No streaming callbacks", chunks > 1)
                return output
            } finally { sdk.closeAndAwait() }
        }

        try {
            val cold = convert("cold", "BXP-003")
            val warm = convert("warm_new_provider", "BXP-003")
            assertEquals(true, cold.metrics?.engineInitializedForRequest)
            assertEquals(false, warm.metrics?.engineInitializedForRequest)
            assertEquals(cold.metrics?.engineId, warm.metrics?.engineId)
            convert("warm_weather", "BXP-001")
            convert("warm_weather_without_early_stop", "BXP-001", stopComplete = false)
            GenUiEngineManager.retire("probe_explicit_release")
            val retired = requireNotNull(GenUiEngineManager.retirementMetrics.value)
            File(out, "retired.json").writeText(gson.toJson(retired))
            assertEquals(cold.metrics?.engineId, retired.engineId)
            assertFalse(retired.pendingRetirement)
            assertEquals(4, retired.requestCount)
            assertEquals("engine_lifetime", retired.acceptanceScope)

            val prewarmStart = SystemClock.elapsedRealtime()
            val loading = async {
                val sdk = session()
                try { sdk.prewarm() } finally { sdk.closeAndAwait() }
            }
            // Simulates upstream IO and proves overlap without sending private text to a server.
            delay(1_200)
            loading.await()
            val prewarmMs = SystemClock.elapsedRealtime() - prewarmStart
            val prewarmed = convert("prewarmed", "BXP-003")
            assertEquals(false, prewarmed.metrics?.engineInitializedForRequest)
            assertNotEquals(cold.metrics?.engineId, prewarmed.metrics?.engineId)
            if (args.getString("verifyIdle", "true") == "true") {
                withTimeout(GenUiEngineManager.IDLE_RELEASE_MILLIS + 25_000) {
                    while (GenUiEngineManager.retirementMetrics.value?.engineId != prewarmed.metrics?.engineId) delay(500)
                }
                val idle = GenUiEngineManager.retirementMetrics.value
                File(out, "idle_retired.json").writeText(gson.toJson(idle))
                assertEquals("idle_timeout", idle?.retirementReason)
                val reloaded = convert("after_idle", "BXP-001")
                assertEquals(true, reloaded.metrics?.engineInitializedForRequest)
                assertNotEquals(prewarmed.metrics?.engineId, reloaded.metrics?.engineId)
            }
            File(out, "summary.json").writeText(gson.toJson(mapOf("passed" to true,
                "device" to Build.MODEL, "modelPath" to model, "prewarmAndUpstreamMs" to prewarmMs,
                "simulatedUpstreamMs" to 1200, "requestCount" to records.size)))
        } catch (failure: Throwable) {
            File(out, "failure.txt").writeText(failure.stackTraceToString())
            throw failure
        } finally { GenUiEngineManager.retire("probe_cleanup") }
    }
}
