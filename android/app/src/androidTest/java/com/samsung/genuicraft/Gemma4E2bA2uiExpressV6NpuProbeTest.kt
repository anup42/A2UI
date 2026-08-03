package com.samsung.genuicraft

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.OnDeviceLitertBackend
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Attempts the explicit LiteRT-LM NPU path and records why it is unavailable when the
 * device does not provide the required vendor runtime or a compatible AOT model.
 */
@RunWith(AndroidJUnit4::class)
class Gemma4E2bA2uiExpressV6NpuProbeTest {
    @Test
    fun probeNpuBackendWithoutClaimingGpuFallback() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_a2ui_express_v6_litert" }
        )
        val modelFile = entry.localFile(context)
        assumeTrue(
            "v6 model is not installed at ${modelFile.absolutePath}",
            modelFile.isFile && modelFile.length() >= entry.minimumFileSizeBytes,
        )

        val resultDir = File(context.filesDir, "result_gemma4_e2b_a2ui_express_v6").apply { mkdirs() }
        val response = OnDeviceLitertBackend(
            modelPath = modelFile.absolutePath,
            acceleratorPreference = InferenceBackendSettings.Accelerator.NPU,
            npuNativeLibraryDir = context.applicationInfo.nativeLibraryDir,
        ).generate(
            InferenceBackend.GenerateRequest(
                prompt = "Return a small A2UI Express status surface for package A-1042.",
                systemPrompt = null,
                temperature = 0.0,
                maxOutputTokens = entry.maxOutputTokens,
                jsonMode = false,
                structuredOutput = true,
            )
        )
        File(resultDir, "npu_probe.txt").writeText(
            "runtimeBackend=${response.runtimeBackend}\n" +
                "error=${response.error.orEmpty()}\n" +
                "text=${response.text}\n"
        )

        if (!response.error.isNullOrBlank()) {
            assumeTrue(
                "NPU unavailable for this model/device: ${response.error}",
                false,
            )
        }
        assertTrue("NPU returned an empty response", response.text.isNotBlank())
        assertTrue("NPU response was not labeled NPU", response.runtimeBackend.orEmpty().startsWith("NPU"))
    }
}
