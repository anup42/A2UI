package com.samsung.genuicraft

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import android.util.Log
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.OnDeviceLitertBackend
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Gemma4E2bA2uiExpressV6GpuOnDeviceTest {
    @Test
    fun v6A2uiExpressRunsOnLiteRtGpu() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_a2ui_express_v6_litert" }
        )
        val modelFile = entry.localFile(context)
        Log.i("V6GpuTest", "model=${modelFile.absolutePath} exists=${modelFile.exists()} isFile=${modelFile.isFile} length=${modelFile.length()}")
        assertTrue("v6 LiteRT model is missing at ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue("v6 model must require GPU execution", entry.requireGpu)

        val expressContract = context.assets.open("pipeline_prompts/genui_gen_a2ui_express_v1.md")
            .bufferedReader()
            .use { it.readText() }
        val trainingAlignedPrompt = expressContract.replace(
            "{response_text}",
            "Monthly Budget\nIncome: $5000\nExpenses: $2100",
        )

        val response = OnDeviceLitertBackend(modelFile.absolutePath).generate(
            InferenceBackend.GenerateRequest(
                systemPrompt = null,
                prompt = trainingAlignedPrompt,
                temperature = 0.0,
                maxOutputTokens = entry.maxOutputTokens,
                jsonMode = false,
                structuredOutput = false,
            )
        )

        assertEquals("GPU", response.runtimeBackend)
        assertNull("GPU generation failed: ${response.error}", response.error)
        assertTrue("GPU response was empty: ${response.error}", response.text.isNotBlank())
        assertTrue("Response was not A2UI Express: ${response.text.take(240)}", response.text.contains("<a2ui>"))
    }
}
