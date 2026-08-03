package com.samsung.genuicraft.inference

import com.samsung.genuicraft.InferenceBackendSettings
import org.junit.Assert.assertEquals
import org.junit.Test

class OnDeviceLitertBackendPolicyTest {

    @Test
    fun `default backend order prefers GPU before CPU fallback`() {
        assertEquals(
            listOf("GPU", "CPU"),
            liteRtBackendOrder(forceCpu = false, requireGpu = false),
        )
    }

    @Test
    fun `GPU required model never falls back to CPU`() {
        assertEquals(
            listOf("GPU"),
            liteRtBackendOrder(forceCpu = false, requireGpu = true),
        )
    }

    @Test
    fun `explicit accelerator selections are never relabeled as automatic fallback`() {
        assertEquals(
            listOf("GPU"),
            liteRtBackendOrder(
                forceCpu = false,
                requireGpu = false,
                accelerator = InferenceBackendSettings.Accelerator.GPU,
            ),
        )
        assertEquals(
            listOf("NPU"),
            liteRtBackendOrder(
                forceCpu = false,
                requireGpu = true,
                accelerator = InferenceBackendSettings.Accelerator.NPU,
            ),
        )
    }

    @Test
    fun `runtime label makes MTP observable`() {
        assertEquals("GPU+MTP", liteRtRuntimeBackendLabel("GPU", speculativeDecodingEnabled = true))
        assertEquals("CPU", liteRtRuntimeBackendLabel("CPU", speculativeDecodingEnabled = false))
    }

    @Test
    fun `stream text accepts incremental and cumulative callbacks`() {
        val incremental = mergeLiteRtStreamText("{\"root\"", ": \"screen\"}")
        val cumulative = mergeLiteRtStreamText("{\"root\"", "{\"root\": \"screen\"}")
        val repeatedSuffix = mergeLiteRtStreamText("{\"root\": \"screen\"", "}")

        assertEquals("{\"root\": \"screen\"}", incremental)
        assertEquals("{\"root\": \"screen\"}", cumulative)
        assertEquals("{\"root\": \"screen\"}", repeatedSuffix)
    }
}
