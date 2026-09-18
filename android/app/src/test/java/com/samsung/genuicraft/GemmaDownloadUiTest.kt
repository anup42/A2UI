package com.samsung.genuicraft

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GemmaDownloadUiTest {
    @Test
    fun managedGemmaRequiresVerifiedModelButRendererOnlyRemainsAvailable() {
        val unavailable = demoActionAvailability(
            working = false,
            useGemma = true,
            modelSource = GemmaModelSource.MANAGED_DOWNLOAD,
            managedModelReady = false,
        )
        assertFalse(unavailable.convertEnabled)
        assertTrue(unavailable.renderEnabled)

        val ready = demoActionAvailability(
            working = false,
            useGemma = true,
            modelSource = GemmaModelSource.MANAGED_DOWNLOAD,
            managedModelReady = true,
        )
        assertTrue(ready.convertEnabled)
        assertTrue(ready.renderEnabled)
    }

    @Test
    fun localFileAndGaussDoNotDependOnManagedDownloadState() {
        assertTrue(demoActionAvailability(false, true, GemmaModelSource.LOCAL_FILE, false).convertEnabled)
        assertTrue(demoActionAvailability(false, false, GemmaModelSource.MANAGED_DOWNLOAD, false).convertEnabled)
    }
}
