package com.samsung.genuicraft

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GemmaDownloadUiTest {
    @Test
    fun managedGemmaRequiresVerifiedModelButRendererOnlyRemainsAvailable() {
        val unavailable = demoActionAvailability(
            working = false,
            modelSource = GemmaModelSource.MANAGED_DOWNLOAD,
            managedModelReady = false,
        )
        assertFalse(unavailable.convertEnabled)
        assertTrue(unavailable.renderEnabled)

        val ready = demoActionAvailability(
            working = false,
            modelSource = GemmaModelSource.MANAGED_DOWNLOAD,
            managedModelReady = true,
        )
        assertTrue(ready.convertEnabled)
        assertTrue(ready.renderEnabled)
    }

    @Test
    fun localFileDoesNotDependOnManagedDownloadState() {
        assertTrue(demoActionAvailability(false, GemmaModelSource.LOCAL_FILE, false).convertEnabled)
        assertFalse(demoActionAvailability(false, GemmaModelSource.MANAGED_DOWNLOAD, false).convertEnabled)
    }
}
