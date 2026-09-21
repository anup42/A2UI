package com.samsung.genuicraft.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OnDeviceModelCatalogTest {
    @Test
    fun settingsExposeOnlyTheThreeRetainedModelsInPriorityOrder() {
        assertEquals(
            listOf(
                "gemma4_e2b_a2ui_mobile",
                "gemma4_e2b_it_litert",
                "gemma3_270m_a2ui_express_int8",
            ),
            OnDeviceModelCatalog.visibleEntries.map { it.id },
        )
        assertEquals(
            listOf(
                "Trained Gemma 4 E2B (A2UI Mobile)",
                "Official Gemma 4 E2B",
                "Gemma 3 270M A2UI Express",
            ),
            OnDeviceModelCatalog.visibleEntries.map { it.displayName },
        )
    }

    @Test
    fun trainedMobileProfileUsesSdkConverterAndRecognizesBothInstalledNames() {
        val canonical = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath(
                "/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/" +
                    "gemma4_e2b_a2ui_mobile.litertlm",
            ),
        )
        val v10Alias = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath(
                "/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/e2b_v10_w4.litertlm",
            ),
        )

        assertEquals("gemma4_e2b_a2ui_mobile", canonical.id)
        assertEquals(canonical, v10Alias)
        assertEquals("gemma4_e2b_a2ui_mobile.litertlm", canonical.fileName)
        assertEquals(8_192, canonical.maxContextTokens)
        assertEquals(2_048, canonical.maxOutputTokens)
        assertTrue(canonical.requireGpu)
        assertTrue(canonical.enableSpeculativeDecoding)
        assertTrue(canonical.trainingCompatiblePrompt)
        assertTrue(canonical.usesTrainedSdkConverter)
        assertFalse(canonical.isDownloadable)
        assertEquals(
            listOf(
                "sdk_models/gemma4_e2b_a2ui_mobile.litertlm",
                "sdk_models/e2b_v10_w4.litertlm",
            ),
            canonical.additionalLocalRelativePaths,
        )
    }

    @Test
    fun officialEntryReusesTheSdkModelsPackageAndLegacyProfilesRemainResolvable() {
        val official = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath(
                "/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/" +
                    "gemma-4-E2B-it.litertlm",
            ),
        )
        assertEquals("gemma4_e2b_it_litert", official.id)
        assertEquals(
            listOf("sdk_models/gemma-4-E2B-it.litertlm"),
            official.additionalLocalRelativePaths,
        )

        val hiddenLegacy = requireNotNull(
            OnDeviceModelCatalog.entryForModelPath("gemma4_e2b_a2ui_express_v6_litert"),
        )
        assertEquals("gemma4_e2b_a2ui_express_v6_litert", hiddenLegacy.id)
        assertFalse(OnDeviceModelCatalog.visibleEntries.any { it.id == hiddenLegacy.id })
    }

    @Test
    fun legacySelectionsPreferTheirInstalledRetainedSuccessor() {
        val available = setOf(
            "gemma4_e2b_a2ui_mobile",
            "gemma4_e2b_it_litert",
            "gemma3_270m_a2ui_express_int8",
        )

        assertEquals(
            "gemma4_e2b_a2ui_mobile",
            OnDeviceModelCatalog.migrationTargetForSelection(
                "/models/gemma-4-e2b-trained-express-int4.litertlm",
                available,
            )?.id,
        )
        assertEquals(
            "gemma4_e2b_it_litert",
            OnDeviceModelCatalog.migrationTargetForSelection(
                "gemma4_e4b_it_litert",
                available,
            )?.id,
        )
        assertEquals(
            "gemma3_270m_a2ui_express_int8",
            OnDeviceModelCatalog.migrationTargetForSelection(
                "/models/gemma-3-270m-ir-int8.litertlm",
                available,
            )?.id,
        )
        assertNull(
            OnDeviceModelCatalog.migrationTargetForSelection(
                "gemma4_e2b_a2ui_express_v6_litert",
                emptySet(),
            ),
        )
        assertNull(
            OnDeviceModelCatalog.migrationTargetForSelection(
                "gemma4_e2b_a2ui_mobile",
                available,
            ),
        )
    }
}
