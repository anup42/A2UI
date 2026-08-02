package com.samsung.genuicraft.pipeline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class IrPromptVersionSettingsTest {

    @Test
    fun optionsExposeOnlyTheProductionExpressFormat() {
        val options = IrPromptVersionSettings.options()

        assertEquals(
            listOf("a2ui_express_v1"),
            options.map { it.id }
        )
        assertEquals(
            listOf(GenUiIrFormat.A2UI_EXPRESS_V1),
            options.map { it.outputFormat }
        )
        assertEquals("a2ui_express_v1", IrPromptVersionSettings.defaultOption().id)
        assertTrue(options.all { it.title.isNotBlank() && it.description.isNotBlank() })
    }
}
