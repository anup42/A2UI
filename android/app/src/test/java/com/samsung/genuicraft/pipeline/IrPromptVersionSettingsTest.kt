package com.samsung.genuicraft.pipeline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class IrPromptVersionSettingsTest {

    @Test
    fun optionsExposeBothSupportedGenerationFormats() {
        val options = IrPromptVersionSettings.options()

        assertEquals(
            listOf("compact_ir_v2", "a2ui_express_v1"),
            options.map { it.id }
        )
        assertEquals(
            listOf(GenUiIrFormat.COMPACT_IR_V2, GenUiIrFormat.A2UI_EXPRESS_V1),
            options.map { it.outputFormat }
        )
        assertEquals("compact_ir_v2", IrPromptVersionSettings.defaultOption().id)
        assertTrue(options.all { it.title.isNotBlank() && it.description.isNotBlank() })
    }
}
