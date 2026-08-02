package com.samsung.genuicraft.pipeline

internal enum class GenUiIrFormat(val wireId: String) {
    A2UI_EXPRESS_V1("a2ui_express_v1"),
    A2UI_V1_WIRE("a2ui_v1_wire"),

    /** Historical formats retained only by the offline migration/test boundary. */
    @Deprecated("FlatSpec is legacy import/comparison-only; never select it for inference.")
    FLAT_SPEC_V1("flat_spec_v1"),
    @Deprecated("Compact IR is migration-only; never select it for inference.")
    COMPACT_IR_V2("compact_ir_v2"),
}
