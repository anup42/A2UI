package com.samsung.genuicraft.pipeline

/** Generated from dataset/schema/genuicraft_a2ui_catalog_v1.json. */
internal object GenUiA2uiCatalog {
    const val CATALOG_ID: String = "https://genui.samsung.com/a2ui/catalogs/genuicraft-mobile/v1"
    val positional: Map<String, List<String>> = mapOf(
        "Alert" to listOf("message", "title", "tone", "timestamp"),
        "AudioPlayer" to listOf("url", "description", "posterUrl", "title"),
        "Button" to listOf("label", "variant", "icon"),
        "Card" to listOf("children", "title", "subtitle", "tone"),
        "Chart" to listOf("chartType", "columns", "statePath", "rows", "title", "subtitle"),
        "CheckBox" to listOf("label", "value", "statePath"),
        "Checklist" to listOf("items", "title", "disclaimer", "source"),
        "ChoicePicker" to listOf("label", "options", "value", "statePath"),
        "Column" to listOf("children", "gap", "align", "justify", "wrap"),
        "CodeBlock" to listOf("code", "language", "title"),
        "ConsoleLog" to listOf("code", "language", "title"),
        "DateTimeInput" to listOf("label", "value", "mode", "placeholder", "statePath"),
        "Divider" to listOf(),
        "EmailPreview" to listOf("subject", "body", "from", "to", "date", "title"),
        "Formula" to listOf("latex", "title", "result", "display"),
        "Icon" to listOf("name", "size", "tint"),
        "Image" to listOf("url", "alt", "fit", "width", "height"),
        "List" to listOf("children", "items"),
        "Modal" to listOf("trigger", "content", "title"),
        "Row" to listOf("children", "gap", "align", "justify", "wrap"),
        "Slider" to listOf("label", "value", "min", "max", "step", "statePath"),
        "Stack" to listOf("children", "direction", "gap", "align", "justify", "wrap"),
        "Table" to listOf("columns", "statePath", "rows", "title", "domain", "preferredPresentation"),
        "Tabs" to listOf("tabs", "activeTabId"),
        "Text" to listOf("text", "variant"),
        "TextField" to listOf("label", "value", "statePath", "placeholder"),
        "Video" to listOf("url", "posterUrl", "description", "title")
    )
}
