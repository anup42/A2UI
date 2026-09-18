package com.samsung.genuicraft.sdk.internal.pipeline

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

    /** Explicit renderer/catalog properties; unknown named properties are rejected. */
    private val rendererProperties: Map<String, Set<String>> = mapOf(
        "Stack" to setOf("direction", "gap", "spacing", "space", "align", "justify", "wrap", "padding", "paddingHorizontal", "paddingVertical", "margin", "marginHorizontal", "marginVertical"),
        "List" to setOf("items"),
        "Card" to setOf("title", "subtitle", "tone", "padding", "paddingHorizontal", "paddingVertical", "margin", "marginHorizontal", "marginVertical"),
        "Table" to setOf("title", "domain", "preferredPresentation", "presentation", "columns", "rows", "statePath", "rowsPath", "dataPath", "primaryColumn", "highlightColumns", "numericColumns", "entityMedia"),
        "Formula" to setOf("latex", "text", "title", "subtitle", "result", "display"),
        "Chart" to setOf("chartType", "title", "subtitle", "yLabel", "columns", "rows", "statePath", "rowsPath", "dataPath", "xKey", "yKey"),
        "CodeBlock" to setOf("code", "language", "title"),
        "ConsoleLog" to setOf("code", "language", "title"),
        "Text" to setOf("text", "variant", "heading", "accessibilityLabel", "contentDescription", "decorative"),
        "EmailPreview" to setOf("from", "to", "cc", "bcc", "subject", "body", "date", "timestamp", "attachments", "title"),
        "Image" to setOf("url", "src", "source", "name", "fit", "contentScale", "alt", "fallbackUrl", "height", "width", "aspectRatio", "accessibilityLabel", "contentDescription", "decorative"),
        "Icon" to setOf("name", "icon", "source", "url", "src", "size", "iconSize", "tint", "alt", "accessibilityLabel", "contentDescription", "decorative"),
        "Button" to setOf("label", "text", "variant", "icon", "accessibilityLabel", "contentDescription", "onClickLabel", "actionLabel"),
        "Divider" to emptySet(),
        "Tabs" to setOf("tabs", "activeTabId"),
        "Modal" to setOf("trigger", "content", "title"),
        "TextField" to setOf("label", "value", "statePath", "placeholder", "accessibilityLabel", "contentDescription"),
        "CheckBox" to setOf("label", "value", "statePath", "accessibilityLabel", "contentDescription"),
        "ChoicePicker" to setOf("label", "value", "statePath", "options", "accessibilityLabel", "contentDescription"),
        "Slider" to setOf("label", "value", "statePath", "min", "max", "step", "accessibilityLabel", "contentDescription"),
        "DateTimeInput" to setOf("label", "value", "statePath", "mode", "placeholder", "accessibilityLabel", "contentDescription"),
        "Video" to setOf("url", "src", "source", "name", "poster", "posterUrl", "thumbnail", "thumbnailUrl", "description", "title"),
        "AudioPlayer" to setOf("url", "src", "source", "name", "poster", "posterUrl", "thumbnail", "thumbnailUrl", "description", "title"),
        "Alert" to setOf("message", "text", "title", "tone", "timestamp", "source", "icon"),
        "Checklist" to setOf("title", "items", "disclaimer", "source"),
    )
    private val commonProperties = setOf(
        "width", "height", "flex", "accessibility", "accessibilityLabel",
        "contentDescription", "decorative", "role", "semanticRole", "ariaLabel",
        // Explicit renderer reference properties from the pinned Express profile.
        "child", "template", "itemTemplate",
    )

    fun isAllowedProperty(component: String, property: String): Boolean {
        if (property == "children") return true
        val canonical = when (component) {
            "Row", "Column" -> "Stack"
            else -> component
        }
        return property in (rendererProperties[canonical].orEmpty() + positional[component].orEmpty() + commonProperties)
    }
}
