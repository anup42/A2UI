package com.samsung.genuicraft.sdk.internal.renderer.flat.capability

/** Generated from dataset/schema/renderer_capabilities.json. Do not edit. */
internal object GeneratedRendererCapabilities {
    const val VERSION: String = "2.0.0"
    val canonicalTypes: Set<String> = setOf("Stack", "List", "Card", "Table", "Formula", "Chart", "CodeBlock", "ConsoleLog", "Text", "EmailPreview", "Image", "Icon", "Button", "Divider", "Tabs", "Modal", "TextField", "CheckBox", "ChoicePicker", "Slider", "DateTimeInput", "Video", "AudioPlayer", "Alert", "Checklist")
    val runtimeTypes: Set<String> = setOf("stack", "list", "card", "table", "formula", "chart", "codeblock", "consolelog", "text", "emailpreview", "image", "icon", "button", "divider", "tabs", "modal", "textfield", "checkbox", "choicepicker", "slider", "datetimeinput", "video", "audioplayer", "alert", "checklist")
    val typeAliases: Map<String, String> = mapOf("stack" to "stack", "list" to "list", "card" to "card", "table" to "table", "formula" to "formula", "chart" to "chart", "barchart" to "chart", "bar_chart" to "chart", "codeblock" to "codeblock", "code" to "codeblock", "code_block" to "codeblock", "pre" to "codeblock", "preformatted" to "codeblock", "consolelog" to "consolelog", "console" to "consolelog", "console_log" to "consolelog", "terminal" to "consolelog", "logoutput" to "consolelog", "log_output" to "consolelog", "text" to "text", "emailpreview" to "emailpreview", "email_preview" to "emailpreview", "image" to "image", "icon" to "icon", "button" to "button", "divider" to "divider", "tabs" to "tabs", "modal" to "modal", "textfield" to "textfield", "checkbox" to "checkbox", "choicepicker" to "choicepicker", "slider" to "slider", "datetimeinput" to "datetimeinput", "video" to "video", "audioplayer" to "audioplayer", "alert" to "alert", "notice" to "alert", "messagecard" to "alert", "message_card" to "alert", "checklist" to "checklist", "check_list" to "checklist")
    val compatibilityTypeDirections: Map<String, String> = mapOf("row" to "horizontal", "column" to "vertical")
    val consumedProps: Map<String, Set<String>> = mapOf(
        "stack" to setOf("direction", "gap", "spacing", "space", "align", "justify", "wrap", "padding", "paddingHorizontal", "paddingVertical", "margin", "marginHorizontal", "marginVertical"),
        "list" to setOf("items"),
        "card" to setOf("title", "subtitle", "tone", "padding", "paddingHorizontal", "paddingVertical", "margin", "marginHorizontal", "marginVertical"),
        "table" to setOf("title", "domain", "preferredPresentation", "presentation", "columns", "rows", "statePath", "rowsPath", "dataPath", "primaryColumn", "highlightColumns", "numericColumns", "entityMedia"),
        "formula" to setOf("latex", "text", "title", "subtitle", "result", "display"),
        "chart" to setOf("chartType", "title", "subtitle", "yLabel", "columns", "rows", "statePath", "rowsPath", "dataPath", "xKey", "yKey"),
        "codeblock" to setOf("code", "language", "title"),
        "consolelog" to setOf("code", "language", "title"),
        "text" to setOf("text", "variant", "heading", "accessibilityLabel", "contentDescription", "decorative"),
        "emailpreview" to setOf("from", "to", "cc", "bcc", "subject", "body", "date", "timestamp", "attachments", "title"),
        "image" to setOf("url", "src", "source", "name", "fit", "contentScale", "alt", "fallbackUrl", "height", "width", "accessibilityLabel", "contentDescription", "decorative"),
        "icon" to setOf("name", "icon", "source", "url", "src", "size", "iconSize", "tint", "accessibilityLabel", "contentDescription", "decorative"),
        "button" to setOf("label", "text", "variant", "icon", "accessibilityLabel", "contentDescription", "onClickLabel", "actionLabel"),
        "divider" to setOf(),
        "tabs" to setOf("tabs", "activeTabId"),
        "modal" to setOf("trigger", "content", "title"),
        "textfield" to setOf("label", "value", "statePath", "placeholder", "accessibilityLabel", "contentDescription"),
        "checkbox" to setOf("label", "value", "statePath", "accessibilityLabel", "contentDescription"),
        "choicepicker" to setOf("label", "value", "statePath", "options", "accessibilityLabel", "contentDescription"),
        "slider" to setOf("label", "value", "statePath", "min", "max", "step", "accessibilityLabel", "contentDescription"),
        "datetimeinput" to setOf("label", "value", "statePath", "mode", "placeholder", "accessibilityLabel", "contentDescription"),
        "video" to setOf("url", "src", "source", "name", "poster", "posterUrl", "thumbnail", "thumbnailUrl", "description", "title"),
        "audioplayer" to setOf("url", "src", "source", "name", "poster", "posterUrl", "thumbnail", "thumbnailUrl", "description", "title"),
        "alert" to setOf("message", "text", "title", "tone", "timestamp", "source", "icon"),
        "checklist" to setOf("title", "items", "disclaimer", "source")
    )
    val tableDomains: Set<String> = setOf("weather", "flight", "booking", "restaurants", "schedule", "status", "playlist", "news", "product", "generic", "comparison", "formula")
    val cardFirstTableDomains: Set<String> = setOf("weather", "flight", "booking", "restaurants", "schedule", "status", "playlist", "news", "product")
    val tableDomainAliases: Map<String, String> = mapOf("restaurant" to "restaurants", "dining" to "restaurants", "music" to "playlist", "entertainment" to "playlist", "headline" to "news", "headlines" to "news", "article" to "news", "articles" to "news", "calculation" to "formula", "calculator" to "formula", "math" to "formula", "travel" to "schedule", "trip" to "schedule", "vacation" to "schedule", "holiday" to "schedule", "itinerary" to "schedule", "places" to "schedule", "attractions" to "schedule", "shopping" to "product", "products" to "product", "catalog" to "product")
    val actionNames: Set<String> = setOf("openUrl", "setState", "pushState", "removeState", "validateForm", "emitEvent")
    val actionRuntimeKeys: Set<String> = setOf("openurl", "setstate", "pushstate", "removestate", "validateform", "emitevent")
    val actionRequired: Map<String, Set<String>> = mapOf(
        "openurl" to setOf(),
        "setstate" to setOf("value"),
        "pushstate" to setOf("value"),
        "removestate" to setOf("index"),
        "validateform" to setOf(),
        "emitevent" to setOf("name")
    )
    val actionRequiredAny: Map<String, List<Set<String>>> = mapOf(
        "openurl" to listOf(setOf("url", "href", "link", "targetUrl")),
        "setstate" to listOf(setOf("statePath", "path")),
        "pushstate" to listOf(setOf("statePath", "path")),
        "removestate" to listOf(setOf("statePath", "path")),
        "validateform" to listOf(),
        "emitevent" to listOf()
    )
    val safeUrlActions: Set<String> = setOf("openurl")
    val chartSubtypes: Set<String> = setOf("bar", "column")
    val chartSubtypeAliases: Map<String, String> = mapOf("bar_chart" to "bar")
}
