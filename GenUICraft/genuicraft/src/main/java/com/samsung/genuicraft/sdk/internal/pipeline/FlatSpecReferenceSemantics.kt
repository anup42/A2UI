package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonObject

/**
 * Single Android inventory of every element-reference edge consumed by the
 * native FlatSpec renderer. Validation, cycle detection, reachability pruning,
 * and IR ID rewriting must all use this object.
 */
internal object FlatSpecReferenceSemantics {
    const val VERSION: String = "1.0.0"

    data class Reference(
        val targetId: String,
        val sourcePath: String,
        val kind: String,
    )

    fun references(element: JsonObject): List<Reference> {
        val out = linkedMapOf<String, Reference>()
        fun add(value: String?, path: String, kind: String) {
            val target = value?.trim().orEmpty()
            if (target.isNotEmpty()) out.putIfAbsent("$path\u0000$target", Reference(target, path, kind))
        }
        element.get("children")?.takeIf { it.isJsonArray }?.asJsonArray?.forEachIndexed { index, child ->
            add(child.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString, "children[$index]", "child")
        }
        add(element.string("child"), "child", "legacy_child")
        add(element.string("template"), "template", "template")
        add(element.string("itemTemplate"), "itemTemplate", "item_template")

        val props = element.get("props")?.takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()
        add(props.string("template"), "props.template", "template")
        add(props.string("itemTemplate"), "props.itemTemplate", "item_template")
        add(props.string("child"), "props.child", "property_child")

        val topLevelRepeat = element.get("repeat")?.takeIf { it.isJsonObject }?.asJsonObject
        val repeat = topLevelRepeat
            ?: props.get("repeat")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: JsonObject()
        val repeatPath = if (topLevelRepeat != null) "repeat" else "props.repeat"
        add(repeat.string("template"), "$repeatPath.template", "repeat_template")
        add(repeat.string("itemTemplate"), "$repeatPath.itemTemplate", "repeat_item_template")
        add(repeat.string("child"), "$repeatPath.child", "repeat_child")

        when (element.string("type")?.lowercase()) {
            "tabs", "tab", "tabgroup" -> {
                props.get("tabs")?.takeIf { it.isJsonArray }?.asJsonArray?.forEachIndexed { index, raw ->
                    if (!raw.isJsonObject) return@forEachIndexed
                    val tab = raw.asJsonObject
                    listOf("child", "content", "id", "element").firstNotNullOfOrNull { key ->
                        tab.string(key)?.takeIf { it.isNotBlank() }?.let { key to it }
                    }?.let { (key, target) -> add(target, "props.tabs[$index].$key", "tab_content") }
                }
            }
            "modal" -> {
                add(props.string("trigger"), "props.trigger", "modal_trigger")
                add(props.string("content"), "props.content", "modal_content")
            }
        }
        return out.values.toList()
    }

    private fun JsonObject.string(key: String): String? =
        get(key)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
}
