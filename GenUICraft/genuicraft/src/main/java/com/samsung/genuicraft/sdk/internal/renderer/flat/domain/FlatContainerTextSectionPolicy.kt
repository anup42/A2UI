package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import java.util.Locale

/** Groups only reachable plain section containers; original Text nodes remain authoritative. */
internal fun planFlatContainerTextSections(
    elementId: String,
    children: List<String>,
    elements: Map<String, FlatElement>,
    isRoot: Boolean,
    direction: String,
    isRepeated: Boolean,
): List<FlatTextSection>? {
    if (!isRoot || direction != "vertical" || isRepeated || children.size < 2) return null
    val visited = mutableSetOf(elementId)
    val parent = elements[elementId] ?: return null
    if (parent.type.lowercase(Locale.ROOT) !in setOf("stack", "column") || !plainSectionContainer(parent)) return null

    fun reachable(id: String): FlatElement? {
        if (!visited.add(id)) return null
        return elements[id]
    }

    val sections = mutableListOf<FlatTextSection>()
    children.forEach { id ->
        val container = reachable(id) ?: return null
        if (!plainSectionContainer(container)) return null
        val textIds = when (container.type.lowercase(Locale.ROOT)) {
            "stack" -> container.children
            "card" -> {
                val onlyChild = container.children.singleOrNull()
                if (onlyChild != null && elements[onlyChild]?.type.equals("stack", true)) {
                    val stack = reachable(onlyChild) ?: return null
                    if (!plainSectionContainer(stack)) return null
                    stack.children
                } else container.children
            }
            else -> return null
        }
        if (textIds.size < 2) return null
        val nodes = textIds.map { textId ->
            val node = reachable(textId) ?: return null
            if (!plainLiteralSectionText(node)) return null
            node
        }
        if (sectionTextVariant(nodes.first()) !in setOf("h2", "h3") ||
            nodes.drop(1).any { sectionTextVariant(it) in setOf("h1", "h2", "h3", "h4", "h5", "h6") }) return null
        sections += FlatTextSection(textIds.first(), textIds.toList())
    }
    return sections
}

private fun passiveSectionNode(node: FlatElement): Boolean =
    node.repeat == null && node.visible == null && node.on.isNullOrEmpty() && node.watch.isNullOrEmpty()

private fun plainSectionContainer(node: FlatElement): Boolean {
    if (!passiveSectionNode(node) || node.props.keys.any {
            it !in setOf("direction", "gap", "padding", "paddingHorizontal", "paddingVertical")
        }) return false
    val direction = node.props["direction"]
    if (direction != null && (direction !is String || !direction.trim().equals("vertical", true))) return false
    return node.props.values.all { value ->
        value == null || value is Number || (value is String && !value.contains("{{"))
    }
}

private fun plainLiteralSectionText(node: FlatElement): Boolean {
    if (!node.type.equals("text", true) || node.children.isNotEmpty() || !passiveSectionNode(node) ||
        node.props.keys.any { it !in setOf("text", "variant", "typography") }) return false
    val text = node.props["text"] as? String ?: return false
    if (text.isBlank() || text.contains("{{")) return false
    return listOf("variant", "typography").all { key ->
        val value = node.props[key]
        value == null || (value is String && !value.contains("{{"))
    }
}

private fun sectionTextVariant(node: FlatElement): String =
    (node.props["variant"] ?: node.props["typography"])?.toString()?.lowercase(Locale.ROOT).orEmpty()
