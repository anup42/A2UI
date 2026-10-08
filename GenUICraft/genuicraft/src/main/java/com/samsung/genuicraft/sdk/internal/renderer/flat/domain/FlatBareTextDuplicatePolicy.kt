package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import java.util.Locale

/** Omit only a later bare reference adjacent to its already-retained plain Card. */
internal fun filterAdjacentBareTextDuplicates(
    elementId: String,
    children: List<String>,
    elements: Map<String, FlatElement>,
    direction: String,
    isRepeated: Boolean,
    retainedWrapperIds: Set<String> = children.toSet(),
): List<String> {
    if (direction != "vertical" || isRepeated || children.distinct().size != children.size) return children
    val parent = elements[elementId] ?: return children
    if (parent.type.lowercase(Locale.ROOT) !in setOf("stack", "column") || parent.children != children ||
        !passiveDuplicateNode(parent) || parent.props.keys.any { it !in setOf("direction", "gap", "padding") } ||
        parent.props.values.any { it != null && it !is Number && (it !is String || it.contains("{{")) }) return children
    val declaredDirection = parent.props["direction"]
    if (declaredDirection != null && (declaredDirection !is String || !declaredDirection.trim().equals("vertical", true))) return children
    return children.filterIndexed { index, id ->
        val previous = children.getOrNull(index - 1)
        val wrapper = previous?.let(elements::get)
        val text = elements[id]
        !(previous != null && previous in retainedWrapperIds && wrapper != null && wrapper.type.equals("card", true) &&
            wrapper.props.isEmpty() && wrapper.children == listOf(id) && passiveDuplicateNode(wrapper) &&
            text != null && plainDuplicateText(text))
    }
}

private fun passiveDuplicateNode(node: FlatElement): Boolean =
    node.repeat == null && node.visible == null && node.on.isNullOrEmpty() && node.watch.isNullOrEmpty()

private fun plainDuplicateText(node: FlatElement): Boolean {
    if (!node.type.equals("text", true) || node.children.isNotEmpty() || !passiveDuplicateNode(node) ||
        node.props.keys.any { it !in setOf("text", "variant", "typography") }) return false
    val text = node.props["text"] as? String ?: return false
    if (text.isBlank() || text.contains("{{")) return false
    if (listOf("variant", "typography").any { key ->
            val value = node.props[key]
            value != null && (value !is String || value.contains("{{"))
        }) return false
    val variant = (node.props["variant"] ?: node.props["typography"])?.toString()?.lowercase(Locale.ROOT).orEmpty()
    return variant in setOf("body", "caption", "label", "")
}
