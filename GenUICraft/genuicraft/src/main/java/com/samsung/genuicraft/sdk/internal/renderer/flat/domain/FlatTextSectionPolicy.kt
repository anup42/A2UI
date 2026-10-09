package com.samsung.genuicraft.sdk.internal.renderer.flat.domain

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement

/** A presentation-only group. Original Text IDs and their order remain authoritative. */
internal data class FlatTextSection(val headingId: String?, val childIds: List<String>) {
    val stableKey: String get() = childIds.first()
}

internal fun planFlatTextSections(
    elementId: String,
    children: List<String>,
    elements: Map<String, FlatElement>,
    isRoot: Boolean,
    direction: String,
    isRepeated: Boolean,
): List<FlatTextSection>? {
    if (!isRoot || direction != "vertical" || isRepeated || children.distinct().size != children.size) return null
    val parent = elements[elementId] ?: return null
    if (parent.type.lowercase() !in setOf("stack", "column") || !isPassiveTextSectionElement(parent)) return null
    val nodes = children.map { elements[it] ?: return null }
    if (nodes.any { node ->
            !isPlainSectionDivider(node) && (!node.type.equals("text", true) || node.children.isNotEmpty() ||
                !isPassiveTextSectionElement(node) || literalSectionText(node) == null)
        }) return planMixedTextSections(elementId, parent, children, nodes, elements)

    val sections = mutableListOf<FlatTextSection>()
    var headingId: String? = null
    var sectionChildren = mutableListOf<String>()
    fun finishSection() {
        if (sectionChildren.isNotEmpty()) sections += FlatTextSection(headingId, sectionChildren.toList())
        headingId = null
        sectionChildren = mutableListOf()
    }
    children.zip(nodes).forEachIndexed { index, (id, node) ->
        if (isPlainSectionDivider(node)) {
            // Card boundaries replace only an unstyled separator between complete heading-led runs.
            if (headingId == null || sectionChildren.size < 2 ||
                nodes.getOrNull(index + 1)?.let(::isTextSectionHeading) != true) return null
            finishSection()
            return@forEachIndexed
        }
        val variant = (node.props["variant"] ?: node.props["typography"])?.toString()?.lowercase().orEmpty()
        when {
            isTextSectionHeading(node) -> {
                finishSection()
                headingId = id
                sectionChildren += id
            }
            variant == "h1" -> {
                finishSection()
                sections += FlatTextSection(null, listOf(id))
            }
            headingId != null -> sectionChildren += id
            else -> sections += FlatTextSection(null, listOf(id))
        }
    }
    finishSection()
    // A pending final heading is allowed so its key survives the next streaming body update.
    if (sections.count { it.headingId != null } < 2 ||
        sections.none { it.headingId != null && it.childIds.size > 1 }) return null
    return sections
}

/** Keep authored tables in place while grouping only complete literal heading/body runs. */
private fun planMixedTextSections(
    elementId: String,
    parent: FlatElement,
    children: List<String>,
    nodes: List<FlatElement>,
    elements: Map<String, FlatElement>,
): List<FlatTextSection>? {
    if (!isPlainMixedSectionContainer(parent)) return null
    val visited = mutableSetOf(elementId)
    fun unchangedTableSibling(id: String, node: FlatElement): Boolean {
        if (!visited.add(id) || !isPassiveTextSectionElement(node)) return false
        return when (node.type.lowercase()) {
            "table" -> node.children.isEmpty()
            "stack", "column" -> {
                if (!isPlainMixedSectionContainer(node)) return false
                val childId = node.children.singleOrNull() ?: return false
                val child = elements[childId] ?: return false
                unchangedTableSibling(childId, child)
            }
            else -> false
        }
    }
    children.zip(nodes).forEach { (id, node) ->
        if (node.type.equals("text", true)) {
            if (!visited.add(id) || !isPlainMixedSectionText(node)) return null
        } else if (!unchangedTableSibling(id, node)) return null
    }

    val sections = mutableListOf<FlatTextSection>()
    var index = 0
    while (index < children.size) {
        val heading = nodes[index]
        var end = index + 1
        if (isTextSectionHeading(heading)) {
            while (end < children.size && nodes[end].type.equals("text", true) &&
                sectionVariant(nodes[end]) in setOf("", "body", "caption", "label")) end++
        }
        if (end > index + 1) {
            sections += FlatTextSection(children[index], children.subList(index, end).toList())
        } else {
            sections += FlatTextSection(null, listOf(children[index]))
        }
        index = end
    }
    return sections.takeIf { plan -> plan.count { it.headingId != null } >= 2 }
}

private fun isPlainMixedSectionContainer(node: FlatElement): Boolean =
    isPassiveTextSectionElement(node) && node.props.keys.all {
        it in setOf("direction", "gap", "padding", "paddingHorizontal", "paddingVertical")
    } && node.props.values.all { value ->
        value == null || value is Number || (value is String && !value.contains("{{"))
    } && (node.props["direction"] == null || node.props["direction"]?.toString()?.equals("vertical", true) == true)

private fun isPlainMixedSectionText(node: FlatElement): Boolean =
    node.children.isEmpty() && isPassiveTextSectionElement(node) &&
        node.props.keys.all { it in setOf("text", "variant", "typography") } &&
        literalSectionText(node)?.isNotBlank() == true &&
        listOf("variant", "typography").all { key ->
            node.props[key] == null || (node.props[key] is String && !node.props[key].toString().contains("{{"))
        }

private fun sectionVariant(node: FlatElement): String =
    (node.props["variant"] ?: node.props["typography"])?.toString()?.lowercase().orEmpty()

private fun isPassiveTextSectionElement(element: FlatElement): Boolean =
    element.repeat == null && element.visible == null && element.on.isNullOrEmpty() && element.watch.isNullOrEmpty() &&
        listOf("action", "actions", "onClick", "onTap", "onSubmit", "href").none { element.props[it] != null }

private fun literalSectionText(element: FlatElement): String? =
    (element.props["text"] ?: element.props["title"] ?: element.props["label"] ?:
        element.props["content"] ?: element.props["value"]).let { value ->
        (value as? String)?.takeUnless { it.contains("{{") }
    }

private fun isPlainSectionDivider(element: FlatElement): Boolean =
    element.type.equals("divider", true) && element.props.isEmpty() && element.children.isEmpty() &&
        isPassiveTextSectionElement(element)

private fun isTextSectionHeading(element: FlatElement): Boolean {
    if (!element.type.equals("text", true)) return false
    val variant = (element.props["variant"] ?: element.props["typography"])?.toString()?.lowercase().orEmpty()
    val text = literalSectionText(element).orEmpty().trim()
    return variant in setOf("h2", "h3") && text.isNotBlank() && text.length <= 120 &&
        !text.endsWith('.') && text.split(Regex("\\s+")).size <= 16
}
