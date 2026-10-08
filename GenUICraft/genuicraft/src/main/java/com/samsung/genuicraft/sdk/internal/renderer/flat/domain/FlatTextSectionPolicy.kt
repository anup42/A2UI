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
        }) return null

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

private fun isPassiveTextSectionElement(element: FlatElement): Boolean =
    element.repeat == null && element.visible == null && element.on.isNullOrEmpty() && element.watch.isNullOrEmpty() &&
        listOf("action", "actions", "onClick", "onTap", "onSubmit", "href").none { element.props[it] != null }

private fun literalSectionText(element: FlatElement): String? =
    (element.props["text"] ?: element.props["title"] ?: element.props["label"] ?:
        element.props["content"] ?: element.props["value"]) as? String

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
