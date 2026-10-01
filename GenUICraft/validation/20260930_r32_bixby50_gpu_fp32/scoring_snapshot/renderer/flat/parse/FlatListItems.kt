package com.samsung.genuicraft.renderer.flat.parse

/** Literal list data. Components are represented by List.children, not call maps. */
internal data class FlatListItem(val text: List<String>, val links: List<String>)

private val listTextFields = listOf("text", "label", "title", "name", "content", "value", "description")
private val listLinkFields = listOf("url", "href", "link", "source")

internal fun parseFlatListItems(value: Any?): List<FlatListItem> {
    val items = value as? List<*> ?: return emptyList()
    return items.mapNotNull { item ->
        when (item) {
            is String -> FlatListItem(listOf(item), emptyList())
            is Map<*, *> -> {
                val allowed = (listTextFields + listLinkFields).toSet()
                // The active compiler rejects unsupported objects. Never
                // display a serialized component call as ordinary prose.
                if (item.isEmpty() || item.keys.any { it !in allowed } || item.values.any { it !is String }) {
                    null
                } else {
                    FlatListItem(
                        listTextFields.mapNotNull { item[it] as? String }.filter { it.isNotBlank() }.distinct(),
                        listLinkFields.mapNotNull { item[it] as? String }.filter { it.isNotBlank() }.distinct()
                    )
                }
            }
            else -> null
        }
    }
}
