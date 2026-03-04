package com.samsung.genuicraft

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class GenUiItemAdapter(
    private val onItemClick: (position: Int) -> Unit
) : RecyclerView.Adapter<GenUiItemAdapter.ItemViewHolder>() {
    private val items = mutableListOf<GenUiRecord>()

    fun submitRecords(records: List<GenUiRecord>) {
        items.clear()
        items.addAll(records)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ItemViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_genui_entry, parent, false)
        return ItemViewHolder(view, onItemClick)
    }

    override fun onBindViewHolder(holder: ItemViewHolder, position: Int) {
        holder.bind(items[position], position)
    }

    override fun getItemCount(): Int = items.size

    class ItemViewHolder(
        view: View,
        onItemClick: (position: Int) -> Unit
    ) : RecyclerView.ViewHolder(view) {
        private val indexBadge: TextView = view.findViewById(R.id.indexBadge)
        private val titleText: TextView = view.findViewById(R.id.itemTitle)
        private val metaText: TextView = view.findViewById(R.id.itemMeta)
        private val sourceText: TextView = view.findViewById(R.id.itemSource)

        init {
            view.setOnClickListener {
                val position = bindingAdapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onItemClick(position)
                }
            }
        }

        fun bind(item: GenUiRecord, position: Int) {
            indexBadge.text = (position + 1).toString().padStart(2, '0')
            titleText.text = item.uiId ?: item.title
            metaText.text = buildMeta(item)
            sourceText.text = item.sourceLabel
            applyAccessibleListLayout()
        }

        private fun buildMeta(item: GenUiRecord): String {
            val parts = mutableListOf<String>()
            if (!item.queryId.isNullOrBlank()) {
                parts += "query: ${item.queryId}"
            }
            if (!item.responseId.isNullOrBlank()) {
                parts += "response: ${item.responseId}"
            }
            if (parts.isEmpty()) {
                return "Tap to render this UI"
            }
            return parts.joinToString("  |  ")
        }

        private fun applyAccessibleListLayout() {
            val configuration = itemView.resources.configuration
            val widthDp = configuration.screenWidthDp
            val fontScale = configuration.fontScale
            val allowExpanded =
                (widthDp <= 320 && fontScale >= 1.15f) ||
                    (widthDp < 411 && fontScale >= 1.3f)

            if (!allowExpanded) {
                titleText.maxLines = 1
                metaText.maxLines = 3
                return
            }

            titleText.maxLines = 2
            metaText.maxLines = 4

            // Apply expanded layout only when text actually wraps.
            titleText.post {
                val wraps = (titleText.layout?.lineCount ?: 0) > 1
                titleText.maxLines = if (wraps) 2 else 1
                metaText.maxLines = if (wraps) 4 else 3
            }
        }
    }
}

