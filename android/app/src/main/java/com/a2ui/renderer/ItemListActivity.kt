package com.samsung.genuicraft

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class ItemListActivity : AppCompatActivity() {
    private lateinit var sourceText: TextView
    private lateinit var countText: TextView
    private lateinit var emptyText: TextView
    private lateinit var recyclerView: RecyclerView

    private val itemAdapter = GenUiItemAdapter { position ->
        startActivity(
            Intent(this, RenderActivity::class.java)
                .putExtra(RenderActivity.EXTRA_RECORD_INDEX, position)
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_item_list)

        sourceText = findViewById(R.id.sourceText)
        countText = findViewById(R.id.countText)
        emptyText = findViewById(R.id.emptyText)
        recyclerView = findViewById(R.id.itemRecyclerView)

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = itemAdapter
    }

    override fun onResume() {
        super.onResume()
        bindSession()
    }

    private fun bindSession() {
        val session = RenderSessionStore.current()
        if (session == null || session.records.isEmpty()) {
            sourceText.text = getString(R.string.list_empty_source)
            countText.text = getString(R.string.list_empty_count)
            emptyText.visibility = View.VISIBLE
            recyclerView.visibility = View.GONE
            return
        }

        sourceText.text = session.sourceLabel
        countText.text = getString(R.string.list_count, session.records.size)
        emptyText.visibility = View.GONE
        recyclerView.visibility = View.VISIBLE
        itemAdapter.submitRecords(session.records)
    }
}

