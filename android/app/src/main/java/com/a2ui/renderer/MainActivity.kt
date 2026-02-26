package com.samsung.genuicraft

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView

    private val filePicker =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri == null) {
                showStatus(getString(R.string.status_file_picker_cancelled), isError = false)
            } else {
                loadFromUri(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        val selectFileButton: MaterialButton = findViewById(R.id.selectFileButton)
        val loadSampleButton: MaterialButton = findViewById(R.id.loadSampleButton)

        selectFileButton.setOnClickListener {
            filePicker.launch(arrayOf("application/json", "text/plain", "*/*"))
        }
        loadSampleButton.setOnClickListener { loadBundledSample() }

        showStatus(getString(R.string.status_ready_select_source), isError = false)
    }

    private fun loadFromUri(uri: Uri) {
        try {
            contentResolver.openInputStream(uri).use { input ->
                if (input == null) {
                    showStatus(getString(R.string.error_uri_open), isError = true)
                    return
                }
                val raw = input.bufferedReader(Charsets.UTF_8).use { it.readText() }
                val sourceLabel = resolveDisplayName(uri)
                val records = GenUiRecordParser.parseRecords(
                    raw = raw,
                    sourceDir = resolveFileSourceDir(uri),
                    sourceLabel = sourceLabel
                )
                openItemList(sourceLabel, records)
            }
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun loadBundledSample() {
        try {
            val sourceLabel = "sample_genui.jsonl"
            val raw = assets.open(sourceLabel).bufferedReader(Charsets.UTF_8).use { it.readText() }
            val records = GenUiRecordParser.parseRecords(
                raw = raw,
                sourceDir = null,
                sourceLabel = sourceLabel
            )
            openItemList(sourceLabel, records)
        } catch (exc: Exception) {
            showError(exc)
        }
    }

    private fun openItemList(sourceLabel: String, records: List<GenUiRecord>) {
        if (records.isEmpty()) {
            showStatus(getString(R.string.error_no_valid_json, sourceLabel), isError = true)
            return
        }

        RenderSessionStore.update(sourceLabel = sourceLabel, records = records)
        showStatus(getString(R.string.status_loaded_records, records.size, sourceLabel), isError = false)
        startActivity(Intent(this, ItemListActivity::class.java))
    }

    private fun resolveDisplayName(uri: Uri): String {
        if (uri.scheme.equals("content", ignoreCase = true)) {
            contentResolver
                .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0 && cursor.moveToFirst()) {
                        val name = cursor.getString(index)
                        if (!name.isNullOrBlank()) {
                            return name
                        }
                    }
                }
        }

        val path = uri.lastPathSegment
        return if (!path.isNullOrBlank()) path.substringAfterLast('/') else uri.toString()
    }

    private fun resolveFileSourceDir(uri: Uri): File? {
        if (!uri.scheme.equals("file", ignoreCase = true)) {
            return null
        }
        return uri.path?.let { File(it).parentFile }
    }

    private fun showError(exc: Exception) {
        val msg = exc.message?.trim().takeUnless { it.isNullOrEmpty() } ?: exc.javaClass.simpleName
        showStatus(getString(R.string.error_render_failed, msg), isError = true)
    }

    private fun showStatus(message: String, isError: Boolean) {
        statusText.text = message
        statusText.setTextColor(if (isError) Color.parseColor("#9E2A2B") else Color.parseColor("#0D5A4D"))
    }
}

