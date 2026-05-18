package com.samsung.genuicraft.inference

import android.content.Context
import java.io.File

object OnDeviceModelCatalog {
    data class Entry(
        val id: String,
        val displayName: String,
        val subtitle: String,
        val repoId: String,
        val fileName: String,
        val downloadUrl: String,
        val approximateSize: String
    ) {
        fun localFile(context: Context): File {
            val dir = File(context.getExternalFilesDir(null), "on_device_models")
            return File(dir, fileName)
        }

        fun localPath(context: Context): String = localFile(context).absolutePath

        fun isDownloaded(context: Context): Boolean {
            val file = localFile(context)
            return file.exists() && file.length() > 0L
        }
    }

    val entries: List<Entry> = listOf(
        Entry(
            id = "gemma4_e2b_it_litert",
            displayName = "Gemma 4 E2B IT",
            subtitle = "Smaller LiteRT IR model. Prefer this for faster on-device Stage 3.",
            repoId = "litert-community/gemma-4-E2B-it-litert-lm",
            fileName = "gemma-4-E2B-it.litertlm",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
            approximateSize = "2.58 GB"
        ),
        Entry(
            id = "gemma4_e4b_it_litert",
            displayName = "Gemma 4 E4B IT",
            subtitle = "Larger LiteRT IR model. Use when quality matters more than size/latency.",
            repoId = "litert-community/gemma-4-E4B-it-litert-lm",
            fileName = "gemma-4-E4B-it.litertlm",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
            approximateSize = "3.66 GB"
        )
    )

    fun selectedEntry(context: Context): Entry? {
        val selectedPath = com.samsung.genuicraft.InferenceBackendSettings.getOnDeviceModelPath(context)
        return entries.firstOrNull { it.localPath(context) == selectedPath }
    }
}
