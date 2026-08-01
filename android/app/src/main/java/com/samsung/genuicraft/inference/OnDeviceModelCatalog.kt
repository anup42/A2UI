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
        val downloadUrl: String?,
        val approximateSize: String,
        val quantization: String,
        val maxContextTokens: Int = 12_288,
        val maxOutputTokens: Int = 3_072,
        val requireGpu: Boolean = false,
        val rawStage3Response: Boolean = false,
        val useRawTrainingWrapper: Boolean = false,
        val stage3TrainingPromptPrefix: String? = null,
    ) {
        val isDownloadable: Boolean
            get() = !downloadUrl.isNullOrBlank()

        fun localFile(context: Context): File {
            val dir = File(context.getExternalFilesDir(null), "on_device_models")
            // Create the directory from the app process so scoped-storage ownership is correct.
            dir.mkdirs()
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
            id = "gemma3_270m_ir_int8",
            displayName = "Gemma 3 270M IR",
            subtitle = "Experimental custom LoRA IR model. INT8 and GPU-only; install it with android/tools/deploy_gemma270m.ps1.",
            repoId = "local/gemma-3-270m-ir-lora",
            fileName = "gemma-3-270m-ir-int8.litertlm",
            downloadUrl = null,
            approximateSize = "~288 MB",
            quantization = "INT8 per-channel",
            maxContextTokens = 4_096,
            maxOutputTokens = 1_024,
            requireGpu = true,
            rawStage3Response = true,
            useRawTrainingWrapper = true,
        ),
        Entry(
            id = "gemma4_e2b_ir_trained_int4",
            displayName = "Gemma 4 E2B Trained IR",
            subtitle = "Custom LoRA-trained Stage 3 IR model. Attention-protected INT4/INT8 and GPU-only; install it with android/tools/deploy_gemma4_e2b_trained.ps1.",
            repoId = "local/gemma-4-e2b-ir-trained",
            fileName = "gemma-4-e2b-ir-trained-int4.litertlm",
            downloadUrl = null,
            approximateSize = "~2.8 GiB",
            quantization = "INT4/INT8 attention-protected blockwise-32",
            maxContextTokens = 4_096,
            maxOutputTokens = 2_048,
            requireGpu = true,
            rawStage3Response = true,
            stage3TrainingPromptPrefix =
                "Given an agent response you have to generate a structured intermediate representation. ",
        ),
        Entry(
            id = "gemma4_e2b_it_litert",
            displayName = "Gemma 4 E2B IT",
            subtitle = "Smaller LiteRT IR model. Prefer this for faster on-device Stage 3.",
            repoId = "litert-community/gemma-4-E2B-it-litert-lm",
            fileName = "gemma-4-E2B-it.litertlm",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
            approximateSize = "2.58 GB",
            quantization = "INT4 per-channel",
        ),
        Entry(
            id = "gemma4_e4b_it_litert",
            displayName = "Gemma 4 E4B IT",
            subtitle = "Larger LiteRT IR model. Use when quality matters more than size/latency.",
            repoId = "litert-community/gemma-4-E4B-it-litert-lm",
            fileName = "gemma-4-E4B-it.litertlm",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
            approximateSize = "3.66 GB",
            quantization = "INT4 per-channel",
        )
    )

    fun selectedEntry(context: Context): Entry? {
        val selectedPath = com.samsung.genuicraft.InferenceBackendSettings.getOnDeviceModelPath(context)
        return entries.firstOrNull { it.localPath(context) == selectedPath }
    }

    fun entryForModelPath(modelPath: String): Entry? {
        val fileName = File(modelPath.trim()).name
        return entries.firstOrNull { it.fileName.equals(fileName, ignoreCase = true) }
    }
}
