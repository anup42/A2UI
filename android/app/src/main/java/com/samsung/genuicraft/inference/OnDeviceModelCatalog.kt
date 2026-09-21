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
        val enableSpeculativeDecoding: Boolean = false,
        val rawStage3Response: Boolean = false,
        val useRawTrainingWrapper: Boolean = false,
        val trainingCompatiblePrompt: Boolean = false,
        val allowGpuQualityFallback: Boolean = false,
        val usesTrainedSdkConverter: Boolean = false,
        val stage3TrainingPromptPrefix: String? = null,
        val additionalLocalRelativePaths: List<String> = emptyList(),
        val minimumFileSizeBytes: Long = 1L,
    ) {
        val isDownloadable: Boolean
            get() = !downloadUrl.isNullOrBlank()

        fun localFile(context: Context): File {
            return localFileCandidates(context).firstOrNull(::isUsable)
                ?: externalLocalFile(context)
        }

        internal fun matchesSelection(selection: String): Boolean {
            val normalized = selection.trim()
            if (id.equals(normalized, ignoreCase = true)) {
                return true
            }
            val selectedFileName = File(normalized).name
            return acceptedFileNames().any { it.equals(selectedFileName, ignoreCase = true) }
        }

        private fun localFileCandidates(context: Context): List<File> {
            val externalRoot = context.getExternalFilesDir(null)
            val internalRoot = context.filesDir
            return buildList {
                additionalLocalRelativePaths.forEach { relativePath ->
                    if (externalRoot != null) add(File(externalRoot, relativePath))
                    add(File(internalRoot, relativePath))
                }
                add(externalLocalFile(context))
                add(internalLocalFile(context))
            }.distinctBy { it.absolutePath }
        }

        private fun acceptedFileNames(): List<String> {
            return buildList {
                add(fileName)
                additionalLocalRelativePaths.forEach { add(File(it).name) }
            }.distinct()
        }

        private fun externalLocalFile(context: Context): File {
            val dir = File(context.getExternalFilesDir(null) ?: context.filesDir, "on_device_models")
            // Create the directory from the app process so scoped-storage ownership is correct.
            dir.mkdirs()
            return File(dir, fileName)
        }

        private fun internalLocalFile(context: Context): File {
            val dir = File(context.filesDir, "on_device_models")
            dir.mkdirs()
            return File(dir, fileName)
        }

        private fun isUsable(file: File): Boolean {
            return file.isFile && file.length() >= minimumFileSizeBytes
        }

        fun localPath(context: Context): String = localFile(context).absolutePath

        fun isDownloaded(context: Context): Boolean {
            return localFileCandidates(context).any(::isUsable)
        }
    }

    val entries: List<Entry> = listOf(
        Entry(
            id = "gemma4_e2b_a2ui_mobile",
            displayName = "Trained Gemma 4 E2B (A2UI Mobile)",
            subtitle =
                "Current trained A2UI Express model for GenUI IR. GPU required; " +
                    "MTP is available when the package includes its drafter.",
            repoId = "local/gemma-4-e2b-a2ui-mobile",
            fileName = "gemma4_e2b_a2ui_mobile.litertlm",
            downloadUrl = null,
            approximateSize = "~2.6 GB",
            quantization = "Mixed W2/W4/W8-A8 mobile topology",
            maxContextTokens = 8_192,
            maxOutputTokens = 2_048,
            requireGpu = true,
            enableSpeculativeDecoding = true,
            trainingCompatiblePrompt = true,
            usesTrainedSdkConverter = true,
            additionalLocalRelativePaths = listOf(
                "sdk_models/gemma4_e2b_a2ui_mobile.litertlm",
                "sdk_models/e2b_v10_w4.litertlm",
            ),
            minimumFileSizeBytes = 2_500_000_000L,
        ),
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
            maxOutputTokens = 3_072,
            requireGpu = true,
            rawStage3Response = true,
            useRawTrainingWrapper = true,
            stage3TrainingPromptPrefix =
                "Given an agent response you have to generate a structured intermediate representation. ",
        ),
        Entry(
            id = "gemma3_270m_a2ui_express_int8",
            displayName = "Gemma 3 270M A2UI Express",
            subtitle = "Custom A2UI Express LoRA/QAT model. GPU is attempted first; AUTO falls back to CPU if this device returns invalid GPU logits.",
            repoId = "local/gemma-3-270m-a2ui-express-int8",
            fileName = "gemma-3-270m-a2ui-express-int8.litertlm",
            downloadUrl = null,
            approximateSize = "~288 MB",
            quantization = "INT8 per-channel",
            maxContextTokens = 4_096,
            maxOutputTokens = 2_048,
            requireGpu = true,
            trainingCompatiblePrompt = true,
            allowGpuQualityFallback = true,
            minimumFileSizeBytes = 200_000_000L,
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
            minimumFileSizeBytes = 2_500_000_000L,
        ),
        Entry(
            id = "gemma4_e2b_trained_express",
            displayName = "Gemma 4 E2B Trained Express",
            subtitle =
                "Retained-scale A2UI Express package in the released Gemma 4 mobile " +
                    "topology. GPU-only with the unchanged default MTP assistant; validate " +
                    "target-only strict IR before enabling MTP.",
            repoId = "local/gemma-4-e2b-trained-express",
            fileName = "gemma-4-e2b-trained-express-int4.litertlm",
            downloadUrl = null,
            approximateSize = "2.59 GB",
            quantization = "Mixed W2/W4/W8-A8 mobile topology",
            maxContextTokens = 4_096,
            maxOutputTokens = 2_048,
            requireGpu = true,
            enableSpeculativeDecoding = true,
            trainingCompatiblePrompt = true,
            minimumFileSizeBytes = 2_500_000_000L,
        ),
        Entry(
            id = "gemma4_e2b_a2ui_express_v6_litert",
            displayName = "Gemma 4 E2B A2UI Express v6",
            subtitle = "Custom v6 A2UI Express model exported for LiteRT-LM. GPU is the default; NPU needs a device-specific compiled model and vendor runtime.",
            repoId = "local/gemma-4-e2b-a2ui-express-v6",
            fileName = "gemma-4-e2b-a2ui-express-v6.litertlm",
            downloadUrl = null,
            approximateSize = "~2.65 GiB (2.84 GB)",
            quantization = "INT4 weights / FP32 activations, blockwise-32",
            maxContextTokens = 4_096,
            maxOutputTokens = 2_048,
            requireGpu = true,
            trainingCompatiblePrompt = true,
            minimumFileSizeBytes = 2_500_000_000L,
        ),
        Entry(
            id = "gemma4_e2b_it_litert",
            displayName = "Official Gemma 4 E2B",
            subtitle = "Official instruction-tuned LiteRT model with an MTP drafter.",
            repoId = "litert-community/gemma-4-E2B-it-litert-lm",
            fileName = "gemma-4-E2B-it.litertlm",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
            approximateSize = "2.58 GB",
            quantization = "INT4 per-channel",
            enableSpeculativeDecoding = true,
            additionalLocalRelativePaths = listOf(
                "sdk_models/gemma-4-E2B-it.litertlm",
            ),
            minimumFileSizeBytes = 2_500_000_000L,
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
            enableSpeculativeDecoding = true,
            minimumFileSizeBytes = 3_500_000_000L,
        )
    )

    /** Models deliberately offered in Settings, independent of legacy runtime lookup support. */
    val visibleEntries: List<Entry> = listOfNotNull(
        entries.firstOrNull { it.id == "gemma4_e2b_a2ui_mobile" },
        entries.firstOrNull { it.id == "gemma4_e2b_it_litert" },
        entries.firstOrNull { it.id == "gemma3_270m_a2ui_express_int8" },
    )

    fun selectedEntry(context: Context): Entry? {
        val selectedPath = com.samsung.genuicraft.InferenceBackendSettings.getOnDeviceModelPath(context)
        return entryForModelPath(selectedPath)
    }

    fun entryForModelPath(modelPath: String): Entry? {
        return entries.firstOrNull { it.matchesSelection(modelPath) }
    }

    internal fun migrationTargetForSelection(
        storedSelection: String,
        availableVisibleEntryIds: Set<String>,
    ): Entry? {
        val selectedEntry = entryForModelPath(storedSelection) ?: return null
        if (visibleEntries.any { it.id == selectedEntry.id }) {
            return null
        }
        val preferredReplacementId = when (selectedEntry.id) {
            "gemma4_e2b_ir_trained_int4",
            "gemma4_e2b_trained_express",
            "gemma4_e2b_a2ui_express_v6_litert" -> "gemma4_e2b_a2ui_mobile"
            "gemma4_e4b_it_litert" -> "gemma4_e2b_it_litert"
            "gemma3_270m_ir_int8" -> "gemma3_270m_a2ui_express_int8"
            else -> return null
        }
        return visibleEntries.firstOrNull {
            it.id == preferredReplacementId && it.id in availableVisibleEntryIds
        }
    }

    /**
     * Moves a hidden legacy selection to an installed visible successor. If no retained model is
     * installed, the old path remains usable through [entryForModelPath] instead of being broken.
     */
    internal fun migrateLegacySelection(context: Context, storedSelection: String): String {
        val selectedEntry = entryForModelPath(storedSelection)
        if (selectedEntry != null &&
            visibleEntries.any { it.id == selectedEntry.id } &&
            selectedEntry.isDownloaded(context)
        ) {
            // Canonicalize a stored ID or alias to the installed file that Settings will display.
            return selectedEntry.localPath(context)
        }
        val availableVisibleEntryIds = visibleEntries
            .filter { it.isDownloaded(context) }
            .mapTo(linkedSetOf()) { it.id }
        return migrationTargetForSelection(storedSelection, availableVisibleEntryIds)
            ?.localPath(context)
            ?: storedSelection.trim()
    }
}
