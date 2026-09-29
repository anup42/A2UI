package com.samsung.genuicraft.sdk.provider

import android.util.Log
import com.google.gson.JsonParser
import java.io.File
import java.security.MessageDigest

/**
 * Opt-in compatibility correction for the pinned LiteRT OpenCL compiler. It is deliberately
 * separate from MODEL_DEFAULT: that path still contains the original half-precision faults.
 * The prepared package must cover both the target and (when requested) the MTP drafter.
 */
internal object GpuFp16Correction {
    const val POLICY = "genuicraft-fp16-rope-qdq-v1"
    const val SOURCE_SHA256 = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
    const val MODEL_SHA256 = "7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373"
    private var verifiedFile: String? = null
    private var loaded = false

    @Synchronized
    fun validateModel(config: ValidatedGemma4Config, expectedModelHash: String = MODEL_SHA256): File {
        val model = File(config.modelPath).canonicalFile
        val manifest = File(model.path + ".fp16.json")
        require(manifest.isFile && manifest.length() < 1024 * 1024) {
            "Corrected FP16 requires a prepared model and its .litertlm.fp16.json manifest."
        }
        val record = manifest.reader().use { JsonParser.parseReader(it).asJsonObject }
        require(record["schema_version"]?.asInt == 1 && record["precision_policy"]?.asString == POLICY) {
            "Unsupported corrected FP16 model policy. Prepare the model with GenUICraft/tools/prepare_fp16_model.py."
        }
        require(record["source_sha256"]?.asString == SOURCE_SHA256 &&
            record["transform_version"]?.asString == "pinned-r64-rope-target-mtp-8192-v1") {
            "Corrected FP16 currently supports only the independently verified rank-64 export."
        }
        require(record["target_rope_corrected"]?.asBoolean == true) { "Target RoPE correction is missing." }
        require(!config.enableSpeculativeDecoding || record["drafter_rope_corrected"]?.asBoolean == true) {
            "This corrected model does not cover the MTP drafter. Disable MTP."
        }
        require(config.maxContextTokens <= record["max_context_tokens"].asInt && config.maxContextTokens <= 8192) {
            "This corrected FP16 model supports at most 8192 context tokens."
        }
        require(record["model_bytes"].asLong == model.length()) { "Corrected FP16 model is incomplete." }
        val expected = record["model_sha256"].asString.lowercase()
        require(expected == expectedModelHash) { "This prepared model has not been verified for corrected FP16." }
        require(expected.matches(Regex("[0-9a-f]{64}"))) { "Invalid corrected model hash." }
        val identity = "${model.path}:${model.length()}:${model.lastModified()}:$expected"
        if (identity != verifiedFile) {
            val digest = MessageDigest.getInstance("SHA-256")
            model.inputStream().buffered(1024 * 1024).use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            require(actual == expected) { "Corrected FP16 model hash differs from its preparation manifest." }
            verifiedFile = identity
        }
        return model
    }

    @Synchronized
    fun installIfAvailable(): Boolean {
        if (!loaded) {
            System.loadLibrary("LiteRtOpenClAccelerator")
            System.loadLibrary("litertlm_jni")
            System.loadLibrary("GenUiFp16Correction")
            loaded = true
        }
        return nativeInstall()
    }

    @Synchronized
    fun enableForCompilation() {
        check(installIfAvailable()) {
            "Corrected FP16 is unavailable for the packaged OpenCL runtime. Use FP32."
        }
        nativeSetEnabled(true)
    }

    @Synchronized
    fun disableForCompilation() {
        if (loaded) {
            nativeSetEnabled(false)
            Log.i("GenUICraftFp16", "policy=$POLICY ${nativeStatistics()}")
        }
    }

    fun requireApplied() {
        check(nativePatchedBlocks() > 0) {
            "The GPU compiler did not apply the FP16 Q/DQ correction. Select FP32."
        }
    }

    private external fun nativeInstall(): Boolean
    private external fun nativeSetEnabled(enabled: Boolean)
    private external fun nativeStatistics(): String
    private external fun nativePatchedBlocks(): Int
}
