package com.samsung.genuicraft.sdk.provider

import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Properties

/**
 * LiteRT-LM 0.16.1 reads prefer_activation_type from section metadata, but its Kotlin API
 * cannot override it. Prepare a private copy using that supported setting; never edit the source,
 * graph, weights, tokenizer, vision preferences, or drafter. Remove this adapter when the released
 * Kotlin EngineConfig exposes activationDataType. No native ABI interception is involved.
 *
 * Schema: google-ai-edge/LiteRT-LM, v0.16.1, schema/core/litertlm_header_schema.fbs.
 * Consumer: runtime/engine/engine_settings.cc, MaybeOverrideActivationType.
 */
internal object GpuFp32ModelCache {
    @Synchronized
    fun prepare(source: File, cacheRoot: File? = null, checkCancelled: () -> Unit = {}): File {
        checkCancelled()
        val original = source.canonicalFile
        val length = original.length()
        val modified = original.lastModified()
        val header = LiteRtPrecisionHeader.read(original)
        val patched = LiteRtPrecisionHeader.fp32(header, length)
        if (header.contentEquals(patched)) return original

        // Header digest also prevents reuse when a package's metadata changes without its size.
        val fingerprint = sha256("${original.path}\n$length\n$modified\n${sha256(header)}".toByteArray())
        val root = File(cacheRoot ?: original.parentFile, "genuicraft-gpu-fp32")
        val directory = File(root, fingerprint)
        check((directory.isDirectory || directory.mkdirs()) && directory.canWrite()) {
            "Cannot prepare GPU FP32 model. Provide a writable Gemma4Config.cacheDir."
        }
        val target = File(directory, "model-gpu-fp32.litertlm")
        val manifest = File(directory, "manifest.properties")
        RandomAccessFile(File(directory, "prepare.lock"), "rw").use { lock ->
            lock.channel.lock().use {
                checkCancelled()
                val saved = runCatching {
                    Properties().apply { manifest.inputStream().use(::load) }
                }.getOrNull()
                if (target.isFile && target.length() == length &&
                    saved?.getProperty("fingerprint") == fingerprint &&
                    saved.getProperty("preparedModified") == target.lastModified().toString() &&
                    runCatching { LiteRtPrecisionHeader.read(target).contentEquals(patched) }.getOrDefault(false)
                ) return target

                // A killed process cannot run finally. Under the exclusive preparation lock,
                // these files can only belong to an abandoned copy of this exact revision.
                directory.listFiles()?.filter {
                    it.isFile && it.name.endsWith(".tmp") &&
                        (it.name.startsWith("prepare-") || it.name.startsWith("manifest-"))
                }?.forEach { check(it.delete()) { "Cannot remove incomplete GPU FP32 cache file: ${it.path}" } }
                check(!target.exists() || target.delete()) { "Cannot replace incomplete GPU FP32 cache: ${target.path}" }
                check(!manifest.exists() || manifest.delete()) { "Cannot replace incomplete GPU FP32 manifest." }
                check(directory.usableSpace >= length + 16L * 1024 * 1024) {
                    "GPU FP32 needs a one-time cached model copy (${length / (1024 * 1024)} MB). " +
                        "Free storage or set Gemma4Config.cacheDir to a location with enough space."
                }
                val pending = File.createTempFile("prepare-", ".tmp", directory)
                val pendingManifest = File.createTempFile("manifest-", ".tmp", directory)
                try {
                    val sourceHash = MessageDigest.getInstance("SHA-256")
                    val preparedHash = MessageDigest.getInstance("SHA-256")
                    RandomAccessFile(original, "r").use { input ->
                        val freshHeader = ByteArray(header.size).also(input::readFully)
                        check(freshHeader.contentEquals(header)) { "Model changed during FP32 preparation. Retry." }
                        sourceHash.update(header)
                        preparedHash.update(patched)
                        FileOutputStream(pending).use { output ->
                            output.write(patched)
                            val buffer = ByteArray(1024 * 1024)
                            while (true) {
                                checkCancelled()
                                val count = input.read(buffer)
                                if (count < 0) break
                                output.write(buffer, 0, count)
                                sourceHash.update(buffer, 0, count)
                                preparedHash.update(buffer, 0, count)
                            }
                            output.fd.sync()
                        }
                    }
                    checkCancelled()
                    check(original.length() == length && original.lastModified() == modified && pending.length() == length) {
                        "Model changed during FP32 preparation. Retry after the model copy completes."
                    }
                    Files.move(pending.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                    val record = Properties().apply {
                        setProperty("fingerprint", fingerprint)
                        setProperty("source", original.path)
                        setProperty("sourceSize", length.toString())
                        setProperty("sourceModified", modified.toString())
                        setProperty("sourceSha256", hex(sourceHash.digest()))
                        setProperty("preparedSha256", hex(preparedHash.digest()))
                        setProperty("preparedModified", target.lastModified().toString())
                        setProperty("textActivationType", "fp32")
                    }
                    FileOutputStream(pendingManifest).use { output ->
                        record.store(output, "GenUICraft metadata-only GPU FP32 cache v1")
                        output.fd.sync()
                    }
                    Files.move(pendingManifest.toPath(), manifest.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                    return target
                } finally {
                    pending.delete()
                    pendingManifest.delete()
                }
            }
        }
    }

    private fun sha256(bytes: ByteArray): String = hex(MessageDigest.getInstance("SHA-256").digest(bytes))
    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }
}

/** Bounded reader for the small container header, not for the multi-GB model sections. */
internal object LiteRtPrecisionHeader {
    private const val PREFIX_BYTES = 32
    private const val MAX_HEADER_BYTES = 1024 * 1024

    fun read(file: File): ByteArray = RandomAccessFile(file, "r").use { input ->
        require(input.length() >= PREFIX_BYTES) { "Invalid LiteRT-LM header: truncated prefix." }
        val prefix = ByteArray(PREFIX_BYTES).also(input::readFully)
        val end = headerSize(prefix, input.length())
        prefix.copyOf(end).also { input.readFully(it, PREFIX_BYTES, end - PREFIX_BYTES) }
    }

    fun fp32(header: ByteArray, modelLength: Long): ByteArray {
        require(header.size == headerSize(header, modelLength)) { "Invalid LiteRT-LM header length." }
        val reader = Reader(header)
        val root = reader.follow(PREFIX_BYTES)
        val stringUses = mutableMapOf<Int, Int>()
        fun trackedString(field: Int): Text {
            val value = reader.string(field)
            stringUses[value.start] = (stringUses[value.start] ?: 0) + 1
            return value
        }
        fun entries(field: Int?): Map<String, Text?> = reader.tables(field).associate { table ->
            val key = trackedString(reader.required(table, 0, 4)).value
            val type = reader.u8(reader.required(table, 1, 1))
            val union = reader.follow(reader.required(table, 2, 4))
            key to if (type == 9) trackedString(reader.required(union, 0, 4)) else null
        }
        reader.field(root, 0, 4)?.let { entries(reader.field(reader.follow(it), 0, 4)) }
        val metadata = reader.follow(reader.required(root, 1, 4))
        val targets = mutableListOf<Text>()
        var textSections = 0
        for (section in reader.tables(reader.required(metadata, 0, 4))) {
            val begin = reader.u64(reader.required(section, 1, 8))
            val end = reader.u64(reader.required(section, 2, 8))
            require(begin >= header.size && end > begin && end <= modelLength) { "Invalid LiteRT-LM section bounds." }
            val values = entries(reader.field(section, 0, 4))
            if (values["model_type"]?.value == "tf_lite_prefill_decode") {
                require(reader.u8(reader.required(section, 3, 1)) == 3) { "Text executor must be a TFLiteModel section." }
                textSections++
                val preference = requireNotNull(values["prefer_activation_type"]) {
                    "GPU FP32 requires text prefer_activation_type metadata in this LiteRT-LM package."
                }
                require(preference.value in setOf("fp16", "fp32")) {
                    "Unsupported text activation preference '${preference.value}' for GPU FP32 preparation."
                }
                if (preference.value == "fp16") targets += preference
            }
        }
        require(textSections == 1) { "GPU FP32 requires exactly one tf_lite_prefill_decode section." }
        return header.copyOf().also { result ->
            targets.forEach { target ->
                // FlatBuffers can deduplicate strings. Do not accidentally alter another modality.
                require(stringUses[target.start] == 1) { "Shared activation metadata requires a fresh FP32 model export." }
                "fp32".toByteArray(Charsets.UTF_8).copyInto(result, target.start)
            }
        }
    }

    private fun headerSize(prefix: ByteArray, modelLength: Long): Int {
        require(prefix.size >= PREFIX_BYTES && prefix.copyOfRange(0, 8).contentEquals("LITERTLM".toByteArray())) {
            "Invalid LiteRT-LM magic."
        }
        val buffer = ByteBuffer.wrap(prefix).order(ByteOrder.LITTLE_ENDIAN)
        require(buffer.getInt(8) == 1) { "Unsupported LiteRT-LM major version." }
        val end = buffer.getLong(24)
        require(end in 48L..MAX_HEADER_BYTES.toLong() && end < modelLength) { "Invalid LiteRT-LM header bounds." }
        return end.toInt()
    }

    private data class Text(val start: Int, val value: String)

    private class Reader(private val bytes: ByteArray) {
        private val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        private fun checked(offset: Long, count: Long): Int {
            require(offset >= PREFIX_BYTES && count >= 0 && offset <= bytes.size.toLong() - count) {
                "Invalid LiteRT-LM metadata offset."
            }
            return offset.toInt()
        }
        fun u8(offset: Int): Int = bytes[checked(offset.toLong(), 1)].toInt() and 255
        private fun u16(offset: Int): Int = buffer.getShort(checked(offset.toLong(), 2)).toInt() and 65535
        private fun i32(offset: Int): Int = buffer.getInt(checked(offset.toLong(), 4))
        private fun u32(offset: Int): Long = i32(offset).toLong() and 0xffffffffL
        fun u64(offset: Int): Long = buffer.getLong(checked(offset.toLong(), 8)).also {
            require(it >= 0) { "Invalid LiteRT-LM section offset." }
        }
        fun follow(offset: Int): Int {
            val distance = u32(offset)
            require(distance > 0) { "Missing LiteRT-LM metadata reference." }
            return checked(offset.toLong() + distance, 4)
        }
        fun field(table: Int, index: Int, width: Int): Int? {
            val vtable = checked(table.toLong() - i32(table), 4)
            val vtableSize = u16(vtable)
            val tableSize = u16(vtable + 2)
            require(vtableSize >= 4 && vtableSize % 2 == 0 && tableSize >= 4) { "Invalid LiteRT-LM table." }
            checked(vtable.toLong(), vtableSize.toLong())
            checked(table.toLong(), tableSize.toLong())
            val entry = vtable + 4 + index * 2
            if (entry + 2 > vtable + vtableSize) return null
            val distance = u16(entry)
            if (distance == 0) return null
            require(distance >= 4 && distance <= tableSize - width) { "Invalid LiteRT-LM table field." }
            return checked(table.toLong() + distance, width.toLong())
        }
        fun required(table: Int, index: Int, width: Int): Int =
            requireNotNull(field(table, index, width)) { "Missing LiteRT-LM metadata field." }
        fun string(field: Int): Text {
            val start = follow(field)
            val count = u32(start)
            val data = checked(start.toLong() + 4, count + 1)
            require(u8(data + count.toInt()) == 0) { "Invalid LiteRT-LM string terminator." }
            return Text(data, String(bytes, data, count.toInt(), Charsets.UTF_8))
        }
        fun tables(field: Int?): List<Int> {
            if (field == null) return emptyList()
            val start = follow(field)
            val count = u32(start)
            require(count <= 4096) { "Too many LiteRT-LM metadata entries." }
            val data = checked(start.toLong() + 4, count * 4)
            return List(count.toInt()) { follow(data + it * 4) }
        }
    }
}
