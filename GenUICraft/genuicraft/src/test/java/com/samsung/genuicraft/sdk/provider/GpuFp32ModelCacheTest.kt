package com.samsung.genuicraft.sdk.provider

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Properties
import java.util.concurrent.CancellationException
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GpuFp32ModelCacheTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test fun changesOnlyTextPreferenceAndPreservesEveryPayloadByte() {
        val bytes = fixture()
        val source = temporary.newFile("trained.litertlm").apply { writeBytes(bytes) }
        val prepared = GpuFp32ModelCache.prepare(source)
        assertNotEquals(source.canonicalFile, prepared)
        assertArrayEquals(bytes, source.readBytes())
        val changed = prepared.readBytes()
        val differences = bytes.indices.filter { bytes[it] != changed[it] }
        assertEquals(2, differences.size) // only the final two characters of fp16 -> fp32
        assertEquals("16", String(differences.map { bytes[it] }.toByteArray()))
        assertEquals("32", String(differences.map { changed[it] }.toByteArray()))
        assertArrayEquals(bytes.copyOfRange(2048, bytes.size), changed.copyOfRange(2048, changed.size))
        assertArrayEquals(LiteRtPrecisionHeader.read(prepared), LiteRtPrecisionHeader.fp32(LiteRtPrecisionHeader.read(prepared), changed.size.toLong()))
        val record = Properties().apply { File(prepared.parentFile, "manifest.properties").inputStream().use(::load) }
        assertNotEquals(record.getProperty("sourceSha256"), record.getProperty("preparedSha256"))
        assertEquals("fp32", record.getProperty("textActivationType"))
    }

    @Test fun reusesCompleteCacheAndRebuildsTruncatedCopies() {
        val source = temporary.newFile("trained.litertlm").apply { writeBytes(fixture()) }
        val prepared = GpuFp32ModelCache.prepare(source)
        val original = prepared.readBytes()
        val modified = prepared.lastModified()
        assertEquals(prepared, GpuFp32ModelCache.prepare(source))
        assertEquals(modified, prepared.lastModified())
        prepared.writeBytes(byteArrayOf(1))
        assertEquals(prepared, GpuFp32ModelCache.prepare(source))
        assertArrayEquals(original, prepared.readBytes())
    }

    @Test fun newSourceRevisionCannotReuseOldCache() {
        val source = temporary.newFile("trained.litertlm").apply { writeBytes(fixture()) }
        val first = GpuFp32ModelCache.prepare(source)
        val oldModified = source.lastModified()
        val changed = source.readBytes().also { it[it.lastIndex] = 91 }
        source.writeBytes(changed)
        assertTrue(source.setLastModified(oldModified + 2000))
        val second = GpuFp32ModelCache.prepare(source)
        assertNotEquals(first, second)
        assertEquals(91.toByte(), second.readBytes().last())
    }

    @Test fun recoversCopyInterruptedBeforeManifestWasPublished() {
        val source = temporary.newFile("trained.litertlm").apply { writeBytes(fixture()) }
        val prepared = GpuFp32ModelCache.prepare(source)
        val expected = prepared.readBytes()
        assertTrue(File(prepared.parentFile, "manifest.properties").delete())
        val interrupted = File(prepared.parentFile, "prepare-abandoned.tmp").apply { writeBytes(expected) }
        val unrelated = File(prepared.parentFile, "keep.tmp").apply { writeText("keep") }
        assertEquals(prepared, GpuFp32ModelCache.prepare(source))
        assertArrayEquals(expected, prepared.readBytes())
        assertFalse(interrupted.exists())
        assertEquals("keep", unrelated.readText())
        assertTrue(File(prepared.parentFile, "manifest.properties").isFile)
    }

    @Test fun existingFp32NeedsNoCopy() {
        val source = temporary.newFile("ready.litertlm").apply { writeBytes(fixture(precision = "fp32")) }
        assertEquals(source.canonicalFile, GpuFp32ModelCache.prepare(source))
        assertFalse(File(source.parentFile, "genuicraft-gpu-fp32").exists())
    }

    @Test fun cancellationNeverPublishesAnIncompleteModel() {
        val source = temporary.newFile("trained.litertlm").apply { writeBytes(fixture()) }
        var checks = 0
        assertThrows(CancellationException::class.java) {
            GpuFp32ModelCache.prepare(source) {
                if (++checks == 3) throw CancellationException("cancel")
            }
        }
        assertEquals(listOf(source), temporary.root.walkTopDown().filter { it.extension == "litertlm" }.toList())
        assertFalse(temporary.root.walkTopDown().any { it.extension == "tmp" })
        assertTrue(GpuFp32ModelCache.prepare(source).isFile)
    }

    @Test fun rejectsAliasedMetadataInsteadOfChangingVisionPrecision() {
        assertThrows(IllegalArgumentException::class.java) { patch(fixture(aliasStrings = true)) }
    }

    @Test fun rejectsMissingTextPreferenceAndInvalidBounds() {
        assertThrows(IllegalArgumentException::class.java) { patch(fixture(precisionKey = "another_setting")) }
        assertThrows(IllegalArgumentException::class.java) { patch(fixture(precision = "fp64")) }
        assertThrows(IllegalArgumentException::class.java) { patch(fixture().also { it[0] = 0 }) }
        assertThrows(IllegalArgumentException::class.java) { patch(fixture().also { it[8] = 2 }) }
        assertThrows(IllegalArgumentException::class.java) {
            patch(fixture().also { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).putInt(32, Int.MAX_VALUE) })
        }
        assertThrows(IllegalArgumentException::class.java) {
            patch(fixture().also { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).putLong(24, Long.MAX_VALUE) })
        }
        assertThrows(IllegalArgumentException::class.java) {
            val bytes = fixture()
            LiteRtPrecisionHeader.fp32(bytes.copyOf(2048), 2050)
        }
    }

    private fun patch(bytes: ByteArray) = LiteRtPrecisionHeader.fp32(bytes.copyOf(2048), bytes.size.toLong())

    /** Synthetic FlatBuffer metadata with text, vision and drafter sections and opaque payloads. */
    private fun fixture(precision: String = "fp16", precisionKey: String = "prefer_activation_type", aliasStrings: Boolean = false): ByteArray {
        val data = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)
        data.put("LITERTLM".toByteArray())
        data.putInt(8, 1)
        data.putLong(24, 2048)
        var cursor = 36
        val strings = mutableListOf<Pair<Int, String>>()
        fun allocate(size: Int, alignment: Int = 4): Int {
            cursor = (cursor + alignment - 1) / alignment * alignment
            return cursor.also { cursor += size }
        }
        fun reference(field: Int, target: Int) { data.putInt(field, target - field) }
        fun table(vararg widths: Int): Pair<Int, List<Int>> {
            var size = 4
            val distances = widths.map { width ->
                size = (size + width - 1) / width * width
                size.also { size += width }
            }
            val vtable = allocate(4 + widths.size * 2, 2)
            val target = allocate(size, 8)
            data.putShort(vtable, (4 + widths.size * 2).toShort())
            data.putShort(vtable + 2, size.toShort())
            distances.forEachIndexed { index, distance -> data.putShort(vtable + 4 + index * 2, distance.toShort()) }
            data.putInt(target, target - vtable)
            return target to distances.map { target + it }
        }
        val (root, rootFields) = table(4, 4)
        reference(32, root)
        // Absent optional system metadata.
        val rootVtable = root - data.getInt(root)
        data.putShort(rootVtable + 4, 0)
        val (metadata, metadataFields) = table(4)
        reference(rootFields[1], metadata)
        val sections = allocate(16)
        reference(metadataFields[0], sections)
        data.putInt(sections, 3)
        listOf("tf_lite_prefill_decode", "tf_lite_vision_encoder", "tf_lite_mtp_drafter").forEachIndexed { index, type ->
            val (section, fields) = table(4, 8, 8, 1)
            reference(sections + 4 + index * 4, section)
            data.putLong(fields[1], 2048L + index * 512)
            data.putLong(fields[2], 2048L + (index + 1) * 512)
            data.put(fields[3], 3)
            val values = mutableListOf("model_type" to type)
            if (index < 2) values += (if (index == 0) precisionKey else "prefer_activation_type") to (if (index == 0) precision else "fp16")
            val entries = allocate(4 + values.size * 4)
            reference(fields[0], entries)
            data.putInt(entries, values.size)
            values.forEachIndexed { itemIndex, (key, value) ->
                val (kv, kvFields) = table(4, 1, 4)
                reference(entries + 4 + itemIndex * 4, kv)
                strings += kvFields[0] to key
                data.put(kvFields[1], 9)
                val (union, unionFields) = table(4)
                reference(kvFields[2], union)
                strings += unionFields[0] to value
            }
        }
        val aliases = mutableMapOf<String, Int>()
        strings.forEach { (field, value) ->
            val target = if (aliasStrings && aliases.containsKey(value)) aliases.getValue(value) else {
                allocate(4 + value.length + 1).also {
                    data.putInt(it, value.length)
                    value.toByteArray().copyInto(data.array(), it + 4)
                    aliases[value] = it
                }
            }
            reference(field, target)
        }
        check(cursor < 2048)
        for (index in 2048 until 4096) data.put(index, (index % 251).toByte())
        return data.array()
    }
}
