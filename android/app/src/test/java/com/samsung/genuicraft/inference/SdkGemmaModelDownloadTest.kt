package com.samsung.genuicraft.inference

import com.samsung.genuicraft.inference.SdkGemmaModelDownload.State
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SdkGemmaModelDownloadTest {
    @get:Rule val temporary = TemporaryFolder()
    private val bytes = "verified model fixture".toByteArray()
    private val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Test fun `completed download is verified before promotion`() = runBlocking {
        val transfer = Transfer()
        val store = Store()
        val controller = controller(transfer, store)
        assertTrue(controller.start() is State.Downloading)
        transfer.destination!!.writeBytes(bytes)
        transfer.snapshot = completed()
        assertEquals(State.Verifying, controller.inspect())
        assertTrue(terminal(controller) is State.Ready)
        assertArrayEquals(bytes, controller.modelFile.readBytes())
        assertFalse(transfer.destination!!.exists())
        assertNull(store.id)
    }

    @Test fun `wrong size and wrong hash cannot become ready`() = runBlocking {
        listOf(bytes.copyOf(bytes.size - 1), ByteArray(bytes.size) { 7 }).forEach { bad ->
            val transfer = Transfer()
            val controller = controller(transfer, Store())
            controller.start()
            transfer.destination!!.writeBytes(bad)
            transfer.snapshot = completed()
            assertEquals(State.Verifying, controller.inspect())
            assertTrue(terminal(controller) is State.Error)
            assertFalse(controller.modelFile.exists())
            assertFalse(transfer.destination!!.exists())
        }
    }

    @Test fun `existing unmarked managed file is verified without enqueue or manual file mutation`() = runBlocking {
        val directory = temporary.newFolder()
        val manual = File(directory, "../manually-staged.litertlm").apply { writeText("keep manual model") }
        val transfer = Transfer()
        val controller = controller(transfer, Store(), directory)
        controller.modelFile.writeBytes(bytes)
        assertEquals(State.Verifying, controller.inspect())
        assertTrue(terminal(controller) is State.Ready)
        assertEquals(0, transfer.enqueues)
        assertEquals("keep manual model", manual.readText())
        assertTrue(controller.cancel() is State.Ready)
        assertArrayEquals(bytes, controller.modelFile.readBytes())
    }

    @Test fun `persisted OS transfer recovers progress above two GiB without restarting`() = runBlocking {
        val transfer = Transfer().apply {
            snapshot = SdkGemmaTransferSnapshot(SdkGemmaTransferStatus.PAUSED, 2_400_000_000L, 2_588_147_712L)
        }
        val store = Store(41L)
        val directory = temporary.newFolder()
        val first = controller(transfer, store, directory)
        val second = controller(transfer, store, directory)
        val expected = State.Downloading(2_400_000_000L, 2_588_147_712L, paused = true)
        assertEquals(expected, first.inspect())
        assertEquals(expected, second.inspect())
        assertEquals(expected, second.start())
        assertEquals(0, transfer.enqueues)
        assertEquals(41L, store.id)
    }

    @Test fun `bookkeeping failure preserves an existing verified model`() = runBlocking {
        val store = object : SdkGemmaDownloadStore {
            override fun readId(): Long? = null
            override fun writeId(id: Long?) { error("Storage temporarily unavailable") }
        }
        val controller = SdkGemmaDownloadController(
            temporary.newFolder(), Transfer(), store, bytes.size.toLong(), sha,
        )
        controller.modelFile.writeBytes(bytes)
        assertEquals(State.Verifying, controller.inspect())
        assertTrue(terminal(controller) is State.Error)
        assertArrayEquals(bytes, controller.modelFile.readBytes())
    }

    @Test fun `cancel racing verification never promotes the partial file`() = runBlocking {
        val entered = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val finished = CompletableDeferred<Unit>()
        val transfer = Transfer()
        val store = Store()
        val directory = temporary.newFolder()
        val controller = SdkGemmaDownloadController(directory, transfer, store, bytes.size.toLong(), sha) { _, _, _ ->
            withContext(NonCancellable) {
                entered.complete(Unit)
                release.await()
                finished.complete(Unit)
            }
        }
        controller.start()
        transfer.destination!!.writeBytes(bytes)
        transfer.snapshot = completed()
        controller.inspect()
        withTimeout(2_000) { entered.await() }
        assertEquals(State.Idle, controller.cancel())
        release.complete(Unit)
        withTimeout(2_000) { finished.await() }
        assertEquals(State.Idle, controller.inspect())
        assertFalse(controller.modelFile.exists())
        assertFalse(transfer.destination!!.exists())
        assertNull(store.id)
    }

    @Test fun `failed or missing OS download requires explicit retry`() = runBlocking {
        val transfer = Transfer().apply { snapshot = null }
        val store = Store(41L)
        val controller = controller(transfer, store)
        assertTrue(controller.inspect() is State.Error)
        assertEquals(0, transfer.enqueues)
        assertTrue(controller.start() is State.Downloading)
        assertEquals(1, transfer.enqueues)
        transfer.snapshot = SdkGemmaTransferSnapshot(SdkGemmaTransferStatus.FAILED, 5L, 20L, "No space")
        assertEquals(State.Error("No space"), controller.inspect())
        assertEquals(1, transfer.enqueues)
        assertTrue(controller.start() is State.Downloading)
        assertEquals(2, transfer.enqueues)
        assertTrue(transfer.removed.contains(1L))
    }

    private fun controller(transfer: Transfer, store: Store, directory: File = temporary.newFolder()) =
        SdkGemmaDownloadController(directory, transfer, store, bytes.size.toLong(), sha)

    private fun completed() = SdkGemmaTransferSnapshot(SdkGemmaTransferStatus.COMPLETE, bytes.size.toLong(), bytes.size.toLong())
    private suspend fun terminal(controller: SdkGemmaDownloadController): State = withTimeout(2_000) {
        controller.state.first { it is State.Ready || it is State.Error }
    }

    private class Store(var id: Long? = null) : SdkGemmaDownloadStore {
        override fun readId() = id
        override fun writeId(id: Long?) { this.id = id }
    }

    private class Transfer : SdkGemmaDownloadTransfer {
        var enqueues = 0
        var destination: File? = null
        var snapshot: SdkGemmaTransferSnapshot? = SdkGemmaTransferSnapshot(SdkGemmaTransferStatus.RUNNING, 0, -1)
        val removed = mutableListOf<Long>()
        override fun enqueue(destination: File): Long {
            this.destination = destination
            enqueues++
            snapshot = SdkGemmaTransferSnapshot(SdkGemmaTransferStatus.RUNNING, 0, -1)
            return enqueues.toLong()
        }
        override fun query(id: Long) = snapshot
        override fun remove(id: Long) { removed += id }
    }
}
