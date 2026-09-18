package com.samsung.genuicraft.inference

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Explicitly started OS-managed download; only a verified package is exposed as Ready. */
class SdkGemmaModelDownload(context: Context) {
    sealed interface State {
        data object Idle : State
        data class Downloading(val downloadedBytes: Long, val totalBytes: Long, val paused: Boolean = false) : State
        data object Verifying : State
        data class Ready(val file: File) : State
        data class Error(val message: String) : State
    }

    private val controller = sharedController(context.applicationContext)
    val modelFile: File get() = controller.modelFile
    val state: StateFlow<State> get() = controller.state
    suspend fun inspect(): State = controller.inspect()
    suspend fun start(): State = controller.start()
    suspend fun cancel(): State = controller.cancel()

    companion object {
        const val MODEL_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it.litertlm"
        const val MODEL_BYTES = 2_588_147_712L
        const val MODEL_SHA256 = "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
        private var shared: SdkGemmaDownloadController? = null

        @Synchronized private fun sharedController(context: Context): SdkGemmaDownloadController {
            shared?.let { return it }
            val directory = File(requireNotNull(context.getExternalFilesDir(null)) {
                "App model storage is unavailable."
            }, "sdk_models/managed")
            val manager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            val preferences = context.getSharedPreferences("sdk_gemma_managed_download", Context.MODE_PRIVATE)
            val store = object : SdkGemmaDownloadStore {
                override fun readId(): Long? = preferences.getLong("download_id", -1L).takeIf { it >= 0L }
                override fun writeId(id: Long?) {
                    val edit = preferences.edit()
                    if (id == null) edit.remove("download_id") else edit.putLong("download_id", id)
                    check(edit.commit()) { "Could not save download recovery state." }
                }
            }
            val transfer = object : SdkGemmaDownloadTransfer {
                override fun enqueue(destination: File): Long = manager.enqueue(
                    DownloadManager.Request(Uri.parse(MODEL_URL))
                        .setTitle("Gemma 4 E2B model")
                        .setDescription("GenUICraft model · 2.59 GB")
                        .setAllowedOverMetered(true)
                        .setAllowedOverRoaming(false)
                        .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                        .setDestinationUri(Uri.fromFile(destination)),
                )
                override fun remove(id: Long) { manager.remove(id) }
                override fun query(id: Long): SdkGemmaTransferSnapshot? =
                    manager.query(DownloadManager.Query().setFilterById(id))?.use { cursor ->
                        if (!cursor.moveToFirst()) return@use null
                        fun long(column: String) = cursor.getLong(cursor.getColumnIndexOrThrow(column))
                        val status = long(DownloadManager.COLUMN_STATUS).toInt()
                        val reason = long(DownloadManager.COLUMN_REASON).toInt()
                        SdkGemmaTransferSnapshot(
                            status = when (status) {
                                DownloadManager.STATUS_SUCCESSFUL -> SdkGemmaTransferStatus.COMPLETE
                                DownloadManager.STATUS_FAILED -> SdkGemmaTransferStatus.FAILED
                                DownloadManager.STATUS_PAUSED -> SdkGemmaTransferStatus.PAUSED
                                else -> SdkGemmaTransferStatus.RUNNING
                            },
                            downloadedBytes = long(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR),
                            totalBytes = long(DownloadManager.COLUMN_TOTAL_SIZE_BYTES),
                            reason = "Android download failed (reason $reason). Check connection and storage, then retry.",
                        )
                    }
            }
            return SdkGemmaDownloadController(directory, transfer, store, MODEL_BYTES, MODEL_SHA256)
                .also { shared = it }
        }
    }
}

internal enum class SdkGemmaTransferStatus { RUNNING, PAUSED, COMPLETE, FAILED }
internal data class SdkGemmaTransferSnapshot(
    val status: SdkGemmaTransferStatus,
    val downloadedBytes: Long,
    val totalBytes: Long,
    val reason: String = "Download failed. Retry to start again.",
)
internal interface SdkGemmaDownloadTransfer {
    fun enqueue(destination: File): Long
    fun query(id: Long): SdkGemmaTransferSnapshot?
    fun remove(id: Long)
}
internal interface SdkGemmaDownloadStore {
    fun readId(): Long?
    fun writeId(id: Long?)
}

/** Injectable core keeps recovery, verification, and cancellation testable without Android. */
internal class SdkGemmaDownloadController(
    directory: File,
    private val transfer: SdkGemmaDownloadTransfer,
    private val store: SdkGemmaDownloadStore,
    private val expectedBytes: Long,
    private val expectedSha256: String,
    private val verify: suspend (File, Long, String) -> Unit = ::verifySdkGemmaModel,
) {
    val modelFile = File(directory, "gemma-4-E2B-it.litertlm")
    private val partialFile = File(directory, "gemma-4-E2B-it.litertlm.download")
    private val mutableState = MutableStateFlow<SdkGemmaModelDownload.State>(SdkGemmaModelDownload.State.Idle)
    val state = mutableState.asStateFlow()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mutex = Mutex()
    private var verification: Job? = null
    private var epoch = 0L
    private var verifiedModified = -1L

    suspend fun inspect(): SdkGemmaModelDownload.State = withContext(Dispatchers.IO) {
        mutex.withLock { safely { inspectLocked() } }
    }

    suspend fun start(): SdkGemmaModelDownload.State = withContext(Dispatchers.IO) {
        mutex.withLock {
            safely {
                val current = inspectLocked()
                if (current is SdkGemmaModelDownload.State.Ready || current is SdkGemmaModelDownload.State.Downloading ||
                    current is SdkGemmaModelDownload.State.Verifying) return@safely current
                check(modelFile.parentFile!!.isDirectory || modelFile.parentFile!!.mkdirs()) {
                    "Could not create managed model storage."
                }
                store.readId()?.let(transfer::remove)
                store.writeId(null)
                check(!partialFile.exists() || partialFile.delete()) { "Could not clear the previous partial model." }
                val id = transfer.enqueue(partialFile)
                try { store.writeId(id) } catch (failure: Exception) { transfer.remove(id); throw failure }
                update(SdkGemmaModelDownload.State.Downloading(0L, expectedBytes))
            }
        }
    }

    suspend fun cancel(): SdkGemmaModelDownload.State = withContext(Dispatchers.IO) {
        mutex.withLock {
            safely {
                epoch++
                verification?.cancel()
                verification = null
                store.readId()?.let(transfer::remove)
                store.writeId(null)
                check(!partialFile.exists() || partialFile.delete()) { "Could not remove the partial download." }
                // The verified managed model and all manually staged files are retained.
                val current = mutableState.value
                update(if (current is SdkGemmaModelDownload.State.Ready) current else SdkGemmaModelDownload.State.Idle)
            }
        }
    }

    private fun inspectLocked(): SdkGemmaModelDownload.State {
        val current = mutableState.value
        if (current is SdkGemmaModelDownload.State.Ready && modelFile.isFile &&
            modelFile.length() == expectedBytes && modelFile.lastModified() == verifiedModified) return current
        if (verification?.isActive == true) return update(SdkGemmaModelDownload.State.Verifying)
        if (modelFile.isFile) return beginVerification(modelFile)
        val id = store.readId() ?: return if (current is SdkGemmaModelDownload.State.Error) current
            else update(SdkGemmaModelDownload.State.Idle)
        val snapshot = transfer.query(id) ?: run {
            store.writeId(null)
            return update(SdkGemmaModelDownload.State.Error("The OS download record is unavailable. Retry the download."))
        }
        return when (snapshot.status) {
            SdkGemmaTransferStatus.COMPLETE -> beginVerification(partialFile)
            SdkGemmaTransferStatus.FAILED -> update(SdkGemmaModelDownload.State.Error(snapshot.reason))
            else -> update(SdkGemmaModelDownload.State.Downloading(
                snapshot.downloadedBytes.coerceAtLeast(0L),
                snapshot.totalBytes.takeIf { it > 0L } ?: expectedBytes,
                snapshot.status == SdkGemmaTransferStatus.PAUSED,
            ))
        }
    }

    private fun beginVerification(file: File): SdkGemmaModelDownload.State {
        val generation = ++epoch
        update(SdkGemmaModelDownload.State.Verifying)
        verification = scope.launch {
            var bytesVerified = false
            try {
                verify(file, expectedBytes, expectedSha256)
                bytesVerified = true
                mutex.withLock {
                    if (generation != epoch) return@withLock
                    currentCoroutineContext().ensureActive()
                    if (file != modelFile) {
                        check(!modelFile.exists()) { "Managed model destination already exists." }
                        check(file.renameTo(modelFile)) { "Verified model could not be moved into place." }
                    }
                    store.readId()?.let(transfer::remove)
                    store.writeId(null)
                    verifiedModified = modelFile.lastModified()
                    verification = null
                    update(SdkGemmaModelDownload.State.Ready(modelFile))
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Exception) {
                mutex.withLock {
                    if (generation == epoch) {
                        if (!bytesVerified) {
                            runCatching { store.readId()?.let(transfer::remove); store.writeId(null) }
                            file.delete() // Only the backend-owned managed file or its staging file.
                        }
                        verification = null
                        update(SdkGemmaModelDownload.State.Error(failure.message ?: "Model verification failed. Retry."))
                    }
                }
            }
        }
        return SdkGemmaModelDownload.State.Verifying
    }

    private inline fun safely(block: () -> SdkGemmaModelDownload.State): SdkGemmaModelDownload.State =
        try { block() } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            update(SdkGemmaModelDownload.State.Error(failure.message ?: "Model download operation failed."))
        }

    private fun update(value: SdkGemmaModelDownload.State): SdkGemmaModelDownload.State {
        mutableState.value = value
        return value
    }
}

internal suspend fun verifySdkGemmaModel(file: File, expectedBytes: Long, expectedSha256: String) {
    check(file.isFile && file.length() == expectedBytes) {
        "Model size mismatch: expected $expectedBytes bytes, found ${if (file.isFile) file.length() else 0L}. Retry."
    }
    val digest = MessageDigest.getInstance("SHA-256")
    var bytesRead = 0L
    file.inputStream().buffered(1024 * 1024).use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            currentCoroutineContext().ensureActive()
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
            bytesRead += read
        }
    }
    val actual = digest.digest().joinToString("") { "%02x".format(it) }
    check(bytesRead == expectedBytes && file.length() == expectedBytes && actual.equals(expectedSha256, true)) {
        "Model SHA-256 verification failed. The downloaded file was not accepted; retry."
    }
}
