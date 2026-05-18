package com.samsung.genuicraft.inference

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

object OnDeviceModelDownloader {
    data class Progress(
        val downloadedBytes: Long,
        val totalBytes: Long?
    ) {
        val fraction: Float?
            get() = totalBytes?.takeIf { it > 0L }?.let { (downloadedBytes.toDouble() / it.toDouble()).coerceIn(0.0, 1.0).toFloat() }
    }

    fun download(
        context: Context,
        entry: OnDeviceModelCatalog.Entry,
        onProgress: (Progress) -> Unit
    ): File {
        val target = entry.localFile(context)
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, "${entry.fileName}.part")
        if (temp.exists()) temp.delete()

        val connection = (URL(entry.downloadUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 15000
            readTimeout = 120000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "GenUICraft/1.1 Android")
        }
        try {
            val code = connection.responseCode
            if (code !in 200..299) {
                val error = connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                throw IllegalStateException("Download failed: HTTP $code ${error.take(160)}")
            }
            val total = connection.contentLengthLong.takeIf { it > 0L }
            var downloaded = 0L
            connection.inputStream.use { input ->
                temp.outputStream().use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        downloaded += read.toLong()
                        onProgress(Progress(downloaded, total))
                    }
                }
            }
            if (target.exists()) target.delete()
            if (!temp.renameTo(target)) {
                temp.copyTo(target, overwrite = true)
                temp.delete()
            }
            return target
        } catch (t: Throwable) {
            temp.delete()
            throw t
        } finally {
            connection.disconnect()
        }
    }
}
