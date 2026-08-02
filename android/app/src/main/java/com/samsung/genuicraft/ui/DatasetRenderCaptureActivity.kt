package com.samsung.genuicraft

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.onSizeChanged
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.time.Instant
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt
import androidx.compose.ui.unit.dp

class DatasetRenderCaptureActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "DatasetRenderCapture"

        const val EXTRA_INPUT_JSONL_PATH = "input_jsonl_path"
        const val EXTRA_OUTPUT_DIR = "output_dir"
        const val EXTRA_START_INDEX = "start_index"
        const val EXTRA_MAX_COUNT = "max_count"
        const val EXTRA_SETTLE_MS = "settle_ms"
        const val EXTRA_CAPTURE_FULL_HEIGHT = "capture_full_height"

        private const val DEFAULT_SETTLE_MS = 1200L
        private const val DEFAULT_READY_TIMEOUT_MS = 10_000L
        private const val POLL_INTERVAL_MS = 80L
    }

    private data class CaptureStats(
        val ok: Boolean,
        val fullHeightPx: Int,
        val tileCount: Int
    )

    private var records: List<GenUiRecord> = emptyList()
    private var currentIndex by mutableIntStateOf(-1)
    private var requestedScrollOffsetPx by mutableIntStateOf(0)

    @Volatile private var observedScrollOffsetPx: Int = 0
    @Volatile private var observedMaxScrollPx: Int = 0
    @Volatile private var observedViewportPx: Int = 0
    @Volatile private var observedMetricsReady: Boolean = false
    @Volatile private var observedCaptureLeftPx: Int = 0
    @Volatile private var observedCaptureTopPx: Int = 0
    @Volatile private var observedCaptureWidthPx: Int = 0
    @Volatile private var observedCaptureHeightPx: Int = 0
    @Volatile private var observedRenderedUiId: String? = null
    @Volatile private var observedNativeRenderReady: Boolean = false
    @Volatile private var observedNativeRenderOk: Boolean = false
    @Volatile private var observedNativeRenderError: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyOneUiWindowBlur()

        val inputPathRaw = intent.getStringExtra(EXTRA_INPUT_JSONL_PATH).orEmpty().trim()
        if (inputPathRaw.isBlank()) {
            finishWithError(null, "Missing required extra: $EXTRA_INPUT_JSONL_PATH")
            return
        }

        val inputFile = File(inputPathRaw)
        if (!inputFile.exists()) {
            finishWithError(null, "Input JSONL not found: ${inputFile.absolutePath}")
            return
        }

        val outputDir = File(
            intent.getStringExtra(EXTRA_OUTPUT_DIR).orEmpty().trim().ifBlank {
                File(getExternalFilesDir(null), "dataset_capture_output").absolutePath
            }
        )
        outputDir.mkdirs()

        val startIndex = intent.getIntExtra(EXTRA_START_INDEX, 0).coerceAtLeast(0)
        val maxCount = intent.getIntExtra(EXTRA_MAX_COUNT, -1)
        val settleMs = intent.getLongExtra(EXTRA_SETTLE_MS, DEFAULT_SETTLE_MS).coerceAtLeast(200L)
        val captureFullHeight = intent.getBooleanExtra(
            EXTRA_CAPTURE_FULL_HEIGHT,
            true
        )

        val rawJson = runCatching { inputFile.readText(Charsets.UTF_8) }.getOrElse { error ->
            finishWithError(outputDir, "Failed to read input JSONL: ${error.message}")
            return
        }
        records = GenUiRecordParser.parseRecords(
            raw = rawJson,
            sourceDir = inputFile.parentFile,
            sourceLabel = inputFile.name
        )
        if (records.isEmpty()) {
            finishWithError(outputDir, "No valid records parsed from: ${inputFile.absolutePath}")
            return
        }

        setContent {
            GenUiCraftTheme {
                DatasetCaptureRenderScreen(
                    record = records.getOrNull(currentIndex),
                    scrollRequestOffsetPx = requestedScrollOffsetPx,
                    onScrollMetrics = { currentOffset, maxOffset, viewportPx ->
                        observedScrollOffsetPx = currentOffset.coerceAtLeast(0)
                        observedMaxScrollPx = maxOffset.coerceAtLeast(0)
                        observedViewportPx = viewportPx.coerceAtLeast(0)
                        observedMetricsReady = true
                    },
                    onCaptureBounds = { left, top, width, height ->
                        observedCaptureLeftPx = left.coerceAtLeast(0)
                        observedCaptureTopPx = top.coerceAtLeast(0)
                        observedCaptureWidthPx = width.coerceAtLeast(0)
                        observedCaptureHeightPx = height.coerceAtLeast(0)
                    },
                    onRenderedRecord = { uiId ->
                        observedRenderedUiId = uiId
                    },
                    onNativeRenderStatus = { ok, error ->
                        observedNativeRenderReady = true
                        observedNativeRenderOk = ok
                        observedNativeRenderError = error
                    },
                    modifier = Modifier.fillMaxSize()
                )
            }
        }

        lifecycleScope.launch {
            runCapture(
                outputDir = outputDir,
                startIndex = startIndex,
                maxCount = maxCount,
                settleMs = settleMs,
                captureFullHeight = captureFullHeight
            )
        }
    }

    private suspend fun runCapture(
        outputDir: File,
        startIndex: Int,
        maxCount: Int,
        settleMs: Long,
        captureFullHeight: Boolean
    ) {
        val safeStart = startIndex.coerceIn(0, records.lastIndex)
        val endExclusive = if (maxCount <= 0) {
            records.size
        } else {
            (safeStart + maxCount).coerceAtMost(records.size)
        }
        if (safeStart >= endExclusive) {
            finishWithError(outputDir, "No records selected for capture.")
            return
        }

        val manifestFile = File(outputDir, "capture_manifest.jsonl")
        if (manifestFile.exists()) {
            manifestFile.delete()
        }
        val completeFile = File(outputDir, "_COMPLETE.json")
        val errorFile = File(outputDir, "_ERROR.txt")
        if (completeFile.exists()) completeFile.delete()
        if (errorFile.exists()) errorFile.delete()

        var captured = 0
        for (index in safeStart until endExclusive) {
            currentIndex = index
            requestedScrollOffsetPx = 0
            observedScrollOffsetPx = 0
            observedMaxScrollPx = 0
            observedViewportPx = 0
            observedMetricsReady = false
            observedRenderedUiId = null
            observedNativeRenderReady = false
            observedNativeRenderOk = false
            observedNativeRenderError = null
            val record = records[index]
            waitForContentReady(DEFAULT_READY_TIMEOUT_MS)
            waitForRenderedRecord(record.uiId, DEFAULT_READY_TIMEOUT_MS)
            waitForNativeRenderStatus(DEFAULT_READY_TIMEOUT_MS)
            waitForScrollMetricsReady(DEFAULT_READY_TIMEOUT_MS)
            waitForCaptureBoundsReady(DEFAULT_READY_TIMEOUT_MS)

            val fileName = buildFileName(index, record)
            val screenshotFile = File(outputDir, fileName)
            val viewportFile = File(
                outputDir,
                fileName.removeSuffix(".png") + "_viewport.png"
            )
            val viewportOk = captureViewportBitmap(viewportFile)
            val stats = if (captureFullHeight) {
                captureFullLengthBitmap(
                    target = screenshotFile,
                    settleMs = settleMs
                )
            } else {
                CaptureStats(ok = true, fullHeightPx = 0, tileCount = 0)
            }
            val candidateRenderOk = observedNativeRenderOk
            val overallOk =
                (!captureFullHeight || stats.ok) &&
                    viewportOk &&
                    candidateRenderOk
            if (overallOk) {
                captured += 1
                Log.i(
                    TAG,
                    "Captured record index=$index uiId=${record.uiId.orEmpty()} tiles=${stats.tileCount} height=${stats.fullHeightPx}px -> ${screenshotFile.name}"
                )
            } else {
                Log.w(TAG, "Capture failed index=$index uiId=${record.uiId.orEmpty()}")
            }

            appendManifestLine(
                file = manifestFile,
                index = index,
                record = record,
                screenshotName = (
                    if (captureFullHeight) screenshotFile.name else ""
                ),
                viewportName = viewportFile.name,
                ok = overallOk,
                viewportOk = viewportOk,
                fullHeightAttempted = captureFullHeight,
                nativeRenderOk = candidateRenderOk,
                nativeRenderError = observedNativeRenderError,
                fullHeightPx = stats.fullHeightPx,
                tileCount = stats.tileCount
            )
        }

        val completedJson = """
            {"status":"ok","captured":$captured,"requested":${endExclusive - safeStart},"start_index":$safeStart,"end_exclusive":$endExclusive,"completed_at":"${Instant.now()}"}
        """.trimIndent()
        completeFile.writeText(completedJson, Charsets.UTF_8)
        Log.i(TAG, "Capture completed captured=$captured requested=${endExclusive - safeStart} output=${outputDir.absolutePath}")
        finish()
    }

    private fun appendManifestLine(
        file: File,
        index: Int,
        record: GenUiRecord,
        screenshotName: String,
        viewportName: String,
        ok: Boolean,
        viewportOk: Boolean,
        fullHeightAttempted: Boolean,
        nativeRenderOk: Boolean,
        nativeRenderError: String?,
        fullHeightPx: Int,
        tileCount: Int
    ) {
        val escapedUiId = jsonEscape(record.uiId.orEmpty())
        val escapedQueryId = jsonEscape(record.queryId.orEmpty())
        val escapedResponseId = jsonEscape(record.responseId.orEmpty())
        val escapedScreenshot = jsonEscape(screenshotName)
        val escapedViewport = jsonEscape(viewportName)
        val escapedRenderError = jsonEscape(nativeRenderError.orEmpty())
        val line =
            """{"index":$index,"ui_id":"$escapedUiId","query_id":"$escapedQueryId","response_id":"$escapedResponseId","screenshot":"$escapedScreenshot","viewport_screenshot":"$escapedViewport","ok":$ok,"viewport_ok":$viewportOk,"full_height_attempted":$fullHeightAttempted,"native_render_ok":$nativeRenderOk,"native_render_error":"$escapedRenderError","full_height_px":$fullHeightPx,"tile_count":$tileCount}"""
        file.appendText("$line\n", Charsets.UTF_8)
    }

    private fun jsonEscape(raw: String): String =
        raw.replace("\\", "\\\\").replace("\"", "\\\"")

    private fun buildFileName(index: Int, record: GenUiRecord): String {
        val base = record.uiId?.takeIf { it.isNotBlank() } ?: "item_${index + 1}"
        val safe = base
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9._-]+"), "_")
            .trim('_')
            .ifBlank { "item_${index + 1}" }
        return "${(index + 1).toString().padStart(2, '0')}_${safe}.png"
    }

    private suspend fun waitForContentReady(timeoutMs: Long) {
        val root = window.decorView.rootView
        val startedAt = System.currentTimeMillis()
        while ((root.width <= 0 || root.height <= 0) && (System.currentTimeMillis() - startedAt) < timeoutMs) {
            delay(POLL_INTERVAL_MS)
        }
        // Give Compose one extra frame for final layout/paint stability.
        delay(120L)
    }

    private suspend fun waitForRenderedRecord(targetUiId: String?, timeoutMs: Long) {
        val normalizedTarget = targetUiId.orEmpty().trim()
        if (normalizedTarget.isBlank()) {
            delay(120L)
            return
        }
        val startedAt = System.currentTimeMillis()
        while ((System.currentTimeMillis() - startedAt) < timeoutMs) {
            if (observedRenderedUiId.orEmpty() == normalizedTarget) {
                delay(120L)
                return
            }
            delay(POLL_INTERVAL_MS)
        }
    }

    private suspend fun waitForScrollMetricsReady(timeoutMs: Long) {
        val startedAt = System.currentTimeMillis()
        while (
            (!observedMetricsReady || observedViewportPx <= 0) &&
            (System.currentTimeMillis() - startedAt) < timeoutMs
        ) {
            delay(POLL_INTERVAL_MS)
        }
        delay(120L)
    }

    private suspend fun waitForNativeRenderStatus(timeoutMs: Long) {
        val startedAt = System.currentTimeMillis()
        while (
            !observedNativeRenderReady &&
            (System.currentTimeMillis() - startedAt) < timeoutMs
        ) {
            delay(POLL_INTERVAL_MS)
        }
        delay(80L)
    }

    private suspend fun waitForCaptureBoundsReady(timeoutMs: Long) {
        val startedAt = System.currentTimeMillis()
        while (
            (observedCaptureWidthPx <= 0 || observedCaptureHeightPx <= 0) &&
            (System.currentTimeMillis() - startedAt) < timeoutMs
        ) {
            delay(POLL_INTERVAL_MS)
        }
        delay(80L)
    }

    private suspend fun waitForScrollPosition(targetOffset: Int, timeoutMs: Long = 4_000L) {
        val startedAt = System.currentTimeMillis()
        while ((System.currentTimeMillis() - startedAt) < timeoutMs) {
            if (abs(observedScrollOffsetPx - targetOffset) <= 2) {
                delay(90L)
                if (abs(observedScrollOffsetPx - targetOffset) <= 2) {
                    return
                }
            }
            delay(POLL_INTERVAL_MS)
        }
    }

    private fun buildCaptureOffsets(maxScroll: Int, viewport: Int): List<Int> {
        if (maxScroll <= 0 || viewport <= 0) {
            return listOf(0)
        }
        val step = (viewport * 0.88f).roundToInt().coerceAtLeast(1)
        val offsets = mutableListOf<Int>()
        var cursor = 0
        while (cursor < maxScroll) {
            offsets += cursor
            val next = (cursor + step).coerceAtMost(maxScroll)
            if (next == cursor) {
                break
            }
            cursor = next
        }
        if (offsets.lastOrNull() != maxScroll) {
            offsets += maxScroll
        }
        return offsets.distinct()
    }

    private suspend fun captureFullLengthBitmap(
        target: File,
        settleMs: Long
    ): CaptureStats {
        val root = window.decorView.rootView
        if (root.width <= 0 || root.height <= 0) {
            return CaptureStats(ok = false, fullHeightPx = 0, tileCount = 0)
        }

        val captureRect = Rect(
            0,
            observedCaptureTopPx.coerceIn(0, root.height),
            root.width,
            (observedCaptureTopPx + observedCaptureHeightPx).coerceIn(0, root.height)
        ).takeIf { it.width() > 0 && it.height() > 0 }
        val captureWidth = root.width
        val viewportHeight = observedViewportPx.takeIf { it > 0 }
            ?: captureRect?.height()
            ?: root.height
        val maxScroll = observedMaxScrollPx.coerceAtLeast(0)
        val totalHeight = (viewportHeight + maxScroll).coerceAtLeast(viewportHeight)
        val offsets = buildCaptureOffsets(maxScroll = maxScroll, viewport = viewportHeight)

        val stitched = runCatching {
            Bitmap.createBitmap(captureWidth, totalHeight, Bitmap.Config.ARGB_8888)
        }.getOrElse { error ->
            Log.e(TAG, "Failed to allocate stitched bitmap: ${error.message}", error)
            return CaptureStats(ok = false, fullHeightPx = 0, tileCount = 0)
        }
        val stitchedCanvas = Canvas(stitched)
        var stitchedBottom = 0

        var tilesDrawn = 0
        for (offset in offsets) {
            requestedScrollOffsetPx = offset
            waitForScrollPosition(offset)
            delay(settleMs)
            val actualOffset = observedScrollOffsetPx.coerceIn(0, maxScroll)
            val shot = captureWindowBitmapRaw(captureRect)
            if (shot == null) {
                stitched.recycle()
                return CaptureStats(ok = false, fullHeightPx = 0, tileCount = tilesDrawn)
            }
            val tileTop = actualOffset.coerceAtLeast(0)
            val srcTop = (stitchedBottom - tileTop).coerceIn(0, shot.height)
            var destTop = tileTop + srcTop
            if (destTop > stitchedBottom) {
                // Avoid transparent seam lines when scroll offsets report tiny forward jumps.
                destTop = stitchedBottom
            }
            val remainingHeight = totalHeight - destTop
            if (remainingHeight > 0 && srcTop < shot.height) {
                val drawHeight = minOf(shot.height - srcTop, remainingHeight)
                val srcRect = Rect(0, srcTop, shot.width, srcTop + drawHeight)
                val destRect = Rect(0, destTop, shot.width, destTop + drawHeight)
                stitchedCanvas.drawBitmap(shot, srcRect, destRect, null)
                stitchedBottom = maxOf(stitchedBottom, destTop + drawHeight)
            }
            shot.recycle()
            tilesDrawn += 1
        }

        return runCatching {
            target.parentFile?.mkdirs()
            val finalHeight = stitchedBottom.coerceIn(1, stitched.height)
            val finalBitmap = if (finalHeight == stitched.height) {
                stitched
            } else {
                Bitmap.createBitmap(stitched, 0, 0, stitched.width, finalHeight)
            }
            FileOutputStream(target).use { stream ->
                finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
                stream.flush()
            }
            if (finalBitmap !== stitched) {
                finalBitmap.recycle()
            }
            stitched.recycle()
            requestedScrollOffsetPx = 0
            CaptureStats(ok = true, fullHeightPx = finalHeight, tileCount = tilesDrawn)
        }.getOrElse { error ->
            stitched.recycle()
            Log.e(TAG, "Failed to persist stitched bitmap: ${error.message}", error)
            CaptureStats(ok = false, fullHeightPx = totalHeight, tileCount = tilesDrawn)
        }
    }

    private fun captureViewportBitmap(target: File): Boolean {
        val captureRect = Rect(
            0,
            observedCaptureTopPx.coerceAtLeast(0),
            window.decorView.rootView.width,
            (observedCaptureTopPx + observedCaptureHeightPx).coerceAtLeast(0)
        )
        val bitmap = captureWindowBitmapRaw(captureRect) ?: return false
        return runCatching {
            target.parentFile?.mkdirs()
            FileOutputStream(target).use { stream ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
                stream.flush()
            }
            bitmap.recycle()
            true
        }.getOrElse { error ->
            bitmap.recycle()
            Log.e(TAG, "Failed to persist viewport bitmap: ${error.message}", error)
            false
        }
    }

    private fun captureWindowBitmapRaw(captureRect: Rect?): Bitmap? {
        return runCatching {
            val root = window.decorView.rootView
            if (root.width <= 0 || root.height <= 0) {
                return null
            }
            val fullBitmap = Bitmap.createBitmap(root.width, root.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(fullBitmap)
            root.draw(canvas)
            val safeRect = captureRect
                ?.let { requested ->
                    Rect(
                        requested.left.coerceIn(0, root.width),
                        requested.top.coerceIn(0, root.height),
                        requested.right.coerceIn(0, root.width),
                        requested.bottom.coerceIn(0, root.height)
                    )
                }
                ?.takeIf { it.width() > 0 && it.height() > 0 }

            val viewport = if (safeRect == null || (safeRect.width() == root.width && safeRect.height() == root.height && safeRect.left == 0 && safeRect.top == 0)) {
                fullBitmap
            } else {
                val cropped = Bitmap.createBitmap(
                    fullBitmap,
                    safeRect.left,
                    safeRect.top,
                    safeRect.width(),
                    safeRect.height()
                )
                fullBitmap.recycle()
                cropped
            }
            trimUniformBlackBands(viewport)
        }.getOrElse { error ->
            Log.e(TAG, "Failed to capture viewport bitmap: ${error.message}", error)
            null
        }
    }

    private fun trimUniformBlackBands(bitmap: Bitmap): Bitmap {
        if (bitmap.width <= 0 || bitmap.height <= 0) {
            return bitmap
        }
        val width = bitmap.width
        val height = bitmap.height
        val maxTrim = minOf(220, height / 3)
        if (maxTrim <= 0) {
            return bitmap
        }

        val rowPixels = IntArray(width)
        val darkThreshold = (width * 0.99f).roundToInt().coerceAtLeast(1)

        fun isMostlyBlackRow(y: Int): Boolean {
            bitmap.getPixels(rowPixels, 0, width, 0, y, width, 1)
            var darkPixels = 0
            rowPixels.forEach { pixel ->
                val alpha = (pixel ushr 24) and 0xFF
                if (alpha < 10) {
                    darkPixels += 1
                    return@forEach
                }
                val red = (pixel ushr 16) and 0xFF
                val green = (pixel ushr 8) and 0xFF
                val blue = pixel and 0xFF
                if (red <= 12 && green <= 12 && blue <= 12) {
                    darkPixels += 1
                }
            }
            return darkPixels >= darkThreshold
        }

        var topTrim = 0
        while (topTrim < maxTrim && topTrim < height && isMostlyBlackRow(topTrim)) {
            topTrim += 1
        }

        var bottomTrim = 0
        while (bottomTrim < maxTrim && (topTrim + bottomTrim) < height && isMostlyBlackRow(height - 1 - bottomTrim)) {
            bottomTrim += 1
        }

        if (topTrim == 0 && bottomTrim == 0) {
            return bitmap
        }

        val targetHeight = height - topTrim - bottomTrim
        if (targetHeight <= 0) {
            return bitmap
        }
        val trimmed = Bitmap.createBitmap(bitmap, 0, topTrim, width, targetHeight)
        bitmap.recycle()
        return trimmed
    }

    private fun finishWithError(outputDir: File?, message: String) {
        Log.e(TAG, message)
        outputDir?.let {
            runCatching {
                it.mkdirs()
                File(it, "_ERROR.txt").writeText(message, Charsets.UTF_8)
            }
        }
        finish()
    }
}

@Composable
private fun DatasetCaptureRenderScreen(
    record: GenUiRecord?,
    scrollRequestOffsetPx: Int,
    onScrollMetrics: (currentOffset: Int, maxOffset: Int, viewportPx: Int) -> Unit,
    onCaptureBounds: (leftPx: Int, topPx: Int, widthPx: Int, heightPx: Int) -> Unit,
    onRenderedRecord: (uiId: String?) -> Unit,
    onNativeRenderStatus: (ok: Boolean, error: String?) -> Unit,
    modifier: Modifier = Modifier
) {
    val result = remember(record?.rawJson, record?.sourceDir) {
        record?.let { GenUiNativeRenderer.renderLegacyForComparison(rawInput = it.rawJson, sourceDir = it.sourceDir) }
    }
    val scrollState = rememberScrollState()
    val deviceConfig = rememberDeviceUiConfig()
    val horizontalPadding = when (deviceConfig.widthClass) {
        DeviceSizeClass.Compact -> 12.dp
        DeviceSizeClass.Medium -> 18.dp
        DeviceSizeClass.Expanded -> 24.dp
    }
    LaunchedEffect(result, record?.uiId) {
        onNativeRenderStatus(
            result != null && result.errorMessage == null && result.surfaces.isNotEmpty(),
            result?.errorMessage
        )
    }

    GenUiScreenBackground(modifier = modifier.fillMaxSize()) { backgroundModifier ->
        BoxWithConstraints(
            modifier = backgroundModifier.onSizeChanged { size ->
                onScrollMetrics(scrollState.value, scrollState.maxValue, size.height)
            }
        ) {
            if (result == null) {
                Box(modifier = Modifier.fillMaxSize())
                return@BoxWithConstraints
            }

            LaunchedEffect(scrollRequestOffsetPx, record?.uiId, record?.responseId) {
                val clamped = scrollRequestOffsetPx.coerceIn(0, scrollState.maxValue)
                if (scrollState.value != clamped) {
                    scrollState.scrollTo(clamped)
                }
                onScrollMetrics(scrollState.value, scrollState.maxValue, constraints.maxHeight)
            }
            LaunchedEffect(scrollState.value, scrollState.maxValue) {
                onScrollMetrics(scrollState.value, scrollState.maxValue, constraints.maxHeight)
            }

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(horizontal = horizontalPadding, vertical = 14.dp)
                    .onGloballyPositioned { coordinates ->
                        val bounds = coordinates.boundsInWindow()
                        onCaptureBounds(
                            bounds.left.roundToInt(),
                            bounds.top.roundToInt(),
                            bounds.width.roundToInt(),
                            bounds.height.roundToInt()
                        )
                        onRenderedRecord(record?.uiId)
                        onScrollMetrics(
                            scrollState.value,
                            scrollState.maxValue,
                            bounds.height.roundToInt().coerceAtLeast(1)
                        )
                    }
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState)
                ) {
                    GenUiNativeRenderer.RenderInline(
                        result = result,
                        sourceDir = record?.sourceDir,
                        onOpenExternalUrl = {},
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
}
