package com.samsung.genuicraft.sdk

import android.content.Context
import android.content.res.Configuration
import android.graphics.Color
import android.util.AttributeSet
import android.view.View
import android.widget.FrameLayout
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.widget.NestedScrollView

/** Android View wrapper around [GenUiContent]. */
class GenUiView(
    context: Context,
    attrs: AttributeSet? = null,
) : FrameLayout(context, attrs) {
    private val composeView = ComposeView(context).apply {
        setViewCompositionStrategy(
            ViewCompositionStrategy.DisposeOnDetachedFromWindowOrReleasedFromPool
        )
    }
    private val scrollView = NestedScrollView(context).apply {
        isFillViewport = true
        addView(
            composeView,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
        )
    }

    var onAction: (GenUiAction) -> Unit = {}
    private var embeddedModeState by mutableStateOf(false)
    /** Render at natural content height for a host-owned scrolling viewport. Set before [render]. */
    var embeddedMode: Boolean
        get() = embeddedModeState
        set(value) {
            if (embeddedModeState == value) return
            embeddedModeState = value
            invalidateContentSizeReport()
            previewRevision++
            previewScrollPosition = null
            if (value) {
                scrollView.removeView(composeView)
                removeView(scrollView)
                addView(composeView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
            } else {
                removeView(composeView)
                scrollView.addView(
                    composeView,
                    LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
                )
                addView(scrollView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
            }
            updateBackgroundColor()
            requestLayout()
        }

    /** Natural rendered size in pixels. Posted after layout; never called for cleared or detached content. */
    var onContentSizeChanged: (widthPx: Int, heightPx: Int) -> Unit = { _, _ -> }

    private var previewScrollPosition: Pair<Int, Int>? = null
    private var previewRevision = 0L
    private var sizeRevision = 0L
    private var hasRenderedContent = false
    private var pendingContentSize: Pair<Int, Int>? = null
    private var reportedContentSize: Pair<Int, Int>? = null

    /** Hide the SDK source disclosure when the host already renders its own sources footer. */
    var showSources by mutableStateOf(true)

    private data class RenderedContent(
        val document: GenUiDocument,
        val snapshot: GenUiRenderSnapshot?,
        val identity: Long,
    )
    private var contentIdentity = 0L
    private var renderedContent by mutableStateOf<RenderedContent?>(null)
    internal val currentRenderSnapshot: GenUiRenderSnapshot?
        get() = renderedContent?.snapshot

    init {
        addView(
            scrollView,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT),
        )
        composeView.setContent {
            renderedContent?.let { content ->
                key(content.identity) {
                    GenUiContentImpl(
                        document = content.document,
                        onAction = { action -> this@GenUiView.onAction(action) },
                        showSources = showSources,
                        onSourcePreviewChanged = ::onSourcePreviewChanged,
                        embeddedMode = embeddedModeState,
                        surfaceKey = content.snapshot?.surfaceKey,
                        interactionEnabled = content.snapshot?.isFinal ?: true,
                    )
                }
            }
        }
    }

    fun render(document: GenUiDocument) {
        previewRevision++
        previewScrollPosition = null
        invalidateContentSizeReport()
        hasRenderedContent = true
        updateBackgroundColor()
        if (!embeddedMode) scrollView.scrollTo(0, 0)
        renderedContent = RenderedContent(document, null, ++contentIdentity)
        requestLayout()
    }

    /**
     * Applies an accepted snapshot on the main thread without remounting the
     * current surface or resetting its scroll/source disclosure state.
     * Older revisions and a preview arriving after finalization are ignored.
     */
    fun renderSnapshot(snapshot: GenUiRenderSnapshot) {
        val current = currentRenderSnapshot
        if (!shouldAcceptRenderSnapshot(current, snapshot)) return
        if (current?.surfaceKey != snapshot.surfaceKey) {
            contentIdentity++
            previewRevision++
            previewScrollPosition = null
            if (!embeddedMode) scrollView.scrollTo(0, 0)
        }
        invalidateContentSizeReport()
        hasRenderedContent = true
        updateBackgroundColor()
        renderedContent = RenderedContent(snapshot.document, snapshot, contentIdentity)
        requestLayout()
    }

    fun render(input: String) {
        render(GenUiCompiler.compile(input))
    }

    fun clear() {
        previewRevision++
        previewScrollPosition = null
        hasRenderedContent = false
        invalidateContentSizeReport()
        renderedContent = null
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        if (!embeddedMode) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec)
            return
        }
        val availableWidth = if (View.MeasureSpec.getMode(widthMeasureSpec) == View.MeasureSpec.UNSPECIFIED) {
            resources.displayMetrics.widthPixels
        } else {
            View.MeasureSpec.getSize(widthMeasureSpec)
        }
        val contentWidth = (availableWidth - paddingLeft - paddingRight).coerceAtLeast(0)
        composeView.measure(
            View.MeasureSpec.makeMeasureSpec(contentWidth, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED),
        )
        val naturalWidth = composeView.measuredWidth + paddingLeft + paddingRight
        val naturalHeight = composeView.measuredHeight + paddingTop + paddingBottom
        setMeasuredDimension(
            resolveSizeAndState(naturalWidth, widthMeasureSpec, composeView.measuredState),
            resolveSizeAndState(naturalHeight, heightMeasureSpec, composeView.measuredState shl View.MEASURED_HEIGHT_STATE_SHIFT),
        )
        if (hasRenderedContent && isAttachedToWindow && naturalWidth > 0 && naturalHeight > 0) {
            reportNaturalSize(naturalWidth, naturalHeight)
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        if (embeddedMode && hasRenderedContent) requestLayout()
    }

    override fun onDetachedFromWindow() {
        invalidateContentSizeReport()
        super.onDetachedFromWindow()
    }

    private fun reportNaturalSize(width: Int, height: Int) {
        val size = width to height
        if (size == reportedContentSize) {
            pendingContentSize = null
            return
        }
        if (size == pendingContentSize) return
        pendingContentSize = size
        val revision = sizeRevision
        post {
            if (revision != sizeRevision || !embeddedMode || !hasRenderedContent ||
                !isAttachedToWindow || pendingContentSize != size) return@post
            pendingContentSize = null
            if (reportedContentSize != size) {
                reportedContentSize = size
                onContentSizeChanged(width, height)
            }
        }
    }

    private fun invalidateContentSizeReport() {
        sizeRevision++
        pendingContentSize = null
        reportedContentSize = null
    }

    private fun updateBackgroundColor() {
        if (embeddedMode) {
            setBackgroundColor(Color.TRANSPARENT)
            return
        }
        val dark = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        setBackgroundColor(Color.parseColor(if (dark) "#101620" else "#F5F7FB"))
    }

    private fun onSourcePreviewChanged(visible: Boolean) {
        if (embeddedMode) return
        val revision = ++previewRevision
        if (visible) {
            previewScrollPosition = scrollView.scrollX to scrollView.scrollY
            return
        }
        val position = previewScrollPosition ?: return
        previewScrollPosition = null
        // Dialog focus restoration can scroll the ComposeView to its top. Restore
        // only this source preview's position, after its window has been removed.
        scrollView.postOnAnimation {
            scrollView.post {
                if (isAttachedToWindow && revision == previewRevision) {
                    scrollView.scrollTo(position.first, position.second)
                }
            }
        }
    }
}

internal fun shouldAcceptRenderSnapshot(current: GenUiRenderSnapshot?, incoming: GenUiRenderSnapshot): Boolean {
    if (current == null || current.surfaceKey != incoming.surfaceKey) return true
    if (current.isFinal && !incoming.isFinal) return false
    if (incoming.attempt != current.attempt) return incoming.attempt > current.attempt
    return incoming.revision > current.revision ||
        (incoming.revision == current.revision && incoming.isFinal && !current.isFinal)
}
