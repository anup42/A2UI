package com.samsung.genuicraft.sdk

import android.content.Context
import android.content.res.Configuration
import android.graphics.Color
import android.util.AttributeSet
import android.widget.FrameLayout
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.compose.runtime.getValue
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
    private var previewScrollPosition: Pair<Int, Int>? = null
    private var previewRevision = 0L

    /** Hide the SDK source disclosure when the host already renders its own sources footer. */
    var showSources by mutableStateOf(true)

    init {
        addView(
            scrollView,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT),
        )
    }

    fun render(document: GenUiDocument) {
        previewRevision++
        previewScrollPosition = null
        val dark = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        setBackgroundColor(Color.parseColor(if (dark) "#101620" else "#F5F7FB"))
        scrollView.scrollTo(0, 0)
        composeView.setContent {
            GenUiContentImpl(
                document = document,
                onAction = { action -> this@GenUiView.onAction(action) },
                showSources = showSources,
                onSourcePreviewChanged = ::onSourcePreviewChanged,
            )
        }
    }

    fun render(input: String) {
        render(GenUiCompiler.compile(input))
    }

    fun clear() {
        previewRevision++
        previewScrollPosition = null
        composeView.setContent {}
    }

    private fun onSourcePreviewChanged(visible: Boolean) {
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
