package com.samsung.genuicraft.sdk

import android.content.Context
import android.content.res.Configuration
import android.graphics.Color
import android.util.AttributeSet
import android.widget.FrameLayout
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
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

    init {
        addView(
            scrollView,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT),
        )
    }

    fun render(document: GenUiDocument) {
        val dark = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        setBackgroundColor(Color.parseColor(if (dark) "#101620" else "#F5F7FB"))
        scrollView.scrollTo(0, 0)
        composeView.setContent {
            GenUiContent(
                document = document,
                onAction = { action -> this@GenUiView.onAction(action) },
            )
        }
    }

    fun render(input: String) {
        render(GenUiCompiler.compile(input))
    }

    fun clear() {
        composeView.setContent {}
    }
}
