package com.samsung.genuicraft

import android.content.Context
import android.os.Build
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity

private const val WINDOW_BACKGROUND_BLUR_RADIUS_PX = 120
private const val WINDOW_BEHIND_BLUR_RADIUS_PX = 56

fun Context.isCrossWindowBlurActive(): Boolean {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) {
        return false
    }
    val windowManager = getSystemService(WindowManager::class.java) ?: return false
    return windowManager.isCrossWindowBlurEnabled
}

fun AppCompatActivity.applyOneUiWindowBlur() {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) {
        return
    }

    val windowManager = getSystemService(WindowManager::class.java) ?: return
    if (!windowManager.isCrossWindowBlurEnabled) {
        window.setBackgroundBlurRadius(0)
        window.clearFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
        return
    }

    window.setBackgroundBlurRadius(WINDOW_BACKGROUND_BLUR_RADIUS_PX)
    window.addFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
    val layoutParams = window.attributes
    if (layoutParams.blurBehindRadius != WINDOW_BEHIND_BLUR_RADIUS_PX) {
        layoutParams.blurBehindRadius = WINDOW_BEHIND_BLUR_RADIUS_PX
        window.attributes = layoutParams
    }
}
