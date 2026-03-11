package com.samsung.genuicraft

import android.content.Context
import android.os.Build
import android.util.Log
import android.view.View
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

private const val WINDOW_BACKGROUND_BLUR_RADIUS_PX = 120
private const val WINDOW_BEHIND_BLUR_RADIUS_PX = 56
private const val SAMSUNG_SEM_WINDOW_BLUR_RADIUS_PX = 96
private const val WINDOW_BLUR_TAG = "GenUiWindowBlur"

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

    val decorView = window.decorView
    val windowManager = getSystemService(WindowManager::class.java) ?: return
    val crossWindowBlurEnabled = windowManager.isCrossWindowBlurEnabled
    if (!crossWindowBlurEnabled) {
        window.setBackgroundBlurRadius(0)
        window.clearFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
        // Android 16 / Samsung builds can report cross-window blur disabled while
        // still supporting Sem window blur APIs (same visual family as quick panel frost).
        if (!applySamsungSemWindowBlur(decorView, SAMSUNG_SEM_WINDOW_BLUR_RADIUS_PX)) {
            clearSamsungSemWindowBlur(decorView)
            Log.d(WINDOW_BLUR_TAG, "Cross-window blur disabled and Samsung Sem blur unavailable.")
        }
    } else {
        clearSamsungSemWindowBlur(decorView)
        window.setBackgroundBlurRadius(WINDOW_BACKGROUND_BLUR_RADIUS_PX)
        window.addFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
        val layoutParams = window.attributes
        if (layoutParams.blurBehindRadius != WINDOW_BEHIND_BLUR_RADIUS_PX) {
            layoutParams.blurBehindRadius = WINDOW_BEHIND_BLUR_RADIUS_PX
            window.attributes = layoutParams
        }
        Log.d(
            WINDOW_BLUR_TAG,
            "Applied cross-window blur (background=$WINDOW_BACKGROUND_BLUR_RADIUS_PX, behind=$WINDOW_BEHIND_BLUR_RADIUS_PX)."
        )
    }
}

private fun isLikelySamsungDevice(): Boolean {
    val manufacturer = Build.MANUFACTURER.orEmpty().lowercase(Locale.US)
    return manufacturer.contains("samsung")
}

private fun applySamsungSemWindowBlur(view: View, radius: Int): Boolean {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S || !isLikelySamsungDevice()) {
        return false
    }
    return runCatching {
        val semBlurInfoClass = Class.forName("android.view.SemBlurInfo")
        val builderClass = Class.forName("android.view.SemBlurInfo\$Builder")
        val blurModeWindow = semBlurInfoClass.getField("BLUR_MODE_WINDOW").getInt(null)
        val builder = builderClass.getConstructor(Int::class.javaPrimitiveType).newInstance(blurModeWindow)
        builderClass.getMethod("setRadius", Int::class.javaPrimitiveType).invoke(builder, radius)
        val blurInfo = builderClass.getMethod("build").invoke(builder)
        View::class.java.getMethod("semSetBlurInfo", semBlurInfoClass).invoke(view, blurInfo)
        Log.d(WINDOW_BLUR_TAG, "Applied Samsung Sem window blur radius=$radius.")
        true
    }.getOrElse { error ->
        Log.w(WINDOW_BLUR_TAG, "Samsung Sem window blur apply failed: ${error.message}")
        false
    }
}

private fun clearSamsungSemWindowBlur(view: View) {
    if (!isLikelySamsungDevice()) {
        return
    }
    runCatching {
        val semBlurInfoClass = Class.forName("android.view.SemBlurInfo")
        View::class.java.getMethod("semSetBlurInfo", semBlurInfoClass).invoke(view, null)
    }.onFailure {
        // Ignore devices/builds that do not expose Samsung Sem blur APIs.
    }
}
