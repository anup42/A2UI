package com.samsung.genuicraft

import android.app.Application
import com.google.android.material.color.DynamicColors

class GenUiCraftApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // This is the Android-supported way for third-party apps to follow
        // wallpaper-driven system dynamic colors on compatible devices.
        DynamicColors.applyToActivitiesIfAvailable(this)
    }
}

