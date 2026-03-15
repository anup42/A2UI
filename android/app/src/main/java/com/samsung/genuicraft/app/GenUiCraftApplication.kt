package com.samsung.genuicraft

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.view.WindowManager
import com.google.android.material.color.DynamicColors

class GenUiCraftApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // This is the Android-supported way for third-party apps to follow
        // wallpaper-driven system dynamic colors on compatible devices.
        DynamicColors.applyToActivitiesIfAvailable(this)
        GeminiApiKeyProvider.refresh(this)

        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {
                activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }

            override fun onActivityStarted(activity: Activity) = Unit
            override fun onActivityResumed(activity: Activity) = Unit
            override fun onActivityPaused(activity: Activity) = Unit
            override fun onActivityStopped(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit
        })
    }
}
