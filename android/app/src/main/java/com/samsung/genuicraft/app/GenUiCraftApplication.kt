package com.samsung.genuicraft

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.view.WindowManager
import coil.ImageLoader
import coil.ImageLoaderFactory
import com.google.android.material.color.DynamicColors
import com.samsung.genuicraft.renderer.flat.compose.*

class GenUiCraftApplication : Application(), ImageLoaderFactory {

    /**
     * Supplies the single renderer-wide Coil loader. `Context.imageLoader`
     * resolves to this, so every image and icon composable shares one memory
     * and disk cache instead of building its own.
     */
    override fun newImageLoader(): ImageLoader = buildFlatImageLoader(this)

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
