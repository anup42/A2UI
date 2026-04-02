package com.samsung.genuicraft

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat

internal object PipelineRunNotifier {
    private const val CHANNEL_ID = "genuicraft_pipeline"
    private const val CHANNEL_NAME = "GenUICraft Pipeline"
    private const val NOTIFICATION_ID = 20041

    fun showRunning(context: Context, status: String) {
        if (!hasNotificationPermission(context)) return
        ensureChannel(context)
        NotificationManagerCompat.from(context).notify(
            NOTIFICATION_ID,
            baseBuilder(context)
                .setContentTitle("GenUICraft is running")
                .setContentText(status.ifBlank { "Generating UI..." })
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setProgress(0, 0, true)
                .build()
        )
    }

    fun showCompleted(context: Context, status: String) {
        if (!hasNotificationPermission(context)) return
        ensureChannel(context)
        NotificationManagerCompat.from(context).notify(
            NOTIFICATION_ID,
            baseBuilder(context)
                .setContentTitle("GenUICraft completed")
                .setContentText(status.ifBlank { "Rendered output is ready." })
                .setOngoing(false)
                .setAutoCancel(true)
                .setProgress(0, 0, false)
                .build()
        )
    }

    fun showFailed(context: Context, message: String) {
        if (!hasNotificationPermission(context)) return
        ensureChannel(context)
        NotificationManagerCompat.from(context).notify(
            NOTIFICATION_ID,
            baseBuilder(context)
                .setContentTitle("GenUICraft failed")
                .setContentText(message.ifBlank { "Generation failed." })
                .setOngoing(false)
                .setAutoCancel(true)
                .setProgress(0, 0, false)
                .build()
        )
    }

    fun clear(context: Context) {
        NotificationManagerCompat.from(context).cancel(NOTIFICATION_ID)
    }

    private fun baseBuilder(context: Context): NotificationCompat.Builder {
        val launchIntent = Intent(context, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_genui)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
            .setSilent(true)
    }

    private fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            CHANNEL_NAME,
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Pipeline status updates while GenUICraft runs in background."
        }
        manager.createNotificationChannel(channel)
    }

    private fun hasNotificationPermission(context: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return true
        }
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED
    }
}
