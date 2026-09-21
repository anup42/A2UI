package com.samsung.genuicraft

import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiRepairKind
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Device replay for captured malformed output. No provider or model is constructed. */
@RunWith(AndroidJUnit4::class)
class SdkRecoveryOnDeviceTest {

    @Test
    fun capturedTrainAndWeatherRecoverRenderAndRemainVisibleWhileScrolling() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val arguments = InstrumentationRegistry.getArguments()
        val device = UiDevice.getInstance(instrumentation)
        val maximumPages = arguments.getString("recoveryScrollPages", DEFAULT_SCROLL_PAGES.toString())
            .orEmpty().toIntOrNull()?.coerceIn(MIN_SCROLL_PAGES, MAX_SCROLL_PAGES)
            ?: DEFAULT_SCROLL_PAGES
        val artifactRoot = internalPath(
            context,
            arguments.getString("recoveryArtifactDir", DEFAULT_ARTIFACT_DIRECTORY)
                ?: DEFAULT_ARTIFACT_DIRECTORY,
        )
        val runDirectory = File(
            artifactRoot,
            "${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(8)}",
        )
        check(runDirectory.mkdirs()) { "Could not create ${runDirectory.absolutePath}" }

        val cases = listOf(
            RecoveryCase(
                id = "BXP-003",
                argument = "recoveryTrainRawPath",
                defaultPath = "recovery-input/BXP-003.raw.express",
                sha256 = "c2a35a2577e2eb329eb3c3d4d0c1409c9e462253f718214699f3f350d92d3dd9",
                rowMarkers = listOf(
                    "Shatabdi Express (120007)",
                    "Rajya Rani Rani Express (20666)",
                    "Wodeyar Express (12461)",
                ),
                proseMarker = "If you want",
            ),
            RecoveryCase(
                id = "BXP-001",
                argument = "recoveryWeatherRawPath",
                defaultPath = "recovery-input/BXP-001.raw.express",
                sha256 = "9f1c75d8fa32efed9b35cc1271e4636fbfa84d7a4d26a9bb74d9a560fb9494fe",
                rowMarkers = listOf("Wed, Sep 9", "Thu, Sep 10", "Fri, Sep 1"),
                proseMarker = "Carry an umbrella or light rain jacket",
            ),
        )

        val reports = cases.map { testCase ->
            val rawFile = internalPath(
                context,
                arguments.getString(testCase.argument, testCase.defaultPath) ?: testCase.defaultPath,
            )
            assertTrue("Missing staged recovery fixture: ${rawFile.absolutePath}", rawFile.isFile)
            assertTrue("Unreadable staged recovery fixture: ${rawFile.absolutePath}", rawFile.canRead())
            val rawBytes = rawFile.readBytes()
            assertEquals("Staged raw fixture changed: ${testCase.id}", testCase.sha256, sha256(rawBytes))

            val outcome = GenUiCompiler.compileWithRepair(
                input = rawBytes.toString(Charsets.UTF_8),
                allowSourceTextFallback = false,
                allowGeneratedDslRepair = true,
            )
            assertEquals(
                "${testCase.id} did not use generated-output repair",
                GenUiRepairKind.GENERATED_DSL_REPAIR,
                outcome.repairKind,
            )

            val caseDirectory = File(runDirectory, testCase.id).apply {
                check(mkdirs()) { "Could not create $absolutePath" }
            }
            File(caseDirectory, "raw.express").writeBytes(rawBytes)
            File(caseDirectory, "repaired.express")
                .writeText(outcome.document.express, Charsets.UTF_8)
            File(caseDirectory, "repaired.a2ui.json")
                .writeText(outcome.document.a2uiJson, Charsets.UTF_8)
            File(caseDirectory, "diagnostics.txt").writeText(
                outcome.diagnostics.joinToString(separator = "\n", postfix = "\n"),
                Charsets.UTF_8,
            )

            val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
            val capture = ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
                scenario.onActivity { activity ->
                    activity.showDocument(
                        outcome.document,
                        "${testCase.id} · generated-output recovery · no inference",
                    )
                }
                instrumentation.waitForIdleSync()
                assertForeground(device, context.packageName, "render ${testCase.id}")
                assertTrue(
                    "${testCase.id} preview did not become ready",
                    device.wait(Until.hasObject(By.pkg(context.packageName).text("Preview")), UI_TIMEOUT),
                )
                captureWhileScrolling(
                    instrumentation = instrumentation,
                    device = device,
                    packageName = context.packageName,
                    directory = caseDirectory,
                    expected = testCase.rowMarkers + testCase.proseMarker,
                    maximumPages = maximumPages,
                )
            }

            testCase.rowMarkers.forEach { expected ->
                assertTrue(
                    "${testCase.id} recovered row was not visible after scrolling: $expected\n" +
                        "Observed UI text: ${capture.texts}",
                    capture.texts.any { it.contains(expected) },
                )
            }
            assertTrue(
                "${testCase.id} surviving prose was not visible after scrolling: ${testCase.proseMarker}",
                capture.texts.any { it.contains(testCase.proseMarker) },
            )

            linkedMapOf<String, Any>(
                "id" to testCase.id,
                "rawPath" to rawFile.absolutePath,
                "rawSha256" to sha256(rawBytes),
                "repairKind" to outcome.repairKind.name,
                "pagesCaptured" to capture.pages,
                "observedTexts" to capture.texts,
                "rowMarkers" to testCase.rowMarkers,
                "proseMarker" to testCase.proseMarker,
            ).also { report ->
                File(caseDirectory, "result.json").writeText(GSON.toJson(report), Charsets.UTF_8)
            }
        }

        File(runDirectory, "summary.json").writeText(
            GSON.toJson(
                linkedMapOf(
                    "fixtureCount" to reports.size,
                    "realInferenceUsed" to false,
                    "sourceFallbackAllowed" to false,
                    "results" to reports,
                )
            ),
            Charsets.UTF_8,
        )
    }

    private fun captureWhileScrolling(
        instrumentation: android.app.Instrumentation,
        device: UiDevice,
        packageName: String,
        directory: File,
        expected: List<String>,
        maximumPages: Int,
    ): UiCapture {
        val observed = linkedSetOf<String>()
        var pages = 0
        repeat(maximumPages) { page ->
            instrumentation.waitForIdleSync()
            device.waitForIdle(UI_IDLE_TIMEOUT)
            assertForeground(device, packageName, "capture recovery page $page")
            dumpFreshHierarchy(
                instrumentation,
                device,
                File(directory, "page_${page.toString().padStart(2, '0')}.xml"),
            )
            observed += device.findObjects(By.pkg(packageName)).flatMap { node ->
                listOfNotNull(
                    runCatching { node.text?.trim() }.getOrNull(),
                    runCatching { node.contentDescription?.trim() }.getOrNull(),
                )
            }.filter(String::isNotBlank)

            val screenshot = File(directory, "page_${page.toString().padStart(2, '0')}.png")
            assertTrue("Screenshot failed: ${screenshot.absolutePath}", device.takeScreenshot(screenshot))
            pages += 1

            val foundEverything = expected.all { wanted -> observed.any { it.contains(wanted) } }
            if (pages >= MIN_SCROLL_PAGES && foundEverything) {
                return UiCapture(observed.toList(), pages)
            }
            if (page < maximumPages - 1) {
                assertForeground(device, packageName, "scroll recovery preview")
                device.swipe(
                    device.displayWidth / 2,
                    device.displayHeight * 84 / 100,
                    device.displayWidth / 2,
                    device.displayHeight * 36 / 100,
                    SWIPE_STEPS,
                )
                Thread.sleep(SCROLL_SETTLE_MS)
            }
        }
        return UiCapture(observed.toList(), pages)
    }

    private fun dumpFreshHierarchy(
        instrumentation: android.app.Instrumentation,
        device: UiDevice,
        target: File,
    ) {
        instrumentation.waitForIdleSync()
        val automation = instrumentation.uiAutomation
        if (Build.VERSION.SDK_INT >= 34) {
            check(automation.clearCache()) { "Unable to clear the accessibility cache before capture." }
        }
        val root = requireNotNull(automation.rootInActiveWindow) {
            "No accessibility root before ${target.name}"
        }
        check(root.refresh()) { "Accessibility root became stale before ${target.name}" }
        device.dumpWindowHierarchy(target)
    }

    private fun internalPath(context: Context, rawPath: String): File {
        val root = context.filesDir.canonicalFile
        val supplied = File(rawPath)
        val resolved = (if (supplied.isAbsolute) supplied else File(root, rawPath)).canonicalFile
        require(resolved == root || resolved.path.startsWith(root.path + File.separator)) {
            "Recovery test paths must stay inside ${root.absolutePath}: $rawPath"
        }
        return resolved
    }

    private fun assertForeground(device: UiDevice, packageName: String, action: String) {
        assertEquals(
            "Refusing to $action because another app is foreground",
            packageName,
            device.currentPackageName,
        )
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private data class RecoveryCase(
        val id: String,
        val argument: String,
        val defaultPath: String,
        val sha256: String,
        val rowMarkers: List<String>,
        val proseMarker: String,
    )

    private data class UiCapture(val texts: List<String>, val pages: Int)

    private companion object {
        const val DEFAULT_ARTIFACT_DIRECTORY = "result/sdk_recovery"
        const val DEFAULT_SCROLL_PAGES = 8
        const val MIN_SCROLL_PAGES = 2
        const val MAX_SCROLL_PAGES = 12
        const val UI_TIMEOUT = 10_000L
        const val UI_IDLE_TIMEOUT = 1_000L
        const val SCROLL_SETTLE_MS = 300L
        const val SWIPE_STEPS = 35
        val GSON = GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create()
    }
}
