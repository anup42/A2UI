package com.samsung.genuicraft.sdk

import java.util.concurrent.CopyOnWriteArrayList
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiPreviewCoordinatorTest {
    @Test fun throttledWorkerPublishesNewestCumulativePrefix() = runBlocking {
        val snapshots = CopyOnWriteArrayList<GenUiRenderSnapshot>()
        val first = CompletableDeferred<Unit>()
        val second = CompletableDeferred<Unit>()
        val coordinator = GenUiPreviewCoordinator(this, emptyList(), { snapshot ->
            snapshots += snapshot
            if (snapshots.size == 1) first.complete(Unit)
            if (snapshots.size == 2) second.complete(Unit)
        }, updateIntervalMs = 500L)
        try {
            val early = "<a2ui>\nroot=Column([first,second,third])\nfirst=Text(\"First\")"
            val older = early + "\nsecond=Text(\"Second\")"
            val latest = older + "\nthird=Text(\"Third\")"
            coordinator.startAttempt(1)
            coordinator.offer(1, early)
            withTimeout(5_000) { first.await() }
            coordinator.offer(1, older)
            delay(50) // Let the worker enter the publication throttle with the older prefix.
            coordinator.offer(1, latest)
            withTimeout(5_000) { second.await() }
            assertTrue("Second publication skipped the latest completed component", snapshots[1].document.express.contains("Third"))
            assertFalse(snapshots[1].isFinal)
            val final = GenUiCompiler.compile(latest + "\n</a2ui>")
            coordinator.finish(GenUiConversionResult.Success(final, "test", 1, 1))
            assertEquals(3, snapshots.size)
            assertTrue(snapshots.last().isFinal)
            assertEquals(snapshots.first().surfaceKey, snapshots.last().surfaceKey)
        } finally {
            coordinator.stop()
        }
    }

    @Test fun replacedAttemptCannotPublishOldPrefix() = runBlocking {
        val snapshots = CopyOnWriteArrayList<GenUiRenderSnapshot>()
        val first = CompletableDeferred<Unit>()
        val second = CompletableDeferred<Unit>()
        val coordinator = GenUiPreviewCoordinator(this, emptyList(), { snapshot ->
            snapshots += snapshot
            if (snapshots.size == 1) first.complete(Unit)
            if (snapshots.size == 2) second.complete(Unit)
        })
        try {
            coordinator.startAttempt(1)
            coordinator.offer(1, "<a2ui>\nroot=Text(\"First attempt\")")
            withTimeout(5_000) { first.await() }
            coordinator.startAttempt(2)
            coordinator.offer(1, "<a2ui>\nroot=Text(\"Stale prefix\")")
            coordinator.offer(2, "<a2ui>\nroot=Text(\"Second attempt\")")
            withTimeout(5_000) { second.await() }
            assertEquals(2, snapshots[1].attempt)
            assertTrue(snapshots[1].document.express.contains("Second attempt"))
            assertFalse(snapshots[1].document.express.contains("Stale prefix"))
            assertFalse(snapshots[0].surfaceKey == snapshots[1].surfaceKey)
        } finally {
            coordinator.stop()
        }
    }
}
