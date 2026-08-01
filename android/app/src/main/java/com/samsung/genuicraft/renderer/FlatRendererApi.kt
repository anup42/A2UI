package com.samsung.genuicraft.renderer

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Stable
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshots.SnapshotStateMap
import androidx.compose.ui.platform.LocalContext
import coil.ImageLoader
import com.samsung.genuicraft.security.SafeContentPolicy
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.compose.*

/**
 * Host-agnostic seam for the renderer.
 *
 * Previously the renderer reached for `LocalContext` and called
 * `context.startActivity(Intent.ACTION_VIEW, ...)` itself, so a host could
 * neither intercept navigation nor supply its own image loading — and the
 * renderer could not be extracted from the app module. Everything it needs from
 * the outside world now arrives through this interface.
 */
interface FlatRendererHost {
    /** Called instead of the renderer launching an intent itself. */
    fun openUrl(url: String)

    /** Maps an IR asset reference to something loadable. */
    fun resolveAssetUrl(raw: String): String = raw

    /** The single loader every image and icon composable should use. */
    val imageLoader: ImageLoader?
        get() = null

    /** Structured renderer diagnostics. Existing string-only hosts remain valid. */
    fun onDiagnostic(diagnostic: FlatDiagnostic) {
        onDiagnostic(diagnostic.message)
    }

    /** @deprecated Override [onDiagnostic] with [FlatDiagnostic] for typed details. */
    @Deprecated("Override onDiagnostic(FlatDiagnostic) instead.")
    fun onDiagnostic(message: String) {
        Log.w(FLAT_SPEC_RENDERER_TAG, message)
    }
}

data class FlatDiagnostic(
    val code: Code,
    val severity: Severity,
    val message: String,
    val elementId: String? = null,
    val details: Map<String, Any?> = emptyMap()
) {
    enum class Severity { INFO, WARNING, ERROR }

    enum class Code {
        UNSUPPORTED_TYPE,
        MISSING_ELEMENT,
        REFERENCE_CYCLE,
        INVALID_ACTION,
        UNSUPPORTED_ACTION,
        UNSUPPORTED_CHART_TYPE,
        EMPTY_REQUIRED_DATA,
        LEGACY_FALLBACK,
        DUPLICATE_REPEAT_KEY,
        WATCH_CASCADE_TERMINATED,
        INVALID_PAYLOAD
    }
}

data class FlatStateMutation(
    val path: String,
    val operation: Operation,
    val before: Any? = null,
    val after: Any? = null
) {
    enum class Operation { SET, PUSH, REMOVE, DELETE, VALIDATE }
}

/** What happened inside the renderer, for hosts and agents that care. */
data class FlatRenderEvent(
    val kind: Kind,
    val action: String? = null,
    val params: Map<String, Any?> = emptyMap(),
    val elementId: String? = null,
    val statePath: String? = null,
    val value: Any? = null,
    val success: Boolean? = null,
    val stateMutations: List<FlatStateMutation> = emptyList()
) {
    enum class Kind { ACTION, STATE_CHANGE, NAVIGATION, DIAGNOSTIC }
}

/**
 * Default host: preserves the renderer's previous behaviour exactly, so the
 * legacy [FlatSpecContent] overload is unchanged.
 */
internal class DefaultFlatRendererHost(
    private val context: Context,
    private val assetResolver: (String) -> String,
    override val imageLoader: ImageLoader?
) : FlatRendererHost {
    override fun openUrl(url: String) {
        SafeContentPolicy.sanitizeActionUrl(url)?.let { safeUrl ->
            runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(safeUrl))) }
        }
    }

    override fun resolveAssetUrl(raw: String): String = assetResolver(raw)
}

/**
 * Externally-owned render state.
 *
 * The renderer used to build its state map with `remember(spec)`, so **any** new
 * spec instance wiped every value the user had entered — which makes an
 * incremental `surfaceUpdate` impossible. Hoisting it lets the host decide
 * whether an update preserves state.
 */
@Stable
class FlatSpecStateHolder(initial: Map<String, Any?>) {

    val state: SnapshotStateMap<String, Any?> =
        mutableStateMapOf<String, Any?>().apply { putAll(initial) }

    fun setAtPath(path: String, value: Any?) {
        val normalized = normalizePointer(path)
        if (normalized.isNotBlank()) {
            FlatSpecParser.setAtPath(state, normalized, value)
        }
    }

    /**
     * Applies an A2UI-style `dataModelUpdate`. Omitting [value] deletes the path,
     * matching the protocol's upsert-or-delete semantics.
     */
    fun applyDataModelUpdate(path: String?, value: Any? = null, delete: Boolean = false) {
        val normalized = normalizePointer(path.orEmpty())
        if (normalized.isBlank()) return
        if (delete) {
            FlatSpecParser.removeAtPath(state, normalized)
        } else {
            FlatSpecParser.setAtPath(state, normalized, value)
        }
    }

    /** Replaces state wholesale, e.g. on an explicit surface reset. */
    fun reset(values: Map<String, Any?>) {
        state.clear()
        state.putAll(values)
    }

    fun snapshot(): Map<String, Any?> = state.toMap()
}

/**
 * Remembers a holder keyed on [key].
 *
 * Pass a stable key (a surface id rather than the spec instance) when state
 * should survive a spec update.
 */
@Composable
fun rememberFlatSpecStateHolder(
    key: Any?,
    initial: Map<String, Any?>
): FlatSpecStateHolder = remember(key) { FlatSpecStateHolder(initial) }

/** Builds the default host from the composition, preserving legacy behaviour. */
@Composable
internal fun rememberDefaultFlatRendererHost(
    resolveAssetUrl: (String) -> String
): FlatRendererHost {
    val context = LocalContext.current
    val imageLoader = rememberFlatImageLoader()
    return remember(context, resolveAssetUrl, imageLoader) {
        DefaultFlatRendererHost(context, resolveAssetUrl, imageLoader)
    }
}
