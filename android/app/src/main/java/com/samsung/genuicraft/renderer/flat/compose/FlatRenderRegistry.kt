package com.samsung.genuicraft.renderer.flat.compose

import com.samsung.genuicraft.renderer.*

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import java.util.Locale
import com.samsung.genuicraft.renderer.flat.capability.GeneratedRendererCapabilities
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*
import com.samsung.genuicraft.renderer.flat.model.*

/**
 * Everything an element renderer needs, in one value.
 *
 * `RenderByType` threaded 14 parameters through every arm of a 25-way `when`,
 * and that repeated parameter list is the main reason `FlatSpecRenderer.kt`
 * resisted being split: moving one composable out meant restating the list.
 * Bundling it means a renderer is just `(FlatRenderContext) -> Unit`, which is
 * what makes a registry — and per-domain files — possible.
 */
internal data class FlatRenderContext(
    val elementId: String,
    val type: String,
    val props: Map<String, Any?>,
    val children: List<String>,
    val onMap: Map<String, Any?>?,
    val elements: Map<String, FlatElement>,
    val state: Map<String, Any?>,
    val repeatScope: RepeatScope?,
    val repeatedChildScopes: List<RepeatScope>?,
    val onOpenUrl: (String) -> Unit,
    val onSetState: (String, Any?) -> Unit,
    val onAction: (Any?, RepeatScope?) -> Int,
    val activePath: Set<String>,
    val modifier: Modifier = Modifier
) {
    /** Overrides a prop, used by the Row/Column aliases to pin `direction`. */
    fun withProp(key: String, value: Any?): FlatRenderContext =
        copy(props = props + mapOf(key to value))
}

internal typealias FlatElementRenderer = @Composable (FlatRenderContext) -> Unit

/**
 * Canonical element type -> renderer. The key set is generated-manifest backed.
 *
 * Alias spellings live in [FLAT_TYPE_ALIASES] so the alias set is data rather
 * than control flow, and so `describeFlatSpecRouting` and the capability
 * manifest can enumerate them instead of restating them.
 */
internal val FLAT_ELEMENT_RENDERERS: Map<String, FlatElementRenderer> = mapOf(
    "stack" to { c ->
        RenderStack(
            c.elementId, c.props, c.children, c.elements, c.state, c.repeatScope,
            c.repeatedChildScopes, c.onOpenUrl, c.onSetState, c.onAction, c.activePath, c.modifier
        )
    },
    "list" to { c ->
        RenderList(
            c.children, c.elements, c.state, c.repeatScope, c.repeatedChildScopes,
            c.onOpenUrl, c.onSetState, c.onAction, c.activePath, c.modifier
        )
    },
    "card" to { c ->
        RenderCard(
            c.elementId, c.props, c.children, c.elements, c.state, c.repeatScope,
            c.repeatedChildScopes, c.onOpenUrl, c.onSetState, c.onAction, c.activePath, c.modifier
        )
    },
    "table" to { c -> RenderDirectTable(c.props, c.state, c.onOpenUrl, c.modifier) },
    "formula" to { c -> RenderFormula(c.props, c.modifier) },
    "chart" to { c -> RenderChart(c.props, c.state, c.modifier) },
    "codeblock" to { c ->
        RenderCodeBlock(
            codeBlock = codeBlockFromProps(c.props, defaultLanguage = "text", isConsole = false),
            modifier = c.modifier
        )
    },
    "consolelog" to { c ->
        RenderCodeBlock(
            codeBlock = codeBlockFromProps(c.props, defaultLanguage = "console", isConsole = true),
            modifier = c.modifier
        )
    },
    "text" to { c -> RenderText(c.props, c.modifier) },
    "emailpreview" to { c -> RenderEmailPreview(c.props, c.modifier) },
    "image" to { c -> RenderImage(c.props, c.onOpenUrl, c.modifier) },
    "icon" to { c -> RenderIcon(c.props, c.modifier) },
    "button" to { c -> RenderButton(c.props, c.onMap, c.repeatScope, c.onAction, c.modifier) },
    "divider" to { c -> RenderDivider(c.modifier) },
    "tabs" to { c ->
        RenderTabs(
            c.props, c.elements, c.state, c.repeatScope, c.onOpenUrl, c.onSetState,
            c.onAction, c.activePath, c.modifier
        )
    },
    "modal" to { c ->
        RenderModal(
            c.props, c.children, c.elements, c.state, c.repeatScope, c.onOpenUrl,
            c.onSetState, c.onAction, c.activePath, c.modifier
        )
    },
    "textfield" to { c -> RenderTextField(c.props, c.onSetState, c.state, c.repeatScope, c.modifier) },
    "checkbox" to { c -> RenderCheckBox(c.props, c.onSetState, c.state, c.repeatScope, c.modifier) },
    "choicepicker" to { c -> RenderChoicePicker(c.props, c.onSetState, c.state, c.repeatScope, c.modifier) },
    "slider" to { c -> RenderSlider(c.props, c.onSetState, c.state, c.repeatScope, c.modifier) },
    "datetimeinput" to { c -> RenderDateTimeInput(c.props, c.onSetState, c.state, c.repeatScope, c.modifier) },
    "video" to { c -> RenderVideo(c.props, c.onOpenUrl, c.modifier) },
    "audioplayer" to { c -> RenderAudioPlayer(c.props, c.onOpenUrl, c.modifier) },
    "alert" to { c -> RenderAlert(c) },
    "checklist" to { c -> RenderChecklist(c) }
)

/**
 * Alias spelling -> canonical type. Mirrors the alias arms the `when` used to
 * carry; kept as data so it can be enumerated by tests and by
 * `renderer_capability.py`'s manifest.
 */
internal val FLAT_TYPE_ALIASES: Map<String, String> =
    GeneratedRendererCapabilities.typeAliases.filter { (alias, canonical) -> alias != canonical } +
        GeneratedRendererCapabilities.compatibilityTypeDirections.keys.associateWith { "stack" }

/** Canonical type for a raw IR type string, or null when unsupported. */
internal fun canonicalFlatType(rawType: String): String? {
    val normalized = rawType.trim().lowercase(Locale.US)
    return when {
        normalized in FLAT_ELEMENT_RENDERERS -> normalized
        else -> FLAT_TYPE_ALIASES[normalized]?.takeIf(FLAT_ELEMENT_RENDERERS::containsKey)
    }
}

/** Renderer for a raw IR type string, resolving aliases. */
internal fun flatRendererFor(rawType: String): FlatElementRenderer? =
    canonicalFlatType(rawType)?.let { FLAT_ELEMENT_RENDERERS[it] }
