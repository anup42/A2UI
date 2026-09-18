package com.samsung.genuicraft.sdk.internal.renderer.native

import androidx.compose.foundation.clickable
import androidx.compose.ui.Modifier
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
import java.io.File
import java.util.Locale

internal object NativeActionParsing {

    enum class ComponentActionKind {
        OpenUrl,
        ShowMessage,
        ShowSurface
    }

    data class ComponentAction(
        val kind: ComponentActionKind,
        val value: String
    )

    sealed interface RuntimeAction {
        data class ShowMessage(val message: String) : RuntimeAction
        data class ShowSurface(val surfaceId: String) : RuntimeAction
    }

    // ── Modifier / execution helpers ──────────────────────────────────

    fun componentActionModifier(
        base: Modifier = Modifier,
        action: ComponentAction?,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (RuntimeAction) -> Unit
    ): Modifier {
        if (action == null) {
            return base
        }
        return base.clickable {
            executeComponentAction(
                action = action,
                sourceDir = sourceDir,
                onOpenExternalUrl = onOpenExternalUrl,
                onRuntimeAction = onRuntimeAction
            )
        }
    }

    fun executeComponentAction(
        action: ComponentAction?,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (RuntimeAction) -> Unit
    ) {
        if (action == null) {
            return
        }
        when (action.kind) {
            ComponentActionKind.OpenUrl -> {
                resolveExternalUrl(action.value, sourceDir)?.let(onOpenExternalUrl)
                    ?: onRuntimeAction(RuntimeAction.ShowMessage("Unable to open link."))
            }

            ComponentActionKind.ShowMessage -> onRuntimeAction(RuntimeAction.ShowMessage(action.value))
            ComponentActionKind.ShowSurface -> onRuntimeAction(RuntimeAction.ShowSurface(action.value))
        }
    }

    // ── Action extraction / parsing ──────────────────────────────────

    fun extractComponentAction(component: JsonObject): ComponentAction? {
        parseActionEnvelope(component.getAsJsonObjectOrNull("action"))?.let { return it }
        parseActionEnvelope(component.getAsJsonObjectOrNull("onClick"))?.let { return it }
        parseActionEvent(component.get("event"))?.let { return it }
        return null
    }

    fun parseActionEnvelope(actionObject: JsonObject?): ComponentAction? {
        if (actionObject == null) {
            return null
        }

        parseFunctionCallAction(actionObject.getAsJsonObjectOrNull("functionCall"))?.let { return it }
        parseFunctionCallAction(actionObject)?.let { return it }
        parseActionEvent(actionObject.get("event"))?.let { return it }

        listOf("onClick", "click", "tap", "press", "onPress", "onSelect", "select").forEach { key ->
            val nested = actionObject.getAsJsonObjectOrNull(key) ?: return@forEach
            parseActionEnvelope(nested)?.let { return it }
        }

        actionObject.getAsJsonArrayOrNull("events")?.forEach { eventEntry ->
            parseActionEvent(eventEntry)?.let { return it }
        }
        actionObject.getAsJsonArrayOrNull("handlers")?.forEach { eventEntry ->
            parseActionEvent(eventEntry)?.let { return it }
        }

        val callName = actionObject.getString("name")
            ?: actionObject.getString("call")
            ?: return null
        val args = actionObject.getAsJsonObjectOrNull("args")
            ?: readLegacyActionContextArgs(actionObject)
            ?: JsonObject()
        return parseActionFromCall(callName, args)
    }

    fun parseActionEvent(event: JsonElement?): ComponentAction? {
        if (event == null || event.isJsonNull) {
            return null
        }
        if (event.isJsonArray) {
            event.asJsonArray.forEach { entry ->
                parseActionEvent(entry)?.let { return it }
            }
            return null
        }
        if (!event.isJsonObject) {
            return null
        }
        val eventObject = event.asJsonObject
        parseActionEnvelope(eventObject)?.let { return it }
        listOf("onClick", "click", "tap", "press", "onPress", "onSelect", "select").forEach { key ->
            parseActionEnvelope(eventObject.getAsJsonObjectOrNull(key))?.let { return it }
        }
        return null
    }

    fun parseFunctionCallAction(functionCall: JsonObject?): ComponentAction? {
        if (functionCall == null) {
            return null
        }
        val callName = functionCall.getString("call")
            ?: functionCall.getString("name")
            ?: return null
        val args = functionCall.getAsJsonObjectOrNull("args")
            ?: readLegacyActionContextArgs(functionCall)
            ?: JsonObject()
        return parseActionFromCall(callName, args)
    }

    fun readLegacyActionContextArgs(actionObject: JsonObject): JsonObject? {
        val contextEntries = actionObject.getAsJsonArrayOrNull("context") ?: return null
        if (contextEntries.size() == 0) {
            return null
        }
        val args = JsonObject()
        contextEntries.forEach { contextElement ->
            val contextObject = contextElement.asJsonObjectOrNull() ?: return@forEach
            val key = contextObject.getString("key") ?: return@forEach
            contextObject.get("value")?.let { args.add(key, it) }
        }
        return if (args.entrySet().isEmpty()) null else args
    }

    fun parseActionFromCall(callName: String, args: JsonObject): ComponentAction? {
        val canonicalCall = callName.trim().lowercase(Locale.US)
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        return when (canonicalCall) {
            "openurl", "openlink", "launchurl", "browseurl" -> {
                readActionArgument(args, "url", "href", "link", "targetUrl")
                    ?.let { ComponentAction(ComponentActionKind.OpenUrl, it) }
            }

            "showmessage", "showtoast", "toast", "snackbar", "message" -> {
                readActionArgument(args, "message", "text", "title")
                    ?.let { ComponentAction(ComponentActionKind.ShowMessage, it) }
            }

            "showsurface", "opensurface", "navigatesurface", "switchsurface" -> {
                readActionArgument(args, "surfaceId", "id", "surface", "targetSurfaceId", "target")
                    ?.let { ComponentAction(ComponentActionKind.ShowSurface, it) }
            }

            else -> null
        }
    }

    fun readActionArgument(args: JsonObject, vararg keys: String): String? {
        keys.forEach { key ->
            val value = NativePayloadParser.readDynamicString(
                args.get(key),
                NativeTextFormatter::normalizeMojibakeText
            ).trim()
            if (value.isNotBlank()) {
                return value
            }
        }
        return null
    }

    // ── URL resolution (equivalent to GenUiNativeRenderer.resolveExternalUrl) ──

    fun resolveExternalUrl(raw: String, sourceDir: File?): String? =
        toExternalUrl(NativePayloadParser.resolveAssetUrl(raw, sourceDir))

    private fun toExternalUrl(value: String?): String? {
        if (value.isNullOrBlank()) {
            return null
        }
        val normalized = NativePayloadParser.canonicalizeNetworkUrlToken(value)
        return SafeContentPolicy.sanitizeActionUrl(normalized)
    }
}
