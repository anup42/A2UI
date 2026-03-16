package com.samsung.genuicraft.renderer.native

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.GenUiTokens
import com.samsung.genuicraft.genUiCardBorderColor
import java.io.File
import java.util.Locale

internal object NativeFormComponents {

    data class ChoiceOption(
        val label: String,
        val value: String
    )

    // ── Helper functions ─────────────────────────────────────────────

    fun readChoiceOptions(component: JsonObject): List<ChoiceOption> {
        val options = component.getAsJsonArrayOrNull("options") ?: return emptyList()
        return options.mapNotNull { option ->
            val obj = option.asJsonObjectOrNull() ?: return@mapNotNull null
            val value = NativeTextFormatter.sanitizeDisplayText(
                NativePayloadParser.readDynamicString(obj.get("value"), NativeTextFormatter::normalizeMojibakeText)
            ).ifBlank {
                NativeTextFormatter.sanitizeDisplayText(
                    NativePayloadParser.readDynamicString(obj.get("label"), NativeTextFormatter::normalizeMojibakeText)
                )
            }
            val label = NativeTextFormatter.sanitizeDisplayText(
                NativePayloadParser.readDynamicString(obj.get("label"), NativeTextFormatter::normalizeMojibakeText)
            ).ifBlank { value }
            if (value.isBlank()) {
                null
            } else {
                ChoiceOption(label = label, value = value)
            }
        }
    }

    fun readDynamicStringList(element: JsonElement?): List<String> {
        if (element == null || element.isJsonNull) {
            return emptyList()
        }
        if (element.isJsonArray) {
            return element.asJsonArray.map {
                NativePayloadParser.readDynamicString(it, NativeTextFormatter::normalizeMojibakeText).trim()
            }.filter { it.isNotBlank() }
        }
        val token = NativePayloadParser.readDynamicString(element, NativeTextFormatter::normalizeMojibakeText).trim()
        if (token.isBlank()) {
            return emptyList()
        }
        return token
            .split(',', ';', '|')
            .map { it.trim() }
            .filter { it.isNotBlank() }
    }

    fun parseBooleanLike(value: String?): Boolean? {
        return when (value?.trim()?.lowercase(Locale.US)) {
            "true", "1", "yes", "on" -> true
            "false", "0", "no", "off" -> false
            else -> null
        }
    }

    fun readDynamicNumber(element: JsonElement?): Float? {
        if (element == null || element.isJsonNull) {
            return null
        }
        if (element.isJsonPrimitive && element.asJsonPrimitive.isNumber) {
            return element.asFloat
        }
        val token = NativePayloadParser.readDynamicString(element, NativeTextFormatter::normalizeMojibakeText)
            .trim().replace(",", "")
        return token.toFloatOrNull()
    }

    fun formatSliderValue(value: Float): String {
        val rounded = value.toDouble()
        val long = rounded.toLong()
        return if (long.toDouble() == rounded) long.toString() else "%.2f".format(Locale.US, rounded)
    }

    /**
     * Resolves a display label from a component ID in the index.
     * This replicates the logic in GenUiNativeRenderer.resolveComponentLabel.
     */
    fun resolveComponentLabel(componentId: String?, index: Map<String, JsonObject>): String {
        if (componentId.isNullOrBlank()) {
            return ""
        }
        val component = index[componentId] ?: return ""
        val type = component.getString("component") ?: return ""
        if (type.equals("Text", ignoreCase = true)) {
            return NativeTextFormatter.sanitizeDisplayText(
                NativePayloadParser.readDynamicString(component.get("text"), NativeTextFormatter::normalizeMojibakeText)
            )
        }
        if (type.equals("Button", ignoreCase = true)) {
            val childId = component.getString("child")
            if (!childId.isNullOrBlank()) {
                val child = index[childId]
                if (child != null && child.getString("component").equals("Text", ignoreCase = true)) {
                    return NativeTextFormatter.sanitizeDisplayText(
                        NativePayloadParser.readDynamicString(child.get("text"), NativeTextFormatter::normalizeMojibakeText)
                    )
                }
            }
        }
        return ""
    }

    // ── Private readDynamicString shortcut ────────────────────────────

    private fun readDynamicString(element: JsonElement?): String =
        NativePayloadParser.readDynamicString(element, NativeTextFormatter::normalizeMojibakeText)

    private fun sanitizeDisplayText(text: String): String =
        NativeTextFormatter.sanitizeDisplayText(text)

    // ── @Composable form components ──────────────────────────────────

    @Composable
    fun RenderModalComponent(
        component: JsonObject,
        index: Map<String, JsonObject>,
        sourceDir: File?,
        onOpenExternalUrl: (String) -> Unit,
        onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
        activePath: Set<String>,
        renderComponent: @Composable (
            id: String,
            index: Map<String, JsonObject>,
            sourceDir: File?,
            onOpenExternalUrl: (String) -> Unit,
            onRuntimeAction: (NativeActionParsing.RuntimeAction) -> Unit,
            activePath: Set<String>
        ) -> Unit
    ) {
        val componentId = component.getString("id")
        val triggerId = component.getString("trigger")
        val contentId = component.getString("content")
        var open by remember(componentId, triggerId, contentId) { mutableStateOf(false) }
        val triggerLabel = resolveComponentLabel(triggerId, index).ifBlank { "Open details" }

        OutlinedButton(
            onClick = { open = true },
            shape = RoundedCornerShape(GenUiTokens.RadiusPill),
            border = BorderStroke(GenUiTokens.BorderMd, genUiCardBorderColor())
        ) {
            Text(triggerLabel, style = MaterialTheme.typography.labelLarge)
        }

        if (!open) {
            return
        }

        AlertDialog(
            onDismissRequest = { open = false },
            title = null,
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (contentId.isNullOrBlank()) {
                        Text(
                            "Missing modal content",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    } else {
                        renderComponent(
                            contentId,
                            index,
                            sourceDir,
                            onOpenExternalUrl,
                            onRuntimeAction,
                            activePath
                        )
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { open = false }) {
                    Text("Close")
                }
            }
        )
    }

    @Composable
    fun RenderTextFieldComponent(component: JsonObject) {
        val componentId = component.getString("id")
        val label = sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Input" }
        val variant = (component.getString("variant") ?: "shortText").lowercase(Locale.US)
        val initialValue = readDynamicString(component.get("value"))
        var value by remember(componentId) { mutableStateOf(initialValue) }

        val keyboardType = if (variant.contains("number")) KeyboardType.Number else KeyboardType.Text
        val singleLine = !variant.contains("longtext")
        val visualTransformation =
            if (variant.contains("obscured")) PasswordVisualTransformation() else VisualTransformation.None

        OutlinedTextField(
            value = value,
            onValueChange = { value = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(label) },
            singleLine = singleLine,
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
            visualTransformation = visualTransformation
        )
    }

    @Composable
    fun RenderCheckBoxComponent(component: JsonObject) {
        val componentId = component.getString("id")
        val label = sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Option" }
        val initial = component.get("value")?.asBooleanOrNull()
            ?: parseBooleanLike(readDynamicString(component.get("value")))
            ?: false
        var checked by remember(componentId) { mutableStateOf(initial) }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Checkbox(
                checked = checked,
                onCheckedChange = { checked = it }
            )
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }

    @Composable
    fun RenderChoicePickerComponent(component: JsonObject) {
        val componentId = component.getString("id")
        val label = sanitizeDisplayText(readDynamicString(component.get("label")))
        val variant = (component.getString("variant") ?: "mutuallyExclusive").lowercase(Locale.US)
        val multiple = variant.contains("multiple")
        val options = readChoiceOptions(component)

        if (options.isEmpty()) {
            Text("ChoicePicker has no options", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            return
        }

        val initialValues = readDynamicStringList(component.get("value")).toMutableSet()
        if (multiple) {
            val selected = remember(componentId) {
                mutableStateListOf<String>().apply { addAll(initialValues) }
            }
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                if (label.isNotBlank()) {
                    Text(label, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurface)
                }
                options.forEach { option ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        val isChecked = selected.contains(option.value)
                        Checkbox(
                            checked = isChecked,
                            onCheckedChange = { checked ->
                                if (checked) {
                                    if (!selected.contains(option.value)) selected.add(option.value)
                                } else {
                                    selected.remove(option.value)
                                }
                            }
                        )
                        Text(option.label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurface)
                    }
                }
            }
            return
        }

        val firstDefault = options.first().value
        var selectedValue by remember(componentId) {
            mutableStateOf(initialValues.firstOrNull()?.takeIf { candidate -> options.any { it.value == candidate } } ?: firstDefault)
        }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            if (label.isNotBlank()) {
                Text(label, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurface)
            }
            options.forEach { option ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = selectedValue == option.value,
                        onClick = { selectedValue = option.value }
                    )
                    Text(option.label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurface)
                }
            }
        }
    }

    @Composable
    fun RenderSliderComponent(component: JsonObject) {
        val componentId = component.getString("id")
        val label = sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Value" }
        val minValue = (component.getAsNumberOrNull("min") ?: 0.0).toFloat()
        val maxValueRaw = (component.getAsNumberOrNull("max") ?: 100.0).toFloat()
        val maxValue = if (maxValueRaw <= minValue) minValue + 1f else maxValueRaw
        val initialValue = readDynamicNumber(component.get("value"))?.coerceIn(minValue, maxValue) ?: minValue
        var sliderValue by remember(componentId) { mutableStateOf(initialValue) }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = "$label: ${formatSliderValue(sliderValue)}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Slider(
                value = sliderValue,
                onValueChange = { sliderValue = it },
                valueRange = minValue..maxValue
            )
        }
    }

    @Composable
    fun RenderDateTimeInputComponent(component: JsonObject) {
        val componentId = component.getString("id")
        val label = sanitizeDisplayText(readDynamicString(component.get("label"))).ifBlank { "Date/time" }
        val initialValue = readDynamicString(component.get("value"))
        val enableDate = component.get("enableDate")?.asBooleanOrNull() ?: true
        val enableTime = component.get("enableTime")?.asBooleanOrNull() ?: false
        val modeHint = when {
            enableDate && enableTime -> "Date + time (ISO 8601)"
            enableDate -> "Date (ISO 8601)"
            enableTime -> "Time (ISO 8601)"
            else -> "Text"
        }
        var value by remember(componentId) { mutableStateOf(initialValue) }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            OutlinedTextField(
                value = value,
                onValueChange = { value = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text(label) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text)
            )
            Text(
                text = modeHint,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
