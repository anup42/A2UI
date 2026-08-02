package com.samsung.genuicraft.renderer.flat.runtime

import com.samsung.genuicraft.renderer.*

import java.util.Locale
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.model.*

/**
 * Real evaluation for the `validateForm` action.
 *
 * `FlatActionRuntime` previously wrote a hardcoded
 * `{"valid": true, "errors": {}}` regardless of input, while
 * `assets/pipeline_prompts/genui_gen_a2ui_express_v1.md` instructs the model to emit
 * `validateForm`. Every generated form therefore reported success.
 *
 * Rules are derived from the input controls themselves, so **no IR change is
 * required**: a control that declares `required`, `pattern`, `minLength`,
 * `maxLength`, `min`, `max` or an email/number `inputType` becomes a rule keyed
 * by the state path it is bound to.
 */
internal object FlatFormValidation {

    private val EMAIL_REGEX = Regex("^[^@\\s]+@[^@\\s.]+\\.[^@\\s]{2,}$")

    internal data class Rule(
        val statePath: String,
        val label: String,
        val required: Boolean,
        val pattern: String?,
        val minLength: Int?,
        val maxLength: Int?,
        val min: Double?,
        val max: Double?,
        val kind: String?
    )

    private val INPUT_TYPES = setOf(
        "textfield",
        "checkbox",
        "choicepicker",
        "slider",
        "datetimeinput"
    )

    private fun intProp(props: Map<String, Any?>, vararg keys: String): Int? {
        keys.forEach { key ->
            val value = props[key] ?: return@forEach
            (value as? Number)?.let { return it.toInt() }
            value.toString().trim().toIntOrNull()?.let { return it }
        }
        return null
    }

    private fun doubleProp(props: Map<String, Any?>, vararg keys: String): Double? {
        keys.forEach { key ->
            val value = props[key] ?: return@forEach
            (value as? Number)?.let { return it.toDouble() }
            value.toString().trim().toDoubleOrNull()?.let { return it }
        }
        return null
    }

    private fun boolProp(props: Map<String, Any?>, vararg keys: String): Boolean {
        keys.forEach { key ->
            val value = props[key] ?: return@forEach
            (value as? Boolean)?.let { return it }
            if (value.toString().equals("true", ignoreCase = true)) return true
        }
        return false
    }

    /** Collects one rule per validated, state-bound input control. */
    fun collectRules(elements: Map<String, FlatElement>): List<Rule> {
        val rules = mutableListOf<Rule>()
        elements.forEach { (elementId, element) ->
            val type = element.type.trim().lowercase(Locale.US)
            if (type !in INPUT_TYPES) return@forEach
            val props = element.props
            val statePath = bindPathFromValueExpression(props["value"], null)
                ?: props["statePath"]?.toString()
            if (statePath.isNullOrBlank()) return@forEach

            val pattern = (props["pattern"] ?: props["regex"] ?: props["validationRegexp"])
                ?.toString()
                ?.takeIf { it.isNotBlank() }
            val kind = (props["inputType"] ?: props["variant"] ?: props["format"])
                ?.toString()
                ?.trim()
                ?.lowercase(Locale.US)
            val required = boolProp(props, "required", "isRequired")
            val minLength = intProp(props, "minLength", "minlength")
            val maxLength = intProp(props, "maxLength", "maxlength")
            val min = doubleProp(props, "min", "minValue")
            val max = doubleProp(props, "max", "maxValue")

            val validated = required ||
                pattern != null ||
                minLength != null ||
                maxLength != null ||
                min != null ||
                max != null ||
                kind == "email" ||
                kind == "number"
            if (!validated) return@forEach

            rules += Rule(
                statePath = normalizePointer(statePath),
                label = props["label"]?.toString()?.takeIf { it.isNotBlank() } ?: elementId,
                required = required,
                pattern = pattern,
                minLength = minLength,
                maxLength = maxLength,
                min = min,
                max = max,
                kind = kind
            )
        }
        return rules
    }

    private fun isEmptyValue(value: Any?): Boolean = when (value) {
        null -> true
        is String -> value.isBlank()
        is Collection<*> -> value.isEmpty()
        is Map<*, *> -> value.isEmpty()
        is Boolean -> !value
        else -> false
    }

    private fun evaluate(rule: Rule, value: Any?): String? {
        if (isEmptyValue(value)) {
            // Only `required` fails on an empty value; the other constraints
            // describe the shape of a value that is present.
            return if (rule.required) "${rule.label} is required." else null
        }
        val text = when (value) {
            is String -> value
            is Number -> value.toString().removeSuffix(".0")
            else -> value.toString()
        }
        rule.minLength?.let { limit ->
            if (text.length < limit) return "${rule.label} must be at least $limit characters."
        }
        rule.maxLength?.let { limit ->
            if (text.length > limit) return "${rule.label} must be at most $limit characters."
        }
        rule.pattern?.let { pattern ->
            val matches = runCatching { Regex(pattern).matches(text) }.getOrNull()
            // An unparseable pattern is a generation defect, not a user error;
            // failing the field would be misleading, so it is ignored.
            if (matches == false) return "${rule.label} is not in the expected format."
        }
        if (rule.kind == "email" && !EMAIL_REGEX.matches(text)) {
            return "${rule.label} must be a valid email address."
        }
        if (rule.kind == "number" || rule.min != null || rule.max != null) {
            val numeric = (value as? Number)?.toDouble() ?: text.trim().toDoubleOrNull()
            if (numeric == null) {
                if (rule.kind == "number") return "${rule.label} must be a number."
            } else {
                rule.min?.let { if (numeric < it) return "${rule.label} must be at least $it." }
                rule.max?.let { if (numeric > it) return "${rule.label} must be at most $it." }
            }
        }
        return null
    }

    /**
     * Result shape is unchanged from the stub — `{"valid": Boolean, "errors":
     * Map<statePath, message>}` — so specs reading `/formValidation/valid`
     * keep working. With no rules the result is `valid = true`, which is why
     * this is safe to enable for existing IR.
     */
    fun validate(
        elements: Map<String, FlatElement>,
        state: Map<String, Any?>
    ): Map<String, Any?> {
        val errors = linkedMapOf<String, String>()
        collectRules(elements).forEach { rule ->
            val value = FlatSpecParser.getAtPath(state, rule.statePath)
            evaluate(rule, value)?.let { message -> errors[rule.statePath] = message }
        }
        return mapOf("valid" to errors.isEmpty(), "errors" to errors)
    }
}
