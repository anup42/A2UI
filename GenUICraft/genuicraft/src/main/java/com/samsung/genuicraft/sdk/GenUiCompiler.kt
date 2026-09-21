package com.samsung.genuicraft.sdk

import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressGeneralRepair
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressOutputRepair
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiWireCodec
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec

/** Deterministically compiles strict A2UI Express or A2UI v0.9 JSON. */
object GenUiCompiler {
    @JvmStatic
    fun compile(input: String): GenUiDocument {
        val content = input.trim()
        require(content.isNotEmpty()) {
            "GenUI input is empty; expected one <a2ui> document or A2UI v0.9 JSON."
        }
        require(content.startsWith("<a2ui>") || content.startsWith("{") || content.startsWith("[")) {
            "GenUI input contains prose or an unsupported prefix; expected <a2ui>, {, or [."
        }
        if (content.startsWith("<a2ui>")) {
            require(content.endsWith("</a2ui>")) {
                "GenUI Express input must end exactly with </a2ui>; trailing prose or an incomplete document is not allowed."
            }
        }

        return try {
            val decoded = GenUiIrCodec.decode(JsonPrimitive(content))
            GenUiDocument(
                express = A2uiExpressCodec.encode(decoded.canonicalGraph),
                a2uiJson = A2uiWireCodec.encode(decoded.canonicalGraph).toString(),
            )
        } catch (error: RuntimeException) {
            throw IllegalArgumentException(
                "Invalid GenUI ${inputKind(content)}: ${error.message ?: "validation or compilation failed"}",
                error,
            )
        }
    }

    /**
     * Opt-in recovery for generated Express output.
     *
     * Strict compilation remains the default API. This method first tries the exact input, then
     * bounded syntax normalization. Set [allowGeneratedDslRepair] only for explicit model-output
     * salvage experiments; these broader repairs can recover renderable structure without proving
     * source fidelity. When [sourceText] is supplied, every generated candidate must pass the SDK's
     * content-integrity gate. If all generated candidates fail and [allowSourceTextFallback] is true,
     * a deterministic source-bound A2UI document is produced from typed source blocks. The returned
     * [GenUiRepairKind] keeps DSL salvage and source fallback distinct.
     */
    @JvmStatic
    @JvmOverloads
    fun compileWithRepair(
        input: String,
        sourceText: String? = null,
        allowSourceTextFallback: Boolean = true,
        allowGeneratedDslRepair: Boolean = false,
    ): GenUiCompileOutcome {
        require(input.length <= 120_000) { "Generated input exceeds the 120,000-character recovery limit." }
        require(sourceText == null || sourceText.length <= 100_000) {
            "Source text exceeds the 100,000-character recovery limit."
        }
        require(sourceText == null || sourceText.isNotBlank()) {
            "Source text must be non-blank when supplied for recovery."
        }
        val diagnostics = mutableListOf<String>()
        val strict = runCatching { compile(input) }
        if (strict.isSuccess) {
            val document = strict.getOrThrow()
            val integrity = sourceText?.let { ContentIntegrity.check(GenUiRequest(it), document) }.orEmpty()
            val recoverBindings = allowGeneratedDslRepair && A2uiExpressGeneralRepair.hasUnresolvedDataBindings(document.express)
            if (integrity.isEmpty() && !recoverBindings) return GenUiCompileOutcome(document, GenUiRepairKind.NONE)
            if (recoverBindings) diagnostics += "Generated table/chart data bindings were unresolved; recovering generated data and readable fragments."
            if (integrity.isNotEmpty()) diagnostics += "Strictly compiled generated document failed mechanical source integrity: ${integrity.joinToString("; ")}"
        } else {
            strict.exceptionOrNull()?.message?.let { diagnostics += "Strict compile rejected output: $it" }
        }

        A2uiExpressOutputRepair.repairs(input, allowGeneratedDslRepair).forEachIndexed { index, repair ->
            val compiled = runCatching { compile(repair.express) }
            if (compiled.isSuccess) {
                val document = compiled.getOrThrow()
                val integrity = sourceText?.let { ContentIntegrity.check(GenUiRequest(it), document) }.orEmpty()
                val recoverBindings = allowGeneratedDslRepair && A2uiExpressGeneralRepair.hasUnresolvedDataBindings(document.express)
                if (integrity.isEmpty() && !recoverBindings) {
                    return GenUiCompileOutcome(
                        document = document,
                        repairKind = when (repair.kind) {
                            A2uiExpressOutputRepair.Kind.STRUCTURAL -> GenUiRepairKind.STRUCTURAL
                            A2uiExpressOutputRepair.Kind.GENERATED_DSL -> GenUiRepairKind.GENERATED_DSL_REPAIR
                        },
                        diagnostics = diagnostics + repair.changes,
                    )
                }
                diagnostics += "Generated repair candidate ${index + 1} (${repair.kind}) applied: ${repair.changes.joinToString("; ")}"
                if (recoverBindings) diagnostics += "Candidate still had unresolved data bindings; continuing generated-content recovery."
                if (integrity.isNotEmpty()) diagnostics += "Generated repair candidate ${index + 1} failed mechanical source integrity: ${integrity.joinToString("; ")}"
            } else {
                diagnostics += "Generated repair candidate ${index + 1} (${repair.kind}) applied: ${repair.changes.joinToString("; ")}"
                diagnostics += "Generated repair candidate ${index + 1} failed full compilation: ${compiled.exceptionOrNull()?.message}"
            }
        }

        require(allowSourceTextFallback) {
            val reason = if (sourceText == null) {
                "No generated DSL repair produced a valid document; source fallback was disabled."
            } else {
                "No source-faithful generated DSL repair produced a valid document; source fallback was disabled."
            }
            (diagnostics + reason)
                .joinToString(" ")
        }
        require(sourceText != null) {
            (diagnostics + "No generated DSL repair produced a valid document.").joinToString(" ")
        }
        val fallback = SourceTextFallback.compile(sourceText)
        val fallbackIssues = ContentIntegrity.check(GenUiRequest(sourceText), fallback)
        require(fallbackIssues.isEmpty()) {
            "Deterministic source fallback failed integrity: ${fallbackIssues.joinToString("; ")}"
        }
        return GenUiCompileOutcome(
            document = fallback,
            repairKind = GenUiRepairKind.SOURCE_TEXT_FALLBACK,
            diagnostics = diagnostics + "Generated output was rejected; rebuilt typed A2UI from exact source blocks.",
        )
    }

    private fun inputKind(content: String): String = when {
        content.startsWith("<a2ui>") -> "Express"
        else -> "A2UI v0.9 JSON"
    }
}
