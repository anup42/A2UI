package com.samsung.genuicraft.sdk

import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
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
     * bounded syntax normalization that never rewrites visible values or graph structure. When those
     * paths fail and [sourceText] is supplied, a deterministic source-bound A2UI document is
     * produced from typed source blocks and checked by the SDK's content-integrity gate. The
     * returned [GenUiRepairKind] keeps that fallback distinct from a repaired model program.
     */
    @JvmStatic
    @JvmOverloads
    fun compileWithRepair(input: String, sourceText: String? = null): GenUiCompileOutcome {
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
            if (integrity.isEmpty()) return GenUiCompileOutcome(document, GenUiRepairKind.NONE)
            diagnostics += "Strictly compiled generated document failed mechanical source integrity: ${integrity.joinToString("; ")}"
        } else {
            strict.exceptionOrNull()?.message?.let { diagnostics += "Strict compile rejected output: $it" }
        }

        val structural = A2uiExpressOutputRepair.repair(input)
        if (structural != null) {
            val compiled = runCatching { compile(structural.express) }
            if (compiled.isSuccess) {
                val document = compiled.getOrThrow()
                val integrity = sourceText?.let { ContentIntegrity.check(GenUiRequest(it), document) }.orEmpty()
                if (integrity.isEmpty()) {
                    return GenUiCompileOutcome(
                        document = document,
                        repairKind = GenUiRepairKind.STRUCTURAL,
                        diagnostics = diagnostics + structural.changes,
                    )
                }
                diagnostics += structural.changes
                diagnostics += "Generated document failed mechanical source integrity: ${integrity.joinToString("; ")}"
            } else {
                diagnostics += structural.changes
                diagnostics += "Normalized candidate failed full compilation: ${compiled.exceptionOrNull()?.message}"
            }
        }

        require(sourceText != null) {
            (diagnostics + "No bounded structural repair produced a valid document.").joinToString(" ")
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
