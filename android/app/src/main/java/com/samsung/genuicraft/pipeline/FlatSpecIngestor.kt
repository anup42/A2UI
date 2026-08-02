package com.samsung.genuicraft.pipeline

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.samsung.genuicraft.renderer.FlatDiagnostic
import com.samsung.genuicraft.renderer.describeFlatSpecRouting
import com.samsung.genuicraft.renderer.flat.model.FlatSpec
import com.samsung.genuicraft.renderer.flat.parse.FlatSpecParser

enum class FlatSpecIngestMode {
    COMPATIBILITY,
    STRICT
}

internal data class FlatRoutingComparison(
    val changed: Boolean,
    val rawRoutes: Map<Int, String>,
    val canonicalRoutes: Map<Int, String>
)

internal sealed interface FlatSpecIngestResult {
    data class CanonicalFlatSpec(
        val canonicalJson: JsonObject,
        val canonicalSpec: FlatSpec,
        /** Compatibility renders the raw valid spec once; strict renders canonical only. */
        val renderSpec: FlatSpec,
        val warnings: List<String>,
        val tableDiagnostics: FlatSpecContract.TableDiagnostics,
        val routingComparison: FlatRoutingComparison? = null,
        val sourceFormat: GenUiIrFormat = GenUiIrFormat.FLAT_SPEC_V1
    ) : FlatSpecIngestResult

    data class GenuineLegacyPayload(
        val payload: JsonElement,
        val migratedFlatSpec: JsonObject?,
        val warnings: List<String>,
        val tableDiagnostics: FlatSpecContract.TableDiagnostics,
        val sourceFormat: GenUiIrFormat = GenUiIrFormat.FLAT_SPEC_V1
    ) : FlatSpecIngestResult

    data class RejectedPayload(
        val diagnostics: List<FlatDiagnostic>
    ) : FlatSpecIngestResult
}

/** The only path from decoded JSON into the canonical flat-renderer contract. */
internal object FlatSpecIngestor {
    @Volatile
    var defaultMode: FlatSpecIngestMode = FlatSpecIngestMode.STRICT

    fun ingest(
        payload: JsonElement?,
        mode: FlatSpecIngestMode = defaultMode
    ): FlatSpecIngestResult {
        if (payload == null) return rejected("Flat-spec payload is null.")
        val compatibility = mode == FlatSpecIngestMode.COMPATIBILITY
        val detectedFormat = runCatching { GenUiIrCodec.detect(payload) }.getOrNull()
        val decoded = runCatching {
            detectedFormat?.let { GenUiIrCodec.decode(payload) }
            // A null detection retains the historical legacy-message migration path.
        }.getOrElse { error ->
            return rejected(error.message ?: "IR format decoding failed.")
        }
        val contractPayload = decoded?.flatSpec ?: payload
        val sourceFormat = decoded?.sourceFormat ?: GenUiIrFormat.FLAT_SPEC_V1
        val result = FlatSpecContract.coerceAndValidate(
            contractPayload,
            compatibilityHeaderInference = compatibility
        )
        if (!result.isValid || result.spec == null) {
            return rejected(result.error ?: "Flat-spec payload failed validation.")
        }
        if (result.convertedFromLegacy) {
            return FlatSpecIngestResult.GenuineLegacyPayload(
                payload = payload,
                migratedFlatSpec = result.spec,
                warnings = result.warnings,
                tableDiagnostics = result.tableDiagnostics,
                sourceFormat = sourceFormat
            )
        }

        val canonicalSpec = FlatSpecParser.parse(result.spec)
            ?: return rejected("Canonical flat-spec could not be parsed by the renderer.")
        val rawSpec = contractPayload.takeIf(FlatSpecContract::looksLikeFlatSpec)?.let(FlatSpecParser::parse)
        val comparison = if (compatibility && rawSpec != null) {
            compareRoutes(rawSpec, canonicalSpec)
        } else {
            null
        }
        val comparisonWarnings = if (comparison?.changed == true) {
            listOf("Compatibility ingest observed raw/canonical routing differences at 360dp or 800dp.")
        } else {
            emptyList()
        }
        return FlatSpecIngestResult.CanonicalFlatSpec(
            canonicalJson = result.spec,
            canonicalSpec = canonicalSpec,
            renderSpec = if (compatibility) rawSpec ?: canonicalSpec else canonicalSpec,
            warnings = result.warnings + comparisonWarnings,
            tableDiagnostics = result.tableDiagnostics,
            routingComparison = comparison,
            sourceFormat = sourceFormat
        )
    }

    private fun compareRoutes(raw: FlatSpec, canonical: FlatSpec): FlatRoutingComparison {
        val widths = listOf(360, 800)
        val rawRoutes = widths.associateWith { width -> describeFlatSpecRouting(raw, width) }
        val canonicalRoutes = widths.associateWith { width -> describeFlatSpecRouting(canonical, width) }
        return FlatRoutingComparison(
            changed = rawRoutes != canonicalRoutes,
            rawRoutes = rawRoutes,
            canonicalRoutes = canonicalRoutes
        )
    }

    private fun rejected(message: String): FlatSpecIngestResult.RejectedPayload =
        FlatSpecIngestResult.RejectedPayload(
            diagnostics = listOf(
                FlatDiagnostic(
                    code = FlatDiagnostic.Code.INVALID_PAYLOAD,
                    severity = FlatDiagnostic.Severity.ERROR,
                    message = message
                )
            )
        )
}
