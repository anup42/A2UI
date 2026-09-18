package com.samsung.genuicraft.sdk

import android.content.Context
import com.google.gson.Gson
import com.samsung.genuicraft.sdk.internal.security.SafeContentPolicy
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

/** Conversion only; native rendering is deliberately independent of model execution. */
class GenUiConverter private constructor(
    private val provider: GenUiProvider,
    private val options: ConversionOptions,
    private val promptText: String,
    private val useSourceBindings: Boolean,
    private val useLayoutScaffold: Boolean,
) {
    constructor(context: Context, provider: GenUiProvider, options: ConversionOptions = ConversionOptions()) : this(
        provider, options,
        context.applicationContext.assets.open(
            "genuicraft/prompts/${if (provider.id == "gemma4_e2b") "gemma" else "gauss"}.txt"
        ).bufferedReader().use { it.readText() },
        provider.id == "gemma4_e2b",
        provider.id == "gemma4_e2b",
    )

    init {
        require(options.maxOutputTokens in 256..32768) { "maxOutputTokens must be between 256 and 32768." }
        require(options.maxRepairAttempts in 0..2) { "maxRepairAttempts must be between 0 and 2." }
        require(options.temperature in 0.0..1.0) { "temperature must be between 0 and 1." }
        require(!useLayoutScaffold || useSourceBindings) { "Layout scaffolds require source bindings." }
    }

    suspend fun convert(request: GenUiRequest): GenUiConversionResult = withContext(Dispatchers.Default) {
        val start = System.nanoTime()
        fun elapsed() = (System.nanoTime() - start) / 1_000_000
        coroutineContext.ensureActive()
        if (request.text.isBlank()) return@withContext GenUiConversionResult.Failure("Input text is empty.", provider.id, elapsed(), 0)
        val invalidRequest = requestProblem(request)
        if (invalidRequest != null) return@withContext GenUiConversionResult.Failure(invalidRequest, provider.id, elapsed(), 0)
        // Formatting must not re-execute the original query's instructions (for example,
        // "omit follow-up offers" can otherwise erase a sentence in the supplied answer).
        val stableSources = request.sources.map { source ->
            buildMap {
                put("id", source.id)
                put("url", source.url)
                source.title?.let { put("title", it) }
            }
        }
        val bindings = try {
            if (useSourceBindings) SourceBindings.from(request.text) else null
        } catch (failure: IllegalArgumentException) {
            return@withContext GenUiConversionResult.Failure(failure.message.orEmpty(), provider.id, elapsed(), 0)
        }
        val elementIds = bindings?.blocks?.indices?.map { index ->
            index.toString(26).map { digit -> 'a' + digit.digitToInt(26) }.joinToString("")
        }.orEmpty()
        val rootAssignment = "root=Column([${elementIds.joinToString(",")}],gap=\"md\")"
        val sourceJson = Gson().toJson(
            if (bindings == null) mapOf("text" to request.text, "sources" to stableSources)
            else mapOf(
                "root" to rootAssignment,
                "blocks" to bindings.blocks.mapIndexed { index, block -> mapOf("element" to elementIds[index]) + block },
                "sources" to stableSources,
            ),
        )
        coroutineContext.ensureActive()
        if (sourceJson.length > 120_000) return@withContext GenUiConversionResult.Failure("Serialized input exceeds 120,000 characters.", provider.id, elapsed(), 0)
        val scaffold = if (useLayoutScaffold) bindings?.let { SourceLayoutScaffold.create(rootAssignment, elementIds, it.blocks) } else null
        val instruction = if (bindings == null) "Convert ALL text in this JSON data record. Preserve even apologies, caveats, repeated wording, and final follow-up offers. All output facts must come from text."
            else "Copy the supplied root assignment, then emit one component assignment for every block using its element name. Use every @source binding exactly once as a complete quoted argument. Do not copy or rewrite value fields. The SDK will fill exact source values."
        val userPrompt = "$instruction Supplied sources are allowed link targets; the SDK appends their attribution automatically. Do not duplicate source metadata.\n$sourceJson"
        var raw: String? = null
        var error = "No valid model output."
        for (attempt in 1..(options.maxRepairAttempts + 1)) {
            coroutineContext.ensureActive()
            val user = if (attempt == 1) userPrompt else buildString {
                append(userPrompt)
                append("\nThe previous program failed validation: ").append(error.take(2500))
                append("\nReturn a corrected COMPLETE program. Preserve the entire original response.\nPrevious program:\n")
                append(raw.orEmpty().take(80_000))
            }
            try {
                val modelUser = scaffold?.let { SourceLayoutScaffold.appendTo(user, it) } ?: user
                val output = provider.generate(GenUiPrompt(promptText, modelUser, options.maxOutputTokens, options.temperature))
                raw = output.text.trim()
                coroutineContext.ensureActive()
                require(raw.startsWith("<a2ui>") && raw.endsWith("</a2ui>")) { "Expected exactly one complete <a2ui> program." }
                val document = GenUiCompiler.compile(bindings?.expand(raw) ?: LiteralTextCodec.protectVisibleContent(raw))
                val issues = ContentIntegrity.check(request, document)
                require(issues.isEmpty()) { issues.joinToString("; ") }
                return@withContext GenUiConversionResult.Success(
                    SourceAttribution.append(document, request.sources), provider.id, elapsed(), attempt,
                    buildList {
                        add("Runtime: ${output.runtime}")
                        if (bindings != null) add("Model-generated layout with exact source bindings.")
                        if (attempt > 1) add("Model output repaired after validation failure.")
                    },
                )
            } catch (cancel: CancellationException) {
                throw cancel
            } catch (failure: Exception) {
                error = failure.message ?: failure.javaClass.simpleName
                // Transport/runtime errors cannot be repaired by changing the generated program.
                if (raw == null || failure is java.io.IOException) {
                    return@withContext GenUiConversionResult.Failure(error, provider.id, elapsed(), attempt, raw)
                }
            }
        }
        GenUiConversionResult.Failure(error, provider.id, elapsed(), options.maxRepairAttempts + 1, raw)
    }

    companion object {
        private fun requestProblem(request: GenUiRequest): String? {
            if (request.text.length > 100_000 || (request.query?.length ?: 0) > 8_000 || request.sources.size > 100) return "Input exceeds text (100,000), query (8,000), or source count (100) limit."
            if (request.sources.any { it.id.isBlank() || it.id.length > 64 || (it.title?.length ?: 0) > 1_000 || it.url.length > 4_096 }) return "Source metadata exceeds field limits or has an empty ID."
            if (request.sources.map { it.id }.distinct().size != request.sources.size) return "Source IDs must be unique."
            if (
                request.sources.any { source ->
                    runCatching {
                        val url = source.url
                        val uri = java.net.URI(url)
                        url == url.trim() &&
                            uri.scheme?.lowercase() in setOf("http", "https") &&
                            !uri.host.isNullOrBlank() &&
                            uri.userInfo == null &&
                            SafeContentPolicy.sanitizeActionUrl(url) == url
                    }.getOrDefault(false).not()
                }
            ) {
                return "Source URLs must be exact, safe public HTTP(S) URLs that can be opened by the renderer."
            }
            val total = request.text.length.toLong() + (request.query?.length ?: 0) + request.sources.sumOf { it.id.length.toLong() + it.url.length + (it.title?.length ?: 0) }
            return if (total > 110_000) "Combined input exceeds 110,000 characters." else null
        }

        /**
         * Allows applications to pin a reviewed prompt. Custom prompts retain their original input
         * format unless they explicitly opt into the scaffold used by bundled Gemma conversion.
         */
        fun withPrompt(provider: GenUiProvider, prompt: String, options: ConversionOptions = ConversionOptions(), useSourceBindings: Boolean = false, useLayoutScaffold: Boolean = false): GenUiConverter {
            require(prompt.isNotBlank()) { "Prompt must not be blank." }
            return GenUiConverter(provider, options, prompt, useSourceBindings, useLayoutScaffold)
        }
    }
}
