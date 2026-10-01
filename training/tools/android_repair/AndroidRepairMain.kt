package com.samsung.genuicraft.sdk

import com.google.gson.Gson
import com.google.gson.JsonParser

/** Batch bridge for the trained profile in GenUiSession: no source reconstruction. */
fun main() {
    val gson = Gson()
    val output = System.out.bufferedWriter(Charsets.UTF_8)
    System.`in`.bufferedReader(Charsets.UTF_8).forEachLine { line ->
        val request = JsonParser.parseString(line).asJsonObject
        val started = System.nanoTime()
        val result = linkedMapOf<String, Any?>("index" to request.get("index").asInt)
        try {
            val compiled = GenUiCompiler.compileWithRepair(
                input = request.get("generated_text").asString.trim(),
                sourceText = null,
                allowSourceTextFallback = false,
                allowGeneratedDslRepair = true,
            )
            result["success"] = true
            result["repair_kind"] = compiled.repairKind.name
            result["express"] = compiled.document.express
            result["a2ui_json"] = compiled.document.a2uiJson
            result["diagnostics"] = compiled.diagnostics
        } catch (error: Exception) {
            result["success"] = false
            result["repair_kind"] = "REJECTED"
            result["error"] = "${error.javaClass.simpleName}: ${error.message}"
        }
        result["repair_seconds"] = (System.nanoTime() - started) / 1_000_000_000.0
        output.write(gson.toJson(result))
        output.newLine()
        output.flush()
    }
}
