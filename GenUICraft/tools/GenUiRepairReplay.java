import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.samsung.genuicraft.sdk.GenUiCompileOutcome;
import com.samsung.genuicraft.sdk.GenUiCompiler;
import com.samsung.genuicraft.sdk.GenUiDocument;
import com.samsung.genuicraft.sdk.GenUiRepairKind;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Runs the published AAR's actual repair code; no source text or model is supplied. */
public final class GenUiRepairReplay {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("Expected input.jsonl output.jsonl");
        Gson gson = new GsonBuilder().disableHtmlEscaping().serializeNulls().create();
        int count = 0;
        int accepted = 0;
        try (BufferedReader reader = Files.newBufferedReader(Path.of(args[0]), StandardCharsets.UTF_8);
             BufferedWriter writer = Files.newBufferedWriter(Path.of(args[1]), StandardCharsets.UTF_8,
                 StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                JsonObject input = JsonParser.parseString(line).getAsJsonObject();
                JsonObject output = new JsonObject();
                output.addProperty("model", input.get("model").getAsString());
                output.addProperty("id", input.get("id").getAsString());
                output.addProperty("model_calls", 0);
                output.addProperty("source_text_supplied_to_repair", false);
                output.addProperty("source_fallback_enabled", false);
                long start = System.nanoTime();
                try {
                    GenUiCompileOutcome result = GenUiCompiler.compileWithRepair(
                        input.get("raw_output").getAsString(), null, false, true);
                    if (result.getRepairKind() == GenUiRepairKind.SOURCE_TEXT_FALLBACK) {
                        throw new AssertionError("Unexpected source fallback");
                    }
                    GenUiDocument document = result.getDocument();
                    GenUiCompiler.compile(document.getExpress());
                    output.addProperty("repair_success", true);
                    output.addProperty("repair_kind", result.getRepairKind().name());
                    output.addProperty("repaired_express", document.getExpress());
                    output.addProperty("repaired_a2ui_json", document.getA2uiJson());
                    output.add("diagnostics", gson.toJsonTree(result.getDiagnostics()));
                    accepted++;
                } catch (IllegalArgumentException | IllegalStateException error) {
                    // Candidate failures are results. JVM/linkage/environment errors
                    // deliberately escape and fail the run instead of scoring as zero.
                    output.addProperty("repair_success", false);
                    output.addProperty("repair_kind", "REJECTED");
                    output.addProperty("exception_class", error.getClass().getName());
                    output.addProperty("repair_error", error.getMessage());
                }
                output.addProperty("host_repair_ms", (System.nanoTime() - start) / 1_000_000.0);
                writer.write(gson.toJson(output));
                writer.newLine();
                writer.flush();
                count++;
                if (count % 10 == 0) System.out.println("SDK repair: " + count + " attempted; " + accepted + " accepted");
            }
        }
        System.out.println("SDK repair complete: " + accepted + "/" + count + " accepted; no fallback or model calls");
    }
}
