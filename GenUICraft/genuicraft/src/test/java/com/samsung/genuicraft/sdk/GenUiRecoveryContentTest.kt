package com.samsung.genuicraft.sdk

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenUiRecoveryContentTest {

    @Test
    fun capturedBxp003RetainsAllThreeTrainRowsAndSurvivingText() {
        val outcome = recoverCaptured(
            resource = "/recovery/BXP-003.raw.express",
            expectedSha256 = "c2a35a2577e2eb329eb3c3d4d0c1409c9e462253f718214699f3f350d92d3dd9",
            reportId = "BXP-003",
        )
        val graph = graph(outcome)

        assertTableRows(
            graph = graph,
            marker = "Shatabdi Express (120007)",
            expected = listOf(
                listOf("Shatabdi Express (120007)", "KSR Bengaluru (BC)", "10:35–10:45", "2 h15 m to 2h225 m", "CC, EC"),
                listOf("Rajya Rani Rani Express (20666)", "KSR Bengaluru (BC)", "11:30", "2h30m", "CC, 2 S GN"),
                listOf("Wodeyar Express (12461)", "KSR Bengaluru (BC)", "15:5", "2h30m", "CC,2S GN"),
            ),
        )
        assertTextComponents(
            graph,
            "A few timetable notes",
            "If you want",
            "can also provide same comparison with arrival times and days of operation included",
        )
        val values = leafStrings(graph)
        assertFalse("Loose damaged-call fragments must not create an extra recovery section", values.contains(
            "Shatabdi Express also shows slight variation in duration, from about 2 h 15 m to 2 h 25 m.",
        ))
        assertFalse("The diagnostic recovery heading must never be rendered", "Additional recovered text" in values)
        assertFalse("Corrupted layout metadata is not answer text", " между" in values)
    }

    @Test
    fun capturedBxp001RetainsAllThreeForecastRowsAndSurvivingText() {
        val outcome = recoverCaptured(
            resource = "/recovery/BXP-001.raw.express",
            expectedSha256 = "9f1c75d8fa32efed9b35cc1271e4636fbfa84d7a4d26a9bb74d9a560fb9494fe",
            reportId = "BXP-001",
        )
        val graph = graph(outcome)

        assertTableRows(
            graph = graph,
            marker = "Wed, Sep 9",
            expected = listOf(
                listOf("Wed, Sep 9", "31", "20", "57%"),
                listOf("Thu, Sep 10", "30", "211", "55%"),
                listOf("Fri, Sep 1", "30", "21", "55%"),
            ),
        )
        assertTextComponents(
            graph,
            "AccuWeather’s Bengaluru forecast shows cloudy or sunny-interval conditions through the next three days with the highest rain chance on Wednesday and continued shower/thunder/storm potential on Thursday and Friday.",
            "Carry an umbrella or light rain jacket, especially for afternoon and evening plans.",
            "Plan outdoor activities earlier in the day, when conditions are likely to stay drier and slightly cooler.",
        )
    }

    @Test
    fun aggregateRecoveryKeepsInlineTableGeneratedStateAndProseTogether() {
        val raw = """
            <a2ui>
            ${'$'}/={generated_rows:[{name:"State alpha",value:"A-17"},{name:"State beta",value:"B-29"}]}
            root=Column([inline,bound,note,broken],gap="not-a-gap")
            inline=Table(columns=["Item","Value"],rows=[["Inline alpha","I-11"],["Inline beta","I-22"]],preferredPresentation="table")
            bound=Table(columns=["name","value"],statePath="/generated_rows",preferredPresentation="cards")
            note=Text("All generated sections must survive.",variant="body")
            broken=Card(children=[missing]
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))

        assertTableRows(
            graph,
            marker = "State alpha",
            expected = listOf(
                listOf("State alpha", "A-17"),
                listOf("State beta", "B-29"),
            ),
        )
        assertTableRows(
            graph,
            marker = "Inline alpha",
            expected = listOf(
                listOf("Inline alpha", "I-11"),
                listOf("Inline beta", "I-22"),
            ),
        )
        assertTextComponents(graph, "All generated sections must survive.")
    }

    @Test
    fun repeatedGeneratedStateRowsAreNotDeduplicated() {
        val raw = """
            <a2ui>
            ${'$'}/={events:[{label:"Same event",time:"09:00"},{label:"Same event",time:"09:00"},{label:"Same event",time:"09:00"}]}
            root=Column([events_table,note,broken])
            events_table=Table(columns=["label","time"],statePath="/events",preferredPresentation="cards")
            note=Text("Duplicate rows are intentional.")
            broken=Row([missing]
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))

        assertTableRows(
            graph,
            marker = "Same event",
            expected = List(3) { listOf("Same event", "09:00") },
        )
        assertTextComponents(graph, "Duplicate rows are intentional.")
    }

    @Test
    fun generatedStateTargetSubstringRoutesFlightOptionsToFlightCards() {
        val raw = """
            <a2ui>
            ${'$'}/={flight_options_schedule:[{airline:"Air India Express",departure:"07:15 PM",arrival:"09:55 PM",duration:"2h 40m",stops:"Non-stop",fare:"₹6,150"}]}
            root=Column([broken])
            broken=Card(children=[missing]
        """.trimIndent()
        val recovered = graph(recover(raw))
        val flightTable = recovered.getAsJsonObject("elements").entrySet()
            .map { it.value.asJsonObject }
            .single { element ->
                element.get("type")?.asString == "Table" &&
                    leafStrings(element).contains("Air India Express")
            }
        val props = flightTable.getAsJsonObject("props")

        assertEquals("flight", props.get("domain").asString)
        assertEquals("cards", props.get("preferredPresentation").asString)
        assertTableRows(recovered, "Air India Express", listOf(
            listOf("Air India Express", "07:15 PM", "09:55 PM", "2h 40m", "Non-stop", "₹6,150"),
        ))
    }

    @Test
    fun flightColumnsRouteGenericGeneratedStateToFlightCards() {
        val raw = """
            <a2ui>
            ${'$'}/={results:[{carrier:"IndiGo",depart:"10:00",destination:"LKO",fare:"₹7,450"}]}
            root=Column([broken])
            broken=Row([missing]
        """.trimIndent()
        val recovered = graph(recover(raw))
        val table = recovered.getAsJsonObject("elements").entrySet()
            .map { it.value.asJsonObject }
            .single { it.get("type")?.asString == "Table" }

        assertEquals("flight", table.getAsJsonObject("props").get("domain").asString)
    }

    @Test
    fun danglingBoundTableStillRecoversEveryCompleteStateRow() {
        val raw = """
            <a2ui>
            ${'$'}/={routes:[{city:"Delhi",gate:"A1"},{city:"Mumbai",gate:"B2"},{city:"Chennai",gate:"C3"}]}
            root=Column([routes_table,note])
            routes_table=Table(columns=["city","gate"],statePath="/routes"
            note=Text("Use the generated route data.")
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))

        assertTableRows(
            graph,
            marker = "Delhi",
            expected = listOf(
                listOf("Delhi", "A1"),
                listOf("Mumbai", "B2"),
                listOf("Chennai", "C3"),
            ),
        )
        assertTextComponents(graph, "Use the generated route data.")
    }

    @Test
    fun truncatedStateKeepsCompleteRowsWithoutCompletingTheBrokenString() {
        val raw = """
            <a2ui>
            ${'$'}/={routes:[{city:"Delhi",gate:"A1"},{city:"Mumbai",gate:"B2"},{city:"Kolk
        """.trimIndent()
        val graph = graph(recover(raw))

        assertTableRows(
            graph,
            marker = "Delhi",
            expected = listOf(
                listOf("Delhi", "A1"),
                listOf("Mumbai", "B2"),
            ),
        )
        val values = leafStrings(graph)
        assertFalse("Recovery must not complete or retain an unterminated string", values.any { "Kolk" in it })
    }

    @Test
    fun partialStateReaderResumesAfterMalformedMiddleField() {
        val raw = """
            <a2ui>
            ${'$'}/={summary:{before:"alpha",broken:???,count:7,after:"omega"}}
            root=Column([summary_table,broken])
            summary_table=Table(columns=["Field","Value"],statePath="/summary")
            broken=Card(children=[missing]
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        val values = leafStrings(graph)

        listOf("alpha", "7", "omega").forEach { expected ->
            assertTrue("Missing complete state value after partial recovery: $expected", expected in values)
        }
        assertFalse("Malformed state bytes must not become visible content", "???" in values)
        val rows = tableRows(graph).firstOrNull { table -> leafStrings(table).contains("alpha") }
            ?: error("No recovered summary table contains the field before the malformed value")
        assertEquals("Only the three complete fields should become rows", 3, rows.size())
    }

    @Test
    fun syntacticallyValidButDanglingBindingRecoversRowsInsteadOfAnEmptyTable() {
        val raw = """
            <a2ui>
            ${'$'}/={data:[{name:"One",value:13},{name:"Two",value:27}]}
            root=Column([table,note])
            table=Table(columns=["name","value"],statePath="/data_data")
            note=Text("The note must not hide missing table data.")
            </a2ui>
        """.trimIndent()
        GenUiCompiler.compile(raw) // Syntax validation alone cannot catch a missing data binding.
        val graph = graph(recover(raw))
        assertTableRows(graph, "One", listOf(listOf("One", "13"), listOf("Two", "27")))
        assertTextComponents(graph, "The note must not hide missing table data.")
    }

    @Test
    fun multilineStateAndQuotedClosingTagDoNotHideFollowingContent() {
        val raw = """
            <a2ui>
            ${'$'}/={data:[
              {name:"One",value:13},
              {name:"Two",value:27}
            ]}
            root=Column([note,broken])
            note=Text("Keep the literal </a2ui> marker.")
            after=Text("Content after the quoted marker.")
            broken=Card([missing]
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        assertTableRows(graph, "One", listOf(listOf("One", "13"), listOf("Two", "27")))
        assertTextComponents(graph, "Keep the literal </a2ui> marker.", "Content after the quoted marker.")
    }

    @Test
    fun standaloneLiteralRecoveryHandlesWindowsLineEndings() {
        val raw = "<a2ui>\r\nBroken(\"First recovered fact\")\r\nBroken(\"Second recovered fact\")\r\n</a2ui>"
        val values = leafStrings(graph(recover(raw)))
        assertTrue("First recovered fact" in values)
        assertTrue("Second recovered fact" in values)
    }

    private fun recoverCaptured(
        resource: String,
        expectedSha256: String,
        reportId: String,
    ): GenUiCompileOutcome {
        val bytes = requireNotNull(javaClass.getResourceAsStream(resource)) {
            "Missing recovery fixture $resource"
        }.use { it.readBytes() }
        assertEquals("Captured fixture bytes changed: $resource", expectedSha256, sha256(bytes))
        val outcome = recover(bytes.toString(Charsets.UTF_8))
        writeReport(reportId, bytes, outcome)
        return outcome
    }

    private fun recover(raw: String): GenUiCompileOutcome {
        val outcome = GenUiCompiler.compileWithRepair(
            input = raw,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, outcome.repairKind)
        return outcome
    }

    private fun graph(outcome: GenUiCompileOutcome): JsonObject =
        GenUiIrCodec.decode(JsonParser.parseString(outcome.document.a2uiJson)).canonicalGraph

    private fun assertTextComponents(graph: JsonObject, vararg expected: String) {
        val actual = graph.getAsJsonObject("elements").entrySet().mapNotNull { (_, raw) ->
            raw.asJsonObject.takeIf { it.get("type")?.asString == "Text" }
                ?.getAsJsonObject("props")?.get("text")?.asString
        }
        expected.forEach { value ->
            assertTrue("Missing recovered Text component: $value\nActual Text values: $actual", value in actual)
        }
    }

    private fun assertTableRows(graph: JsonObject, marker: String, expected: List<List<String>>) {
        val rows = tableRows(graph).firstOrNull { table -> leafStrings(table).contains(marker) }
            ?: error("No recovered table contains '$marker'. Recovered document: $graph")
        val expectedSignatures = expected.map(::rowSignature)
        val actualSignatures = rows.map { rowSignature(leafStrings(it)) }
        assertEquals("Recovered rows containing '$marker' changed", expectedSignatures, actualSignatures)
    }

    private fun tableRows(graph: JsonObject): List<JsonArray> {
        val state = graph.getAsJsonObject("state") ?: JsonObject()
        return graph.getAsJsonObject("elements").entrySet().mapNotNull { (_, raw) ->
            val element = raw.asJsonObject
            if (element.get("type")?.asString != "Table") return@mapNotNull null
            val props = element.getAsJsonObject("props") ?: return@mapNotNull null
            props.get("rows")?.takeIf { it.isJsonArray }?.asJsonArray
                ?: props.get("statePath")
                    ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                    ?.asString
                    ?.let { resolvePointer(state, it) }
                    ?.takeIf { it.isJsonArray }
                    ?.asJsonArray
        }
    }

    private fun resolvePointer(root: JsonElement, rawPath: String): JsonElement? {
        if (rawPath.isEmpty() || rawPath == "/") return root
        if (!rawPath.startsWith('/')) return null
        var current = root
        rawPath.split('/').drop(1).forEach { rawSegment ->
            val segment = rawSegment.replace("~1", "/").replace("~0", "~")
            current = when {
                current.isJsonObject -> current.asJsonObject.get(segment) ?: return null
                current.isJsonArray -> segment.toIntOrNull()?.let { index ->
                    current.asJsonArray.takeIf { index in 0 until it.size() }?.get(index)
                } ?: return null
                else -> return null
            }
        }
        return current
    }

    private fun leafStrings(value: JsonElement): List<String> = when {
        value.isJsonNull -> emptyList()
        value.isJsonPrimitive -> listOf(value.asJsonPrimitive.asString)
        value.isJsonArray -> value.asJsonArray.flatMap(::leafStrings)
        value.isJsonObject -> value.asJsonObject.entrySet().flatMap { (_, child) -> leafStrings(child) }
        else -> emptyList()
    }

    private fun rowSignature(values: List<String>): String =
        values.filter(String::isNotEmpty).sorted().joinToString("\u001f")

    private fun writeReport(id: String, raw: ByteArray, outcome: GenUiCompileOutcome) {
        val directory = File("build/reports/recovery_content/$id").apply { mkdirs() }
        File(directory, "raw.express").writeBytes(raw)
        File(directory, "repaired.express").writeText(outcome.document.express, Charsets.UTF_8)
        File(directory, "repaired.a2ui.json").writeText(outcome.document.a2uiJson, Charsets.UTF_8)
        File(directory, "diagnostics.txt").writeText(
            outcome.diagnostics.joinToString(separator = "\n", postfix = "\n"),
            Charsets.UTF_8,
        )
        File(directory, "repair-kind.txt").writeText("${outcome.repairKind}\n", Charsets.UTF_8)
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
