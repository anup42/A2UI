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
    fun capturedFp32RestaurantRepairPreservesGeneratedColumnLabelsAndValues() {
        val raw = File("../validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_on/BXP-008/output.express")
            .readText(Charsets.UTF_8)
        val outcome = recover(raw)
        val graph = graph(outcome)
        val tables = graph.getAsJsonObject("elements").entrySet().map { it.value.asJsonObject }
            .filter { it.get("type")?.asString == "Table" }
        assertEquals("Recovered state should be represented by one table", 1, tables.size)
        val props = tables.single().getAsJsonObject("props")
        assertEquals(
            listOf("Restaurant", "Neighborhood", "Signature dish", "Approx. cost per person", "Opening hours"),
            props.getAsJsonArray("columns").map { it.asJsonObject.get("label").asString },
        )
        assertEquals("restaurant", props.get("primaryColumn").asString)
        assertTableRows(graph, "Vidyarthi Bhavan", listOf(
            listOf("Vidyarthi Bhavan", "Basavanagudi (Gandhi Bazaar)",
                "South Indian breakfast classics; widely known for its crisp dosas", "₹1–200",
                "6:30am–11:30am, 2:00pm–8:00pm on Mon–Thu; closed Fri; 6:30am–12:00pm, 2:30pm–8:00pm on Sat–Sun[2]"),
            listOf("Central Tiffin Room", "Malleshwaram", "Benne dosas, bajji, and coffee", "₹1–200",
                "7:00am–12:30pm, 4:00pm–9:30pm daily[4]"),
            listOf("Brahmins' Coffee Bar", "Shankarapura, near Shankar Mutt Road", "Idli, vada, and coffee",
                "₹1–200", "6:00am–12:00pm, 3:00pm–7:00pm Mon–Sat; closed Sun[5]"),
        ))
        assertFalse("Additional recovered text" in leafStrings(graph))
    }

    @Test
    fun capturedFp32FdRepairKeepsTitleBeforeModelAuthoredComparisonTable() {
        val raw = File("../validation/20260928_fp32_bixby50_mtp/batches/group_03_mtp_off/BXP-013/output.express")
            .readText(Charsets.UTF_8)
        val outcome = recover(raw)
        val graph = graph(outcome)
        assertEquals("One-year FD rates", reachableTextValues(graph).first())
        val state = graph.getAsJsonObject("state")
        assertTrue("Complete state-backed table should retain generated state", state.has("quick_view_data"))
        val table = graph.getAsJsonObject("elements").entrySet().map { it.value.asJsonObject }
            .single { it.get("type")?.asString == "Table" }
        val props = table.getAsJsonObject("props")
        assertEquals("/quick_view_data", props.get("statePath").asString)
        assertEquals(
            listOf("Bank", "One-year retail FD rate", "Assumed deposit size", "Rate date / effective date"),
            props.getAsJsonArray("columns").map { it.asJsonObject.get("label").asString },
        )
        assertTableRows(graph, "SBI", listOf(
            listOf("SBI", "6.25%", "Below ₹3 crore",
                "Current retail grid shown on SBI’s retail domestic term-deposit page"),
            listOf("HDFC Bank", "6.25%", "Below ₹3 crore", "19 August 2026"),
            listOf("ICICI Bank", "6.25%", "Below ₹3 crore",
                "Current retail grid shown in the cited rate snapshot"),
        ))
    }

    @Test
    fun unusedGeneratedStatePreventsChoosingAPrunedCompleteGraph() {
        val raw = """
            <a2ui>
            ${'$'}/={used:[{name:"Shown"}],unused:[{name:"Otherwise lost"}]}
            root=Column([heading,table,broken])
            heading=Text("Answer")
            table=Table(columns=[{key:"name",label:"Name"}],statePath="/used")
            broken=Card([missing])
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        assertTrue("Generated unused state must remain visible in salvage", "Otherwise lost" in leafStrings(graph))
    }

    @Test
    fun completeGraphPreservesReachableRepeatStateWhenAnOrphanNeedsAttachment() {
        val raw = """
            <a2ui>
            ${'$'}/={rows:[{name:"First"},{name:"Second"}]}
            root=Column([heading,repeated])
            heading=Text("Repeated results")
            repeated=Column([template],repeat={statePath:"/rows",template:"template"})
            template=Text("${'$'}item.name")
            orphan=Text("Final generated note")
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        assertTrue(graph.getAsJsonObject("state").has("rows"))
        assertTrue(graph.getAsJsonObject("elements").entrySet().any { (_, rawElement) ->
            rawElement.asJsonObject.getAsJsonObject("repeat")?.get("statePath")?.asString == "/rows"
        })
        assertEquals("Repeated results", reachableTextValues(graph).first())
        assertTrue("Final generated note" in reachableTextValues(graph))
    }

    @Test
    fun nestedBoundTableKeepsGeneratedSchemaWhenCompleteGraphIsRecoverable() {
        val raw = """
            <a2ui>
            ${'$'}/={data:{rows:[{name:"First",percent:"57%"}]}}
            root=Column([heading,table,broken])
            heading=Text("Forecast")
            table=Table(columns=[{key:"name",label:"Day"},{key:"percent",label:"Rain probability"}],statePath="/data/rows")
            broken=Card([missing])
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        assertEquals("Forecast", reachableTextValues(graph).first())
        val table = graph.getAsJsonObject("elements").entrySet().map { it.value.asJsonObject }
            .single { it.get("type")?.asString == "Table" }
        assertEquals("/data/rows", table.getAsJsonObject("props").get("statePath").asString)
        assertEquals(listOf("Day", "Rain probability"),
            table.getAsJsonObject("props").getAsJsonArray("columns").map { it.asJsonObject.get("label").asString })
    }

    @Test
    fun capturedStrictAqiAttachesGeneratedGuidanceWithoutChangingLiteralText() {
        val raw = File("../validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-002/output.express")
            .readText(Charsets.UTF_8)
        GenUiCompiler.compile(raw)
        val outcome = recover(raw)
        val graph = graph(outcome)
        val visible = reachableTextValues(graph)
        assertEquals(1, visible.count { it == "Main pollutant: PM2.5 [5][7]" })
        assertEquals(1, visible.count { it == "Outdoor-activity guidance" })
        assertTrue(visible.any { it.startsWith("For the Poor category, air pollution can cause breathing discomfort") })
    }

    @Test
    fun capturedStrictMetroAttachesPassAndInterchangeDetailsWithoutRepeatingHeadings() {
        val raw = File("../validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-005/output.express")
            .readText(Charsets.UTF_8)
        GenUiCompiler.compile(raw)
        val visible = reachableTextValues(graph(recover(raw)))
        assertEquals(1, visible.count { it == "Ticket and pass options" })
        assertEquals(1, visible.count { it == "Interchange stations" })
        assertTrue(visible.any { it.startsWith("Tourist passes listed in current fare guides include 1-day") })
        assertTrue(visible.any { it.startsWith("The main interchange is Nadaprabhu Kempegowda Station") })
        assertTrue(visible.any { it.startsWith("Buy a token for one trip") })
    }

    @Test
    fun orphanRepairPrunesAlreadyVisibleChildrenAndLeavesConditionalContentDormant() {
        val raw = """
            <a2ui>
            root=Column([heading,detail])
            heading=Text("Heading")
            detail=Text("Existing detail")
            orphan=Card([detail,new_detail])
            new_detail=Text("New detail")
            dormant=Column([conditional_child],visible=false)
            conditional_child=Text("Conditional detail")
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        val visible = reachableTextValues(graph)
        assertEquals(1, visible.count { it == "Existing detail" })
        assertEquals(1, visible.count { it == "New detail" })
        assertFalse("Conditional detail" in visible)
    }

    @Test
    fun deliberateRepeatedTemplateDoesNotTriggerGraphRepair() {
        val raw = """
            <a2ui>
            ${'$'}/={rows:["A","B"]}
            root=Column([repeated,repeated])
            repeated=Column([template],repeat={statePath:"/rows",template:"template"})
            template=Text("Repeated template")
            </a2ui>
        """.trimIndent()
        val outcome = GenUiCompiler.compileWithRepair(raw, allowSourceTextFallback = false, allowGeneratedDslRepair = true)
        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)
    }

    @Test
    fun graphRepairDoesNotLetGuardedBranchesStealVisibleRootContent() {
        listOf(
            "hidden=Column([detail],visible=false)",
            "hidden=Column([detail],repeat={statePath:\"/rows\",template:\"detail\"})",
        ).forEach { guarded ->
            val raw = """
                <a2ui>
                ${'$'}/={rows:[]}
                root=Column([hidden,detail])
                $guarded
                detail=Text("Important visible fact")
                orphan=Text("Another generated fact")
                </a2ui>
            """.trimIndent()
            val graph = graph(recover(raw))
            val elements = graph.getAsJsonObject("elements")
            val detailId = elements.entrySet().single { (_, element) ->
                element.asJsonObject.get("type")?.asString == "Text" &&
                    element.asJsonObject.getAsJsonObject("props")?.get("text")?.asString == "Important visible fact"
            }.key
            assertTrue("Direct visible reference was lost for $guarded",
                elements.getAsJsonObject("root").getAsJsonArray("children").any { it.asString == detailId })
        }
    }

    @Test
    fun orphanModalContentIsNotPromotedIntoTheMainAnswer() {
        val raw = """
            <a2ui>
            root=Column([answer])
            answer=Text("Visible answer")
            dormant_modal=Modal(trigger=button,content=modal_text)
            button=Button("Open details")
            modal_text=Text("Dormant modal content")
            </a2ui>
        """.trimIndent()
        val outcome = GenUiCompiler.compileWithRepair(raw, allowSourceTextFallback = false, allowGeneratedDslRepair = true)
        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)
    }

    @Test
    fun recoveredTableWithSameRowsButDifferentUnitsIsNotDeduplicated() {
        val raw = """
            <a2ui>
            root=Column([first])
            first=Table(columns=["Rain chance"],rows=[["57"]])
            second=Table(columns=["Rain chance (%)"],rows=[["57"]])
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        val labels = graph.getAsJsonObject("elements").entrySet().map { it.value.asJsonObject }
            .filter { it.get("type")?.asString == "Table" }
            .map { it.getAsJsonObject("props").getAsJsonArray("columns").first().asString }
        assertEquals(listOf("Rain chance", "Rain chance (%)"), labels)
    }

    @Test
    fun equalStatusTextSurvivesInDistinctGeneratedEntityCards() {
        val raw = """
            <a2ui>
            root=Column([first])
            first=Card([first_status],title="Cafe A")
            first_status=Text("Open now")
            second=Card([second_status],title="Cafe B")
            second_status=Text("Open now")
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        assertEquals(2, reachableTextValues(graph).count { it == "Open now" })
        val titles = graph.getAsJsonObject("elements").entrySet().mapNotNull { (_, rawElement) ->
            rawElement.asJsonObject.takeIf { it.get("type")?.asString == "Card" }
                ?.getAsJsonObject("props")?.get("title")?.asString
        }
        assertEquals(listOf("Cafe A", "Cafe B"), titles)
    }

    @Test
    fun sharedStatusElementStaysReferencedByBothVisibleEntityCards() {
        val raw = """
            <a2ui>
            root=Column([first,second])
            first=Card([shared_status],title="Cafe A")
            second=Card([shared_status],title="Cafe B")
            shared_status=Text("Open now")
            orphan=Text("Another generated fact")
            </a2ui>
        """.trimIndent()
        val graph = graph(recover(raw))
        val elements = graph.getAsJsonObject("elements")
        val sharedId = elements.entrySet().single { (_, rawElement) ->
            rawElement.asJsonObject.get("type")?.asString == "Text" &&
                rawElement.asJsonObject.getAsJsonObject("props")?.get("text")?.asString == "Open now"
        }.key
        val cards = elements.entrySet().map { it.value.asJsonObject }
            .filter { it.get("type")?.asString == "Card" }
        assertEquals(2, cards.size)
        assertTrue(cards.all { card -> card.getAsJsonArray("children").any { it.asString == sharedId } })
    }

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
        assertFalse("Forecast recovery must not show a diagnostic card", "Additional recovered text" in leafStrings(graph))
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

    private fun reachableTextValues(graph: JsonObject): List<String> {
        val elements = graph.getAsJsonObject("elements")
        val texts = mutableListOf<String>()
        val visiting = mutableSetOf<String>()
        fun visit(id: String) {
            if (!visiting.add(id)) return
            val element = elements.getAsJsonObject(id) ?: return
            if (element.get("type")?.asString == "Text") {
                element.getAsJsonObject("props")?.get("text")?.asString?.let(texts::add)
            }
            element.getAsJsonArray("children")?.forEach { visit(it.asString) }
        }
        visit(graph.get("root").asString)
        return texts
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
