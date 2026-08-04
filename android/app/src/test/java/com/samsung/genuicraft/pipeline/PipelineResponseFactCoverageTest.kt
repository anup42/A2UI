package com.samsung.genuicraft.pipeline

import com.google.gson.JsonParser
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineResponseFactCoverageTest {
    @Test
    fun missingFacts_ignoresCarrierInActionUrlAndFindsOmittedVisibleFacts() {
        val response = """
            # Delivery Status for Order A1042

            **Carrier:** SwiftShip
            **Current Status:** Out for delivery
            **Estimated Delivery Time:** Today by 4:30 PM

            Action: [Button: Track Package] https://www.swiftship.com/track
        """.trimIndent()
        val canonical = JsonParser.parseString(
            """
            {"root":"root","state":{},"elements":{
              "root":{"type":"Stack","props":{},"children":["status","button"]},
              "status":{"type":"Text","props":{"text":"Out for delivery"},"children":[]},
              "button":{"type":"Button","props":{"label":"Track Package"},"children":[],
                "on":{"press":{"action":"openUrl","params":{"url":"https://www.swiftship.com/track"}}}}
            }}
            """.trimIndent()
        )

        val missing = ResponseFactCoverage.missingFacts(response, canonical)

        assertTrue(missing.any { it.label == "Carrier" && it.value == "SwiftShip" })
        assertTrue(missing.any { it.label == "Estimated Delivery Time" && it.value == "Today by 4:30 PM" })
    }

    @Test
    fun missingFacts_acceptsValuesRenderedInTableRows() {
        val response = """
            | Detail | Value |
            | :--- | :--- |
            | Carrier | SwiftShip |
            | Current status | Out for delivery |
            | ETA | Today, 4:30 PM |
        """.trimIndent()
        val canonical = JsonParser.parseString(
            """
            {"root":"root","state":{},"elements":{
              "root":{"type":"Table","props":{"columns":["Detail","Value"],
                "rows":[["Carrier","SwiftShip"],["Status","Out for delivery"],["ETA","Today by 4:30 PM"]]},
                "children":[]}
            }}
            """.trimIndent()
        )

        assertTrue(ResponseFactCoverage.missingFacts(response, canonical).isEmpty())
    }

    @Test
    fun missingFacts_doesNotTreatMediaAndActionsAsStructuredFacts() {
        val response = """
            Media: Icon=https://cdn.example.test/star.svg
            Action: [Button: Track Package] https://example.test/track
        """.trimIndent()
        val canonical = JsonParser.parseString(
            """{"root":"root","state":{},"elements":{"root":{"type":"Text","props":{"text":"Ready"},"children":[]}}}"""
        )

        assertTrue(ResponseFactCoverage.missingFacts(response, canonical).isEmpty())
    }

    @Test
    fun missingFacts_ignoresBareSourceUrlsAndBlankMediaLabels() {
        val response = """
            Camera:
            Weight:
            https://fujifilm-x.com/global/products/cameras/x-s20/specifications/
        """.trimIndent()
        val canonical = JsonParser.parseString(
            """{"root":"root","state":{},"elements":{"root":{"type":"Text","props":{"text":"Ready"},"children":[]}}}"""
        )

        assertTrue(ResponseFactCoverage.missingFacts(response, canonical).isEmpty())
    }

    @Test
    fun missingFacts_preservesHeadingStatusAndOrderIdFromNaturalResponseProse() {
        val response = """
            # SwiftShip Order Status

            ## Order A1042 - Out for Delivery

            Your order A1042 is currently **Out for Delivery** with SwiftShip.

            **Estimated Time of Arrival:** Today by 4:30 PM

            ## Track Package
            Action: [Button: Track Package]
        """.trimIndent()
        val incomplete = JsonParser.parseString(
            """
            {"root":"root","state":{},"elements":{
              "root":{"type":"Stack","props":{},"children":["header","eta","button"]},
              "header":{"type":"Text","props":{"text":"SwiftShip Order Status"},"children":[]},
              "eta":{"type":"Text","props":{"text":"Today by 4:30 PM"},"children":[]},
              "button":{"type":"Button","props":{"label":"Track Package"},"children":[],
                "on":{"press":{"action":"openUrl","params":{"url":"https://example.test/track/A1042"}}}}
            }}
            """.trimIndent()
        )

        val missing = ResponseFactCoverage.missingFacts(response, incomplete)

        assertTrue(missing.any { it.value == "Order A1042 - Out for Delivery" })
        assertTrue(missing.any { it.value == "Out for Delivery" })
        assertTrue(missing.any { it.value == "A1042" })
    }
}
