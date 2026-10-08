package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatDepositRateSummaryTest {
    @Test fun `accepted FD comparison preserves every bank rate deposit bucket and effective date wording`() {
        val headers = listOf("Bank", "One-year retail FD rate", "Assumed deposit size", "Rate date / effective date")
        val rows = listOf(
            listOf("SBI", "6.25%", "Below ₹3 crore",
                "Current retail grid shown on SBI’s retail domestic term-deposit page"),
            listOf("HDFC Bank", "6.25%", "Below ₹3 crore", "19 August 2026"),
            listOf("ICICI Bank", "6.25%", "Below ₹3 crore",
                "Current retail grid shown in the cited rate snapshot"),
        )
        val profile = depositRateSummaryProfile(headers, rows)!!

        assertEquals(0, profile.bank)
        assertEquals(1, profile.rate)
        assertEquals(listOf(2, 3), profile.detailIndexes(headers))
        assertEquals(listOf("SBI", "HDFC Bank", "ICICI Bank"), rows.map { it[profile.bank] })
        assertEquals(listOf("6.25%", "6.25%", "6.25%"), rows.map { it[profile.rate] })
        val representedIndexes = (listOf(profile.bank, profile.rate) + profile.detailIndexes(headers)).sorted()
        assertEquals(headers.indices.toList(), representedIndexes)
        rows.forEach { row ->
            assertEquals(headers.zip(row), representedIndexes.map { headers[it] to row[it] })
            assertEquals(listOf(row[2], row[3]), profile.detailIndexes(headers).map(row::get))
        }
    }

    @Test fun `permuted bank and fixed deposit rate roles keep unknown labeled details in source order`() {
        val headers = listOf("Rate date / effective date", "Bank name", "Unknown extra qualification",
            "Fixed deposit rate", "Assumed deposit size", "Withdrawal caveat")
        val row = listOf("Effective date not stated in the cited snapshot [8]", "Bank A [1]",
            "Not verified; check the original terms [9]", "Rate unavailable [3]", "Bucket not stated [5]",
            "Exact penalty unknown; an exemption may apply [11].")
        val profile = depositRateSummaryProfile(headers, listOf(row))!!

        assertEquals(1, profile.bank)
        assertEquals(3, profile.rate)
        assertEquals(listOf(0, 2, 4, 5), profile.detailIndexes(headers))
        assertEquals(listOf(row[0], row[2], row[4], row[5]), profile.detailIndexes(headers).map(row::get))
        assertEquals("Rate unavailable [3]", row[profile.rate])
        assertEquals(headers.indices.toList(), (listOf(profile.bank, profile.rate) + profile.detailIndexes(headers)).sorted())
    }

    @Test fun `generic rates forex feature matrices and missing row identities retain their existing route`() {
        listOf("Loan interest rate", "Home loan rate", "Exchange rate", "Rate", "FD tenure").forEach { unrelatedHeader ->
            assertNull(depositRateSummaryProfile(listOf("Bank", unrelatedHeader), listOf(listOf("Bank A", "6.25%"))))
        }
        assertNull(depositRateSummaryProfile(listOf("Currency", "Exchange rate"), listOf(listOf("USD", "84.00"))))
        assertNull(depositRateSummaryProfile(listOf("Feature", "SBI FD rate", "HDFC FD rate"),
            listOf(listOf("One year", "6.25%", "6.25%"))))
        assertNull(depositRateSummaryProfile(listOf("Name", "One-year FD rate"), listOf(listOf("A", "6.25%"))))
        assertNull(depositRateSummaryProfile(listOf("Bank", "One-year FD rate"), emptyList()))
        assertNull(depositRateSummaryProfile(listOf("Bank", "One-year FD rate"), listOf(listOf("", "6.25%"))))
        assertNull(depositRateSummaryProfile(listOf("One-year FD rate", "Bank"), listOf(listOf("6.25%"))))
        assertNull(depositRateSummaryProfile(listOf("Bank", "One-year FD rate"),
            listOf(listOf("A", "6.25%"), listOf(" ", "6.50%"))))
        assertNull(depositRateSummaryProfile(listOf("Bank", "One-year FD rate"),
            listOf(listOf("A", "6.25%", "Unheaded condition [7]"))))
    }

    @Test fun `authored media contact and action columns retain the existing renderer`() {
        listOf("Photo URL", "Image", "Media", "Website", "URL", "Phone", "Booking", "Action", "Action label",
            "Source link", "Terms link", "Link", "Href", "CTA", "CTA label", "Button label").forEach { authoredColumn ->
            assertNull(depositRateSummaryProfile(
                listOf("Bank", "One-year FD rate", authoredColumn),
                listOf(listOf("A", "6.25%", "Original authored value [4]")),
            ))
        }
    }

    @Test fun `URL forms already supported by entity actions keep their existing route even under a plain detail label`() {
        listOf("www.who.int", "who.int", "//www.who.int", "tel:+911234567890").forEach { authoredUrl ->
            assertNull(depositRateSummaryProfile(
                listOf("Bank", "One-year FD rate", "Terms"),
                listOf(listOf("SBI", "6.25%", authoredUrl)),
            ))
        }
        assertEquals(DepositRateSummaryProfile(0, 1), depositRateSummaryProfile(
            listOf("Bank", "One-year FD rate", "Rate date / effective date"),
            listOf(listOf("SBI", "6.25%", "19 August 2026")),
        ))
    }
}
