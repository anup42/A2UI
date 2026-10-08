package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatCurrencyQuoteSummaryTest {
    @Test fun `accepted currency quotes retain every exact pair rate and shared timestamp cell`() {
        val headers = listOf("Currency", "Rate", "Quoted As Of")
        val rows = listOf(
            listOf("USD/INR", "95.5614", "28-Aug-2026 14:23:47 IST"),
            listOf("EUR/INR", "111.2956", "28-Aug-2026 14:23:47 IST"),
            listOf("GBP/INR", "129.8867", "28-Aug-2026 14:23:47 IST"),
        )
        val profile = currencyQuoteSummaryProfile(headers, rows)!!
        assertEquals(CurrencyQuoteSummaryProfile(0, 1, 2), profile)
        assertEquals(listOf("95.5614", "111.2956", "129.8867"), rows.map { it[profile.rate] })
        val indexes = listOfNotNull(profile.currency, profile.rate, profile.quotedAt).sorted()
        assertEquals(headers.indices.toList(), indexes)
        rows.forEach { row -> assertEquals(headers.zip(row), indexes.map { headers[it] to row[it] }) }
        assertEquals(listOf("28-Aug-2026 14:23:47 IST"), rows.map { it[profile.quotedAt!!] }.distinct())
    }

    @Test fun `permuted quote roles preserve pair citations and differing timestamps per row`() {
        listOf("As of", "Quoted at").forEach { timeHeader ->
            val headers = listOf(timeHeader, "Exchange rate", "Currency pair")
            val rows = listOf(
                listOf("28-Aug-2026 14:23:47 IST", "95.5614", "USD/INR [1][3]"),
                listOf("28-Aug-2026 14:24:02 IST", "111.2956", "EUR/INR [2]"),
            )
            val profile = currencyQuoteSummaryProfile(headers, rows)!!
            assertEquals(CurrencyQuoteSummaryProfile(2, 1, 0), profile)
            assertEquals(listOf("USD/INR [1][3]", "EUR/INR [2]"), rows.map { it[profile.currency] })
            assertEquals(rows.map { it[0] }, rows.map { it[profile.quotedAt!!] })
            assertEquals(2, rows.map { it[profile.quotedAt!!] }.distinct().size)
            val indexes = listOfNotNull(profile.currency, profile.rate, profile.quotedAt).sorted()
            assertEquals(headers.indices.toList(), indexes)
            rows.forEach { row -> assertEquals(headers.zip(row), indexes.map { headers[it] to row[it] }) }
        }
        assertEquals(CurrencyQuoteSummaryProfile(0, 1, null),
            currencyQuoteSummaryProfile(listOf("Pair", "Exchange rate"), listOf(listOf("USD/INR", "95.5614"))))
    }

    @Test fun `weak bank deposit equity and malformed pair schemas keep the existing route`() {
        assertNull(currencyQuoteSummaryProfile(listOf("Bank", "Rate"), listOf(listOf("SBI", "6.25%"))))
        assertNull(currencyQuoteSummaryProfile(listOf("Bank", "One-year FD rate"), listOf(listOf("SBI", "6.25%"))))
        assertNull(currencyQuoteSummaryProfile(listOf("Currency", "One-year FD rate"), listOf(listOf("USD/INR", "6.25%"))))
        assertNull(currencyQuoteSummaryProfile(listOf("Stock", "Price"), listOf(listOf("AAPL", "250.00"))))
        assertNull(currencyQuoteSummaryProfile(listOf("Name", "Rate"), listOf(listOf("USD/INR", "95.5614"))))
        listOf("USDINR", "USD→INR", "US/INR", "USD/INRR",
            "United States Dollar/INR", "USD/INR as of today", "USD/INR [source]").forEach { unrelatedPair ->
            assertNull(currencyQuoteSummaryProfile(listOf("Pair", "Rate"), listOf(listOf(unrelatedPair, "95.5614"))))
        }
        assertNull(currencyQuoteSummaryProfile(listOf("Pair", "Rate"),
            listOf(listOf("USD/INR", "95.5614"), listOf("AAPL/INR", "250.00"))))
    }

    @Test fun `unknown columns blank cells ragged rows and URLs keep all authored content on the existing route`() {
        val headers = listOf("Currency", "Rate", "Quoted As Of")
        val validRow = listOf("USD/INR", "95.5614", "28-Aug-2026 14:23:47 IST")
        assertNull(currencyQuoteSummaryProfile(headers, emptyList()))
        assertNull(currencyQuoteSummaryProfile(listOf("Currency", "Rate", "Change"), listOf(validRow)))
        assertNull(currencyQuoteSummaryProfile(headers + "Source note", listOf(validRow + "Keep this qualification [7]")))
        assertNull(currencyQuoteSummaryProfile(headers, listOf(validRow.dropLast(1))))
        assertNull(currencyQuoteSummaryProfile(headers, listOf(validRow + "Unheaded value [9]")))
        headers.indices.forEach { index ->
            listOf("", " ", "https://www.example.com/quote").forEach { invalidValue ->
                val row = validRow.mapIndexed { cellIndex, value -> if (cellIndex == index) invalidValue else value }
                assertNull(currencyQuoteSummaryProfile(headers, listOf(row)))
            }
        }
    }

    @Test fun `URL forms already supported by entity actions in rate or timestamp retain the existing route`() {
        val headers = listOf("Currency", "Rate", "Quoted As Of")
        val validRow = listOf("USD/INR", "95.5614", "28-Aug-2026 14:23:47 IST")
        listOf("www.who.int", "who.int", "//www.who.int", "tel:+911234567890").forEach { authoredUrl ->
            listOf(1, 2).forEach { index ->
                val row = validRow.mapIndexed { cellIndex, value -> if (cellIndex == index) authoredUrl else value }
                assertNull(currencyQuoteSummaryProfile(headers, listOf(row)))
            }
        }
        assertEquals(CurrencyQuoteSummaryProfile(0, 1, 2), currencyQuoteSummaryProfile(headers, listOf(validRow)))
    }
}
