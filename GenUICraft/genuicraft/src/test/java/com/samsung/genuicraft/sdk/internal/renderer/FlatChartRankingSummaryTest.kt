package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatChartRankingSummaryTest {
    @Test fun `accepted chart keeps every authored rank raw title and date including the repeated track at rank four`() {
        val headers = listOf("Rank", "Song Title", "Chart Week Date")
        val rows = listOf(
            listOf("1", "“Hate That I Made You Love Me” — Ariana Grande", "September 5, 2026"),
            listOf("2", "“Choosin' Texas” — Ella Langley", "September 5, 2026"),
            listOf("3", "“Loser” — Tame Impala", "September 5, 2026"),
            listOf("4", "“Hate That I Made You Love Me” — Ariana Grande", "September 5, 2026"),
            listOf("5", "“BbY WOW” — Karol G With Judeline & rusowsky", "September 5, 2026"),
        )
        val snapshot = rows.map { it.toList() }
        val profile = chartRankingProfile(headers, rows)!!

        assertEquals(ChartRankingProfile(0, 1, 2), profile)
        assertEquals(listOf(2), profile.detailIndexes(headers))
        assertEquals(listOf("1", "2", "3", "4", "5"), rows.map { it[profile.rank] })
        assertEquals(rows[0][profile.title], rows[3][profile.title])
        assertEquals("4", rows[3][profile.rank])
        assertEquals(5, rows.size)
        assertEquals(listOf("September 5, 2026"), rows.map { it[profile.date] }.distinct())
        val indexes = listOf(profile.rank, profile.title, profile.date).sorted()
        assertEquals(headers.indices.toList(), indexes)
        rows.forEach { row -> assertEquals(headers.zip(row), indexes.map { headers[it] to row[it] }) }
        assertEquals(snapshot, rows)
    }

    @Test fun `permuted roles retain arbitrary supplied ranks differing dates and every qualified extra field`() {
        val headers = listOf("Source qualification", "Chart date", "Track Title", "Position", "Additional detail")
        val rows = listOf(
            listOf("Unverified chart caveat [8]", "September 5, 2026", "“First track” — Original Artist [2]",
                "T-7", "Keep the complete exception; no inferred trend [9]."),
            listOf("No further detail in the excerpt [11]", "September 12, 2026", "“Second track” — Other Artist [3]",
                "=7 [5]", "Unknown movement; retain this qualification [13]."),
        )
        val profile = chartRankingProfile(headers, rows)!!
        assertEquals(ChartRankingProfile(3, 2, 1), profile)
        assertEquals(listOf(0, 1, 4), profile.detailIndexes(headers))
        assertEquals(listOf("T-7", "=7 [5]"), rows.map { it[profile.rank] })
        assertEquals(listOf("September 5, 2026", "September 12, 2026"), rows.map { it[profile.date] })
        val details = headers.indices.filter { it !in setOf(profile.rank, profile.title, profile.date) }
        assertEquals(listOf(0, 4), details)
        val indexes = (listOf(profile.rank, profile.title, profile.date) + details).sorted()
        assertEquals(headers.indices.toList(), indexes)
        rows.forEach { row -> assertEquals(headers.zip(row), indexes.map { headers[it] to row[it] }) }
    }

    @Test fun `chart aliases require ranking identity title and chart-specific date rather than ordinary playlist dates`() {
        listOf("Chart rank", "Position", "Rank").forEach { rankHeader ->
            listOf("Title", "Song", "Track").forEach { titleHeader ->
                assertEquals(ChartRankingProfile(0, 1, 2), chartRankingProfile(
                    listOf(rankHeader, titleHeader, "Chart week"),
                    listOf(listOf("Authored rank", "Whole title — Original artist", "Week not stated [4]")),
                ))
            }
        }
        assertNull(chartRankingProfile(listOf("Rank", "Song Title", "Release date"), listOf(listOf("1", "A", "2026"))))
        assertNull(chartRankingProfile(listOf("Artist", "Song Title", "Chart Week Date"), listOf(listOf("A", "B", "2026"))))
        assertNull(chartRankingProfile(listOf("Rank", "Artist", "Chart Week Date"), listOf(listOf("1", "A", "2026"))))
        assertNull(chartRankingProfile(listOf("Rank", "Song Title", "Chart"), listOf(listOf("1", "A", "Global 200"))))
    }

    @Test fun `empty ragged missing identity and authored URL media or action content retain the existing route`() {
        val headers = listOf("Rank", "Song Title", "Chart Week Date")
        val valid = listOf("4", "“Original track” — Original artist [2]", "September 5, 2026")
        assertNull(chartRankingProfile(headers, emptyList()))
        assertNull(chartRankingProfile(headers, listOf(valid.dropLast(1))))
        assertNull(chartRankingProfile(headers, listOf(valid + "Unheaded value [9]")))
        listOf(0, 1).forEach { index ->
            listOf("", " ").forEach { blank ->
                val row = valid.mapIndexed { cellIndex, value -> if (cellIndex == index) blank else value }
                assertNull(chartRankingProfile(headers, listOf(row)))
            }
        }
        listOf("Photo URL", "Image", "Media", "Website", "Action").forEach { authoredHeader ->
            assertNull(chartRankingProfile(headers + authoredHeader, listOf(valid + "Original authored value")))
        }
        listOf("https://www.who.int", "www.who.int", "who.int", "//www.who.int", "tel:+911234567890").forEach { authoredUrl ->
            val row = valid.mapIndexed { index, value -> if (index == 2) authoredUrl else value }
            assertNull(chartRankingProfile(headers, listOf(row)))
        }
    }
}
