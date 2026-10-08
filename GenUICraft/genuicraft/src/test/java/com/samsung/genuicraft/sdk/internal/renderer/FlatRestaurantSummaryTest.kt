package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatRestaurantSummaryTest {
    @Test fun `breakfast schema preserves every source column split hours closures cost and citations`() {
        val headers = listOf("Restaurant", "Neighborhood", "Signature dish", "Approx. cost per person", "Opening hours")
        val rows = listOf(
            listOf("Vidyarthi Bhavan", "Basavanagudi (Gandhi Bazaar)",
                "South Indian breakfast classics; widely known for its crisp dosas", "₹1–200",
                "6:30am–11:30am, 2:00pm–8:00pm on Mon–Thu; closed Fri; 6:30am–12:00pm, 2:30pm–8:00pm on Sat–Sun[2]"),
            listOf("Central Tiffin Room", "Malleshwaram", "Benne dosas, bajji, and coffee", "₹1–200",
                "7:00am–12:30pm, 4:00pm–9:30pm daily[4]"),
            listOf("Brahmins' Coffee Bar", "Shankarapura, near Shankar Mutt Road", "Idli, vada, and coffee", "₹1–200",
                "6:00am–12:00pm, 3:00pm–7:00pm Mon–Sat; closed Sun[5]"),
        )
        val profile = restaurantSummaryProfile(headers, rows)!!
        assertEquals(RestaurantSummaryProfile(0, 1, 2), profile)
        assertEquals(listOf(3, 4), profile.detailIndexes(headers))
        val representedIndexes = listOf(profile.name, profile.neighborhood, profile.dish) + profile.detailIndexes(headers)
        assertEquals(headers.indices.toList(), representedIndexes.sorted())
        assertEquals(listOf("₹1–200", "₹1–200", "₹1–200"), rows.map { it[profile.detailIndexes(headers).first()] })
        assertEquals(rows.map { it[4] }, rows.map { it[profile.detailIndexes(headers).last()] })
        assertEquals(headers, representedIndexes.sorted().map(headers::get))
    }

    @Test fun `permuted roles retain unknown details in their original order`() {
        val headers = listOf("Opening hours", "Restaurant name", "Source note", "Speciality", "Neighbourhood", "Approx. cost per person", "Extra qualification")
        val row = listOf("6:00am–12:00pm; closed Sun [5]", "Breakfast place [1]", "Not verified [11]",
            "Idli and coffee", "Central area", "₹1–200", "Full exception text; keep both clauses [9].")
        val profile = restaurantSummaryProfile(headers, listOf(row))!!

        assertEquals(RestaurantSummaryProfile(1, 4, 3), profile)
        assertEquals(listOf(0, 2, 5, 6), profile.detailIndexes(headers))
        assertEquals(
            listOf("6:00am–12:00pm; closed Sun [5]", "Not verified [11]", "₹1–200", "Full exception text; keep both clauses [9]."),
            profile.detailIndexes(headers).map(row::get),
        )
        assertEquals(headers.indices.toList(), (listOf(profile.name, profile.neighborhood, profile.dish) + profile.detailIndexes(headers)).sorted())
    }

    @Test fun `weak mixed or authored media action schemas keep the existing route`() {
        assertNull(restaurantSummaryProfile(listOf("Name", "Neighborhood", "Signature dish"), listOf(listOf("A", "B", "C"))))
        assertNull(restaurantSummaryProfile(listOf("Restaurant", "Neighborhood", "Cuisine"), listOf(listOf("A", "B", "C"))))
        assertNull(restaurantSummaryProfile(listOf("Restaurant", "Neighborhood", "Signature dish"), listOf(listOf("", "B", "C"))))
        assertNull(restaurantSummaryProfile(listOf("Restaurant", "Neighborhood", "Signature dish"), emptyList()))
        listOf("Feature", "Step", "Photo URL", "Website", "Booking action").forEach { guardedColumn ->
            assertNull(restaurantSummaryProfile(
                listOf("Restaurant", "Neighborhood", "Signature dish", guardedColumn),
                listOf(listOf("A", "B", "C", "Original authored value")),
            ))
        }
    }
}
