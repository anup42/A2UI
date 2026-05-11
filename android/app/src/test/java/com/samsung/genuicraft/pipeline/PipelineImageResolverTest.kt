package com.samsung.genuicraft.pipeline

import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineImageResolverTest {
    @Test
    fun shouldValidateRemoteImage_validatesPhotoUrlsOnly() {
        assertTrue(
            PipelineImageResolver.shouldValidateRemoteImage(
                "https://upload.wikimedia.org/wikipedia/commons/2/2c/BANGALORE_PALACE.jpg"
            )
        )
        assertFalse(
            PipelineImageResolver.shouldValidateRemoteImage(
                "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/geo-alt.svg"
            )
        )
        assertFalse(PipelineImageResolver.shouldValidateRemoteImage("../assets/local.jpg"))
    }

    @Test
    fun commonsSearchQuery_usesFilenameAndPromptContextWithoutUrlNoise() {
        val query = PipelineImageResolver.buildCommonsSearchQueryForTest(
            currentUrl = "https://upload.wikimedia.org/wikipedia/commons/9/99/Bangalore_Palace_from_the_sky.jpg",
            queryText = "show 4 day itinerary for vacation in bengaluru"
        )

        assertTrue(query.contains("Bangalore"))
        assertTrue(query.contains("Palace"))
        assertTrue(query.contains("bengaluru"))
        assertFalse(query.contains("https"))
        assertFalse(query.contains("jpg"))
    }

    @Test
    fun commonsImageExtraction_prefersOriginalImageInfoUrl() {
        val raw = """
            {
              "query": {
                "pages": {
                  "123": {
                    "title": "File:Lalbagh Botanical Garden Bangalore.jpg",
                    "imageinfo": [
                      {
                        "mime": "image/jpeg",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/1/11/Lalbagh_Botanical_Garden_Bangalore.jpg",
                        "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Lalbagh_Botanical_Garden_Bangalore.jpg/1200px-Lalbagh_Botanical_Garden_Bangalore.jpg",
                        "extmetadata": {
                          "ObjectName": { "value": "Lalbagh Botanical Garden Bangalore" }
                        }
                      }
                    ]
                  }
                }
              }
            }
        """.trimIndent()

        assertEquals(
            "https://upload.wikimedia.org/wikipedia/commons/1/11/Lalbagh_Botanical_Garden_Bangalore.jpg",
            PipelineImageResolver.extractCommonsImageUrlForTest(raw)
        )
    }

    @Test
    fun travelRowSearchQueries_prioritizeDestinationAndSpecificPlace() {
        val queries = PipelineImageResolver.travelRowSearchQueriesForTest(
            rowValues = mapOf(
                "area" to "West Coast",
                "morning" to "Arrive and settle in near Kata or Karon Beach",
                "afternoon" to "Relax on the beach and enjoy a sunset walk"
            ),
            queryText = "show 4day vacation itinerary in phuket"
        )

        assertTrue(queries.first().contains("phuket", ignoreCase = true))
        assertTrue(queries.first().contains("Kata", ignoreCase = true))
        assertTrue(queries.first().contains("Beach", ignoreCase = true))
        assertFalse(queries.first().contains("Arrive", ignoreCase = true))
    }
}
