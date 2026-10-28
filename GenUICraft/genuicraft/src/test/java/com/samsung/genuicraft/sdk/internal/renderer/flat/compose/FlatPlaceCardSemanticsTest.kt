package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatPlaceCardSemanticsTest {
    @Test
    fun preservesSourceRatingReviewTokenPhotoOrderAndActionUrls() {
        val props = mapOf<String, Any?>(
            "title" to "Sri Ranganathaswamy Temple",
            "rating" to "4.7",
            "reviewCount" to "(33585)",
            "address" to "CMFH+XJ4, Srirangapatna",
            "addressUrl" to "https://www.google.com/maps/search/?api=1&query=CMFH%2BXJ4",
            "images" to listOf(
                mapOf("url" to "https://lh3.googleusercontent.com/first.jpg", "title" to "First photo", "onClickUrl" to "http://www.aishwaryagroupofhotels.com/"),
                mapOf("url" to "https://lh3.googleusercontent.com/second.jpg", "title" to "Second photo"),
                mapOf("url" to "https://lh3.googleusercontent.com/third.jpg", "title" to "Third photo", "onClickUrl" to "javascript:alert(1)"),
            ),
            "actions" to listOf(
                mapOf("label" to "Call", "url" to "tel:+918212345678"),
                mapOf("label" to "Directions", "url" to "https://www.google.com/maps/dir/?api=1&destination=Mysore"),
            ),
        )

        val card = placeCardFromProps(props)

        assertEquals("4.7", card.rating)
        assertEquals("(33585)", card.reviewCount)
        assertEquals(listOf("First photo", "Second photo", "Third photo"), card.images.map { it.title })
        assertEquals("http://www.aishwaryagroupofhotels.com/", card.images[0].onClickUrl)
        assertNull(card.images[1].onClickUrl)
        assertNull(card.images[2].onClickUrl)
        assertEquals(listOf("Call", "Directions"), card.actions.map { it.label })
        assertEquals("tel:+918212345678", card.actions[0].url)
        assertEquals("https://www.google.com/maps/dir/?api=1&destination=Mysore", card.actions[1].url)
        assertEquals("Photo 3 of 3, loaded: Third photo", placeCardPhotoDescription(2, 3, "loaded", card.images[2].title))
        assertEquals("Photo 1 of 3, unavailable: First photo", placeCardPhotoDescription(0, 3, "unavailable", card.images[0].title))
    }

    @Test
    fun dropsUnsafeUrlsWithoutDroppingSafeSiblings() {
        val card = placeCardFromProps(
            mapOf(
                "addressUrl" to "javascript:alert(1)",
                "images" to listOf(
                    mapOf("url" to "http://lh3.googleusercontent.com/unsafe.jpg", "title" to "Unsafe"),
                    mapOf("url" to "https://lh3.googleusercontent.com/safe.jpg", "title" to "Safe"),
                ),
                "actions" to listOf(
                    mapOf("label" to "Bad", "url" to "file:///etc/passwd"),
                    mapOf("label" to "Website", "url" to "https://www.samsung.com/"),
                ),
            )
        )

        assertNull(card.addressUrl)
        assertEquals(listOf("Safe"), card.images.map { it.title })
        assertEquals(listOf("Website"), card.actions.map { it.label })
    }

    @Test
    fun strictExpressRoundTripKeepsStructuredPlaceCardProps() {
        val source = JsonParser.parseString(
            """{"root":"root","state":{},"elements":{"root":{"type":"PlaceCard","props":{"title":"Mysore Palace","rating":"4.7","reviewCount":"(0)","address":"Mysuru","addressUrl":"https://www.google.com/maps/search/?api=1&query=Mysuru","images":[{"url":"https://lh3.googleusercontent.com/a.jpg","title":"Front","onClickUrl":"http://www.mysorepalace.tv/360_Eng/index.html"},{"url":"https://lh3.googleusercontent.com/b.jpg","title":"Inside"}],"actions":[{"label":"Directions","url":"https://www.google.com/maps/dir/?api=1&destination=Mysuru"}]},"children":[]}}}"""
        ).asJsonObject

        val encoded = A2uiExpressCodec.encode(source)
        val decoded = A2uiExpressCodec.decode(encoded)
        val props = decoded.getAsJsonObject("elements").getAsJsonObject("root").getAsJsonObject("props")

        assertTrue(encoded.contains("PlaceCard("))
        assertEquals("(0)", props.get("reviewCount").asString)
        assertEquals(2, props.getAsJsonArray("images").size())
        assertEquals("http://www.mysorepalace.tv/360_Eng/index.html", props.getAsJsonArray("images")[0].asJsonObject.get("onClickUrl").asString)
        assertEquals(source.getAsJsonObject("elements").getAsJsonObject("root").getAsJsonObject("props"), props)
    }
}
