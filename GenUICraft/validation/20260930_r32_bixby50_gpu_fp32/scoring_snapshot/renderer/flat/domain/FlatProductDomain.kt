package com.samsung.genuicraft.renderer.flat.domain

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ShoppingBag
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.samsung.genuicraft.renderer.flat.compose.RenderImage
import com.samsung.genuicraft.renderer.flat.compose.table.normalizeTableHeaderForMatch
import com.samsung.genuicraft.security.SafeContentPolicy

private val PRODUCT_NAME_TOKENS = setOf("name", "product", "productname", "title", "item", "model")
private val PRODUCT_IMAGE_TOKENS = setOf("image", "imageurl", "photo", "thumbnail", "picture")
private val PRODUCT_PRICE_TOKENS = setOf("price", "cost", "amount", "saleprice")
private val PRODUCT_RATING_TOKENS = setOf("rating", "score", "stars", "review")
private val PRODUCT_AVAILABILITY_TOKENS = setOf("availability", "stock", "status")
private val PRODUCT_SELLER_TOKENS = setOf("seller", "store", "vendor", "retailer")
private val PRODUCT_BADGE_TOKENS = setOf("badge", "label", "deal", "offer")
private val PRODUCT_URL_TOKENS = setOf("url", "link", "href", "producturl", "actionurl")

private fun productColumnIndex(headers: List<String>, tokens: Set<String>): Int =
    headers.indexOfFirst { normalizeTableHeaderForMatch(it).replace(" ", "") in tokens }

private fun productValue(row: List<String>, index: Int): String =
    row.getOrNull(index)?.trim().orEmpty()

/** Dedicated product-card route for product lookup and reusable entity intents. */
@Composable
internal fun renderProductRowsIfPossible(
    headers: List<String>,
    rows: List<List<String>>,
    onOpenUrl: (String) -> Unit,
    modifier: Modifier = Modifier,
    title: String? = null
): Boolean {
    if (rows.isEmpty()) return false
    val nameIndex = productColumnIndex(headers, PRODUCT_NAME_TOKENS)
    if (nameIndex < 0) return false
    val imageIndex = productColumnIndex(headers, PRODUCT_IMAGE_TOKENS)
    val priceIndex = productColumnIndex(headers, PRODUCT_PRICE_TOKENS)
    val ratingIndex = productColumnIndex(headers, PRODUCT_RATING_TOKENS)
    val availabilityIndex = productColumnIndex(headers, PRODUCT_AVAILABILITY_TOKENS)
    val sellerIndex = productColumnIndex(headers, PRODUCT_SELLER_TOKENS)
    val badgeIndex = productColumnIndex(headers, PRODUCT_BADGE_TOKENS)
    val urlIndex = productColumnIndex(headers, PRODUCT_URL_TOKENS)

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        title?.trim()?.takeIf { it.isNotBlank() }?.let { heading ->
            Text(
                text = heading,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier
                    .padding(horizontal = 16.dp, vertical = 4.dp)
                    .semantics { this.heading() }
            )
        }
        rows.take(20).forEach { row ->
            val name = productValue(row, nameIndex)
            if (name.isBlank()) return@forEach
            val imageUrl = productValue(row, imageIndex)
            val price = productValue(row, priceIndex)
            val rating = productValue(row, ratingIndex)
            val availability = productValue(row, availabilityIndex)
            val seller = productValue(row, sellerIndex)
            val badge = productValue(row, badgeIndex)
            val actionUrl = SafeContentPolicy.sanitizeActionUrl(productValue(row, urlIndex))

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .semantics {
                        contentDescription = listOf(name, price, rating, availability, seller)
                            .filter { it.isNotBlank() }
                            .joinToString(". ")
                    },
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                )
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (imageUrl.isNotBlank()) {
                        RenderImage(
                            props = mapOf(
                                "url" to imageUrl,
                                "alt" to name,
                                "height" to 160
                            ),
                            onOpenUrl = onOpenUrl,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.ShoppingBag,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = name,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.weight(1f)
                        )
                        if (badge.isNotBlank()) {
                            AssistChip(
                                onClick = {},
                                enabled = false,
                                label = { Text(badge) },
                                colors = AssistChipDefaults.assistChipColors(
                                    disabledLabelColor = MaterialTheme.colorScheme.onSecondaryContainer,
                                    disabledContainerColor = MaterialTheme.colorScheme.secondaryContainer
                                )
                            )
                        }
                    }
                    if (price.isNotBlank()) {
                        Text(
                            text = price,
                            style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    listOf(
                        "Rating" to rating,
                        "Availability" to availability,
                        "Seller" to seller
                    ).filter { (_, value) -> value.isNotBlank() }.forEach { (label, value) ->
                        Text(
                            text = "$label: $value",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (actionUrl != null) {
                        Button(
                            onClick = { onOpenUrl(actionUrl) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("View product")
                        }
                    }
                }
            }
        }
    }
    return true
}
