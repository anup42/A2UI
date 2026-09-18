package com.samsung.genuicraft.sdk.internal.renderer.flat.compose

import com.samsung.genuicraft.sdk.internal.renderer.*

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccessTime
import androidx.compose.material.icons.filled.AccountBalance
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Air
import androidx.compose.material.icons.filled.Article
import androidx.compose.material.icons.filled.AttachMoney
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Book
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.CreditCard
import androidx.compose.material.icons.filled.Directions
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.EventAvailable
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FilterList
import androidx.compose.material.icons.filled.FlightTakeoff
import androidx.compose.material.icons.filled.Help
import androidx.compose.material.icons.filled.Hotel
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.LocalHospital
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.MusicNote
import androidx.compose.material.icons.filled.Nightlight
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RateReview
import androidx.compose.material.icons.filled.Restaurant
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.School
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Storefront
import androidx.compose.material.icons.filled.Thermostat
import androidx.compose.material.icons.filled.TrendingDown
import androidx.compose.material.icons.filled.TrendingUp
import androidx.compose.material.icons.filled.Umbrella
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.WaterDrop
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material.icons.filled.Work
import androidx.compose.ui.graphics.vector.ImageVector
import java.util.Locale

/**
 * Maps an IR-declared icon *name* to a bundled Material vector.
 *
 * Before this existed, `Icon.props.name` was only ever treated as a URL: a bare
 * name like `calendar_today` failed
 * `SafeContentPolicy.sanitizeMediaUrl(..., ICON)` (it is not a URL), so
 * `RenderIcon` returned early and the element rendered **nothing at all**.
 *
 * Names are normalized so snake_case, kebab-case, camelCase and spaced forms all
 * resolve, and Bootstrap-icon names are accepted because the generator's icon
 * URLs come from that set — a spec may reasonably emit the bare name instead of
 * the CDN URL.
 *
 * `material-icons-extended` is already a dependency, so this adds no artifact.
 */
internal object FlatIconCatalog {

    private fun normalize(raw: String): String =
        raw.trim()
            .removeSuffix(".svg")
            .lowercase(Locale.US)
            .replace('-', '_')
            .replace(' ', '_')
            // camelCase -> snake_case is unnecessary after lowercasing; collapse
            // repeated separators instead so "geo__alt" matches "geo_alt".
            .replace(Regex("_+"), "_")
            .trim('_')

    private val VECTORS: Map<String, ImageVector> = buildMap {
        fun put(vector: ImageVector, vararg names: String) {
            names.forEach { name -> put(normalize(name), vector) }
        }

        put(Icons.Filled.CalendarToday, "calendar_today", "calendar", "calendar_event", "date")
        put(Icons.Filled.Schedule, "schedule", "clock", "time")
        put(Icons.Filled.AccessTime, "access_time", "clock_history", "duration")
        put(Icons.Filled.EventAvailable, "event_available", "event", "booking")
        put(Icons.Filled.LocationOn, "location_on", "location", "geo_alt", "pin")
        put(Icons.Filled.Place, "place", "map", "map_marker")
        put(Icons.Filled.Directions, "directions", "signpost_split", "route", "navigation")
        put(Icons.Filled.FlightTakeoff, "flight_takeoff", "flight", "airplane", "plane")
        put(Icons.Filled.Hotel, "hotel", "bed", "lodging", "accommodation")
        put(Icons.Filled.Restaurant, "restaurant", "cup_hot", "food", "dining")
        put(Icons.Filled.Storefront, "storefront", "shop", "store", "market")
        put(Icons.Filled.ShoppingCart, "shopping_cart", "cart", "basket")
        put(Icons.Filled.AttachMoney, "attach_money", "currency_dollar", "money", "price", "cost")
        put(Icons.Filled.CreditCard, "credit_card", "card", "payment")
        put(Icons.Filled.AccountBalance, "account_balance", "bank", "building")
        put(Icons.Filled.TrendingUp, "trending_up", "graph_up", "arrow_up_right", "increase")
        put(Icons.Filled.TrendingDown, "trending_down", "graph_down", "decrease")
        put(Icons.Filled.WbSunny, "wb_sunny", "sun", "sunny", "clear", "brightness_high")
        put(Icons.Filled.Cloud, "cloud", "clouds", "cloudy", "overcast")
        put(Icons.Filled.WaterDrop, "water_drop", "water", "droplet", "humidity", "rain")
        put(Icons.Filled.Umbrella, "umbrella", "rainy", "precipitation")
        put(Icons.Filled.Air, "air", "wind", "breeze")
        put(Icons.Filled.Thermostat, "thermostat", "thermometer", "temperature")
        put(Icons.Filled.Nightlight, "nightlight", "moon", "moon_stars", "night")
        put(Icons.Filled.Bolt, "bolt", "lightning", "lightning_charge", "energy", "power")
        put(Icons.Filled.Star, "star", "star_fill", "rating", "favourite")
        put(Icons.Filled.Favorite, "favorite", "heart", "heart_fill", "like")
        put(Icons.Filled.RateReview, "rate_review", "review", "chat_quote", "feedback")
        put(Icons.Filled.Person, "person", "user", "profile", "guest")
        put(Icons.Filled.Call, "call", "phone", "telephone", "telephone_fill")
        put(Icons.Filled.Email, "email", "mail", "envelope", "message")
        put(Icons.Filled.Link, "link", "link_45deg", "url", "hyperlink")
        put(Icons.Filled.Language, "language", "globe", "web", "translate")
        put(Icons.Filled.Article, "article", "newspaper", "news", "file_text", "document")
        put(Icons.Filled.Book, "book", "journal", "reading")
        put(Icons.Filled.School, "school", "mortarboard", "education", "study")
        put(Icons.Filled.Work, "work", "briefcase", "job", "career")
        put(Icons.Filled.LocalHospital, "local_hospital", "capsule", "health", "medical")
        put(Icons.Filled.MusicNote, "music_note", "music", "song", "track", "playlist")
        put(Icons.Filled.PlayArrow, "play_arrow", "play", "play_fill", "start")
        put(Icons.Filled.CheckCircle, "check_circle", "check2_circle", "success", "complete", "done")
        put(Icons.Filled.Check, "check", "check2", "tick")
        put(Icons.Filled.Close, "close", "x", "cancel", "remove")
        put(Icons.Filled.Warning, "warning", "exclamation_triangle", "alert", "caution")
        put(Icons.Filled.Error, "error", "exclamation_circle", "failure", "critical")
        put(Icons.Filled.Info, "info", "info_circle", "information", "note")
        put(Icons.Filled.Notifications, "notifications", "bell", "alerts")
        put(Icons.Filled.Search, "search", "magnifying_glass", "find")
        put(Icons.Filled.FilterList, "filter_list", "filter", "funnel", "sort")
        put(Icons.Filled.Settings, "settings", "gear", "config", "preferences")
        put(Icons.Filled.Share, "share", "share_fill")
        put(Icons.Filled.ContentCopy, "content_copy", "copy", "clipboard")
        put(Icons.Filled.Download, "download", "save", "export")
        put(Icons.Filled.Edit, "edit", "pencil", "compose")
        put(Icons.Filled.Add, "add", "plus", "new")
        put(Icons.Filled.Lock, "lock", "secure", "private")
        put(Icons.Filled.Image, "image", "photo", "picture")
        // Observed in the corpus but previously unmapped, so these rendered
        // nothing at all.
        put(Icons.Filled.Help, "question_circle", "help", "question", "faq", "support")
        put(Icons.Filled.Build, "wrench_adjustable", "wrench", "tools", "build", "maintenance")
        put(Icons.Filled.Chat, "chat_left_text", "chat", "comment", "discussion")
    }

    private val MEDIA_SUFFIXES = listOf(".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico")

    /**
     * Returns the bundled vector for an IR-declared icon *name*, or null when the
     * value is a URL or asset path that the Coil path should handle.
     *
     * Deliberately conservative: anything containing a path separator or ending
     * in a media suffix is rejected outright. Matching on a path's last segment
     * would hijack real bundled assets — the generator emits paths like
     * `../assets/r_000001_01_1_building.svg`, and a file that happened to be
     * named `star.svg` would otherwise silently render a Material vector instead
     * of the intended image.
     */
    fun vectorFor(name: String?): ImageVector? {
        if (name.isNullOrBlank()) return null
        val trimmed = name.trim()
        if (trimmed.contains("://") ||
            trimmed.contains('/') ||
            trimmed.contains('\\') ||
            MEDIA_SUFFIXES.any { suffix -> trimmed.endsWith(suffix, ignoreCase = true) }
        ) {
            return null
        }
        return VECTORS[normalize(trimmed)]
    }

    /** Exposed for tests: the number of distinct names understood. */
    internal fun nameCount(): Int = VECTORS.size
}
