package com.samsung.genuicraft.sdk

import com.google.gson.Gson
import org.junit.Assert.*
import org.junit.Test

class ContentIntegrityTest {
    private fun doc(text: String) = GenUiCompiler.compile("<a2ui>\nroot=Column([answer])\nanswer=Text(${Gson().toJson(text)})\n</a2ui>")

    @Test fun `larger number does not conceal an omitted numeric fact`() {
        assertTrue(ContentIntegrity.check(GenUiRequest("Rain 15%. [2]"), doc("Rain 150%. [2]")).any { it.contains("numeric") })
    }

    @Test fun `invented source URL is rejected`() {
        assertTrue(ContentIntegrity.check(GenUiRequest("Read the source."), doc("Read the source. https://invented.example/")).any { it.contains("URLs must") })
    }

    @Test fun `missing citation is rejected even with complete prose`() {
        assertTrue(ContentIntegrity.check(GenUiRequest("Ready. [12]"), doc("Ready.")).any { it.contains("citation") })
    }

    @Test fun `sign units currency and invented facts cannot disappear into a word set`() {
        listOf("Price fell -5%" to "Price fell 5", "Cost ₹500" to "Cost 500", "Ready at 10:30" to "Ready at 10 30", "Mass 50 kg" to "Mass 50 mg", "Ready" to "Ready 99%").forEach { (source, output) ->
            assertTrue("Accepted changed fact: $source -> $output", ContentIntegrity.check(GenUiRequest(source), doc(output)).any { it.contains("numeric") })
        }
    }

    @Test fun `negation and repeated facts are preserved`() {
        assertTrue(ContentIntegrity.check(GenUiRequest("Do not go. Go tomorrow."), doc("Do go. Go tomorrow.")).isNotEmpty())
        assertTrue(ContentIntegrity.check(GenUiRequest("A 5. B 5."), doc("A 5. B.")).isNotEmpty())
    }

    @Test fun `layout properties cannot satisfy visible source wording`() {
        assertTrue(ContentIntegrity.check(GenUiRequest("vertical Ready"), doc("Ready")).any { it.contains("wording") })
    }

    @Test fun `model cannot add executable actions unrelated to supplied links`() {
        listOf("emitEvent(\"send\")", "openUrl(\"tel:123\")").forEach { action ->
            val value = GenUiCompiler.compile("<a2ui>\nroot=Column([answer,button])\nanswer=Text(\"Ready\")\nbutton=Button(\"Go\",onPress=$action)\n</a2ui>")
            assertTrue(ContentIntegrity.check(GenUiRequest("Ready"), value).any { it.contains("actions") })
        }
    }

    @Test fun `exact source links are allowed`() {
        val value = GenUiCompiler.compile("<a2ui>\nroot=Column([answer,button])\nanswer=Text(\"Ready\")\nbutton=Button(\"Source\",onPress=openUrl(\"https://example.com\"))\n</a2ui>")
        assertTrue(ContentIntegrity.check(GenUiRequest("Ready", sources=listOf(GenUiSource("7", "https://example.com", "Report"))), value).isEmpty())
    }

    @Test fun `unreachable source text cannot satisfy content preservation`() {
        val value = GenUiCompiler.compile("<a2ui>\nroot=Column([answer])\nanswer=Text(\"Ready\")\norphan=Text(\"Do not travel.\")\n</a2ui>")
        assertTrue(ContentIntegrity.check(GenUiRequest("Ready. Do not travel."), value).any { it.contains("unreachable") })
    }

    @Test fun `shared child must not display a source fact twice`() {
        val value = GenUiCompiler.compile("<a2ui>\nroot=Column([answer,answer])\nanswer=Text(\"Cost 500\")\n</a2ui>")
        assertTrue(ContentIntegrity.check(GenUiRequest("Cost 500"), value).any { it.contains("repeated child") })
    }
}
