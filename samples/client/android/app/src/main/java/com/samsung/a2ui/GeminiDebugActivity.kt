package com.samsung.a2ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class GeminiDebugActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent {
      MaterialTheme {
        Surface {
          GeminiDebugScreen()
        }
      }
    }
  }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun GeminiDebugScreen() {
  val snapshot = GeminiDebugStore.snapshot()
  val scrollState = rememberScrollState()

  Scaffold(
    topBar = {
      TopAppBar(title = { Text("Gemini Debug") })
    }
  ) { padding ->
    Column(
      modifier = Modifier
        .fillMaxSize()
        .padding(padding)
        .padding(16.dp)
        .verticalScroll(scrollState)
    ) {
      DebugSection(
        title = "Prompt Sent to Gemini (Request JSON)",
        content = snapshot.requestJson ?: "No Gemini request captured yet."
      )
      DebugSection(
        title = "Gemini Raw Response",
        content = snapshot.rawResponse ?: "No Gemini response captured yet."
      )
      DebugSection(
        title = "A2UI JSON Used for Rendering",
        content = snapshot.uiJson ?: "No A2UI JSON captured yet."
      )
    }
  }
}

@Composable
private fun DebugSection(title: String, content: String) {
  Text(
    text = title,
    style = MaterialTheme.typography.titleMedium,
    modifier = Modifier.padding(bottom = 8.dp)
  )
  SelectionContainer {
    Text(
      text = content,
      style = MaterialTheme.typography.bodySmall,
      modifier = Modifier
        .padding(bottom = 20.dp)
        .background(MaterialTheme.colorScheme.surfaceVariant)
        .padding(12.dp)
    )
  }
}
