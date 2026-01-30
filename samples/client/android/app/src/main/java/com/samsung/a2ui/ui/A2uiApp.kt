package com.samsung.a2ui.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import android.content.Intent
import com.samsung.a2ui.RendererMode
import androidx.lifecycle.viewmodel.compose.viewModel
import com.samsung.a2ui.MainViewModel
import com.samsung.a2ui.ui.components.A2uiSurfaceView
import com.samsung.a2ui.ui.components.A2uiWebView
import com.samsung.a2ui.GeminiDebugActivity

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun A2uiApp(viewModel: MainViewModel = viewModel()) {
  val surfaces = viewModel.surfaces
  val isLoading = viewModel.isLoading
  val error = viewModel.errorMessage
  val rendererMode = viewModel.rendererMode
  val geminiApiKey = viewModel.geminiApiKey
  val geminiModel = viewModel.geminiModel
  val cacheStatus = viewModel.cacheStatus
  val processor = viewModel.messageProcessor
  val litPayload = viewModel.litPayload
  val context = LocalContext.current

  var prompt by remember { mutableStateOf("") }

  Scaffold(
    topBar = {
      TopAppBar(title = { Text("A2UI Android Demo") })
    }
  ) { padding ->
    Column(
      modifier = Modifier
        .fillMaxSize()
        .padding(padding)
        .padding(16.dp)
    ) {
      OutlinedTextField(
        value = geminiApiKey,
        onValueChange = viewModel::updateGeminiApiKey,
        label = { Text("Gemini API Key") },
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
        modifier = Modifier.fillMaxWidth()
      )

      Spacer(modifier = Modifier.height(12.dp))

      OutlinedTextField(
        value = geminiModel,
        onValueChange = { viewModel.updateGeminiModel(it) },
        label = { Text("Gemini Model") },
        placeholder = { Text("gemini-2.5-flash-lite") },
        modifier = Modifier.fillMaxWidth()
      )

      Spacer(modifier = Modifier.height(12.dp))

      Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Renderer:", style = MaterialTheme.typography.labelLarge)
        Spacer(modifier = Modifier.width(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
          RadioButton(
            selected = rendererMode == RendererMode.NATIVE,
            onClick = { viewModel.updateRendererMode(RendererMode.NATIVE) }
          )
          Text("Native")
        }
        Spacer(modifier = Modifier.width(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
          RadioButton(
            selected = rendererMode == RendererMode.LIT_WEBVIEW,
            onClick = { viewModel.updateRendererMode(RendererMode.LIT_WEBVIEW) }
          )
          Text("Lit (WebView)")
        }
      }

      Spacer(modifier = Modifier.height(12.dp))

      OutlinedTextField(
        value = prompt,
        onValueChange = { prompt = it },
        label = { Text("Prompt or A2UI JSON") },
        modifier = Modifier.fillMaxWidth(),
        minLines = 3
      )

      Spacer(modifier = Modifier.height(12.dp))

      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End,
        verticalAlignment = Alignment.CenterVertically
      ) {
        Button(onClick = {
          context.startActivity(Intent(context, GeminiDebugActivity::class.java))
        }) {
          Text("Gemini Debug")
        }
        Spacer(modifier = Modifier.width(12.dp))
        Button(
          onClick = { viewModel.clearCacheForPrompt(prompt) },
          enabled = prompt.isNotBlank()
        ) {
          Text("Clear Cache")
        }
        Spacer(modifier = Modifier.width(12.dp))
        Button(onClick = { viewModel.submitInput(prompt) }) {
          Text("Send")
        }
      }

      if (isLoading) {
        Spacer(modifier = Modifier.height(12.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
          CircularProgressIndicator()
        }
      }

      if (!error.isNullOrEmpty()) {
        Spacer(modifier = Modifier.height(12.dp))
        Text(text = error, color = MaterialTheme.colorScheme.error)
      }

      if (!cacheStatus.isNullOrEmpty()) {
        Spacer(modifier = Modifier.height(12.dp))
        Text(text = cacheStatus, color = MaterialTheme.colorScheme.primary)
      }

      Spacer(modifier = Modifier.height(12.dp))

      if (rendererMode == RendererMode.LIT_WEBVIEW) {
        A2uiWebView(
          payload = litPayload,
          modifier = Modifier
            .fillMaxWidth()
            .weight(1f)
        )
      } else {
        LazyColumn(
          modifier = Modifier
            .fillMaxWidth()
            .weight(1f)
        ) {
          items(surfaces.entries.toList(), key = { it.key }) { (surfaceId, surface) ->
            A2uiSurfaceView(
              surfaceId = surfaceId,
              surface = surface,
              processor = processor,
              onAction = viewModel::sendUserAction,
              onValueChange = viewModel::updateBoundValue
            )
          }
        }
      }
    }
  }
}
