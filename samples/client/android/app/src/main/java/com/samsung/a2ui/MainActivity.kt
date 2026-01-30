package com.samsung.a2ui

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.samsung.a2ui.ui.A2uiApp

class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent {
      MaterialTheme {
        Surface {
          A2uiApp(viewModel)
        }
      }
    }
    handleIntent(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    handleIntent(intent)
  }

  private fun handleIntent(intent: Intent) {
    val backend = intent.getStringExtra(EXTRA_BACKEND)?.lowercase()
    if (backend == "gemini") {
      viewModel.updateBackendMode(BackendMode.GEMINI_DIRECT)
    } else if (backend == "a2a") {
      viewModel.updateBackendMode(BackendMode.A2A_SERVER)
    }

    val prompt = intent.getStringExtra(EXTRA_PROMPT)
    if (!prompt.isNullOrBlank()) {
      Log.i(TAG, "Auto prompt received. backend=$backend length=${prompt.length}")
      viewModel.submitInput(prompt)
    }
  }

  companion object {
    private const val TAG = "A2UI-Activity"
    const val EXTRA_PROMPT = "a2ui_prompt"
    const val EXTRA_BACKEND = "a2ui_backend"
  }
}
