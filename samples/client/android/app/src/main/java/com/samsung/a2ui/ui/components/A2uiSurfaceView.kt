package com.samsung.a2ui.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.samsung.a2ui.a2ui.A2uiMessageProcessor
import com.samsung.a2ui.a2ui.ComponentNode
import com.samsung.a2ui.a2ui.Surface

@Composable
fun A2uiSurfaceView(
  surfaceId: String,
  surface: Surface,
  processor: A2uiMessageProcessor,
  onAction: (Map<String, Any?>) -> Unit,
  onValueChange: (ComponentNode, String, Any?, String) -> Unit
) {
  Card(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
    Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
      Text(
        text = "Surface: $surfaceId",
        style = MaterialTheme.typography.labelLarge
      )
      val tree = surface.componentTree
      if (tree != null) {
        A2uiRenderNode(
          node = tree,
          surfaceId = surfaceId,
          processor = processor,
          onAction = onAction,
          onValueChange = onValueChange
        )
      } else {
        Text("No content yet.")
      }
    }
  }
}
