package com.samsung.a2ui.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.samsung.a2ui.a2ui.A2uiMessageProcessor
import com.samsung.a2ui.a2ui.A2uiValueResolver
import com.samsung.a2ui.a2ui.ComponentNode
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone

@Composable
fun A2uiRenderNode(
  node: ComponentNode,
  surfaceId: String,
  processor: A2uiMessageProcessor,
  onAction: (Map<String, Any?>) -> Unit,
  onValueChange: (ComponentNode, String, Any?, String) -> Unit,
  modifier: Modifier = Modifier
) {
  when (node.type) {
    "Text" -> {
      val text = A2uiValueResolver.resolveString(node.properties["text"], node, processor, surfaceId)
      val usageHint = node.properties["usageHint"] as? String
      Text(
        text = text ?: "",
        style = textStyleForHint(usageHint),
        modifier = modifier.padding(4.dp)
      )
    }
    "Image" -> {
      val url = A2uiValueResolver.resolveString(node.properties["url"], node, processor, surfaceId)
      val usageHint = node.properties["usageHint"] as? String
      val fit = node.properties["fit"] as? String
      if (url != null) {
        AsyncImage(
          model = url,
          contentDescription = null,
          contentScale = contentScaleForFit(fit),
          modifier = imageModifierForHint(usageHint).then(modifier)
        )
      }
    }
    "Icon" -> {
      val name = A2uiValueResolver.resolveString(node.properties["name"], node, processor, surfaceId) ?: ""
      Image(
        imageVector = iconForName(name),
        contentDescription = name,
        modifier = modifier.size(24.dp)
      )
    }
    "Row" -> {
      val children = node.properties["children"].asNodeList()
      Row(
        modifier = modifier
          .fillMaxWidth()
          .padding(4.dp),
        horizontalArrangement = rowArrangement(node.properties["distribution"] as? String),
        verticalAlignment = alignmentForRow(node.properties["alignment"] as? String)
      ) {
        children.forEach { child ->
          val childModifier = if (child.weight != null) Modifier.weight(child.weight) else Modifier
          A2uiRenderNode(
            node = child,
            surfaceId = surfaceId,
            processor = processor,
            onAction = onAction,
            onValueChange = onValueChange,
            modifier = childModifier.padding(4.dp)
          )
        }
      }
    }
    "Column" -> {
      val children = node.properties["children"].asNodeList()
      Column(
        modifier = modifier
          .fillMaxWidth()
          .padding(4.dp),
        verticalArrangement = columnArrangement(node.properties["distribution"] as? String),
        horizontalAlignment = alignmentForColumn(node.properties["alignment"] as? String)
      ) {
        children.forEach { child ->
          val childModifier = if (child.weight != null) Modifier.weight(child.weight) else Modifier
          A2uiRenderNode(
            node = child,
            surfaceId = surfaceId,
            processor = processor,
            onAction = onAction,
            onValueChange = onValueChange,
            modifier = childModifier.padding(4.dp)
          )
        }
      }
    }
    "List" -> {
      val children = node.properties["children"].asNodeList()
      val direction = node.properties["direction"] as? String
      if (direction == "horizontal") {
        Row(modifier = modifier.fillMaxWidth()) {
          children.forEach { child ->
            val childModifier = if (child.weight != null) Modifier.weight(child.weight) else Modifier
            A2uiRenderNode(
              child,
              surfaceId,
              processor,
              onAction,
              onValueChange,
              childModifier.padding(4.dp)
            )
          }
        }
      } else {
        Column(modifier = modifier.fillMaxWidth()) {
          children.forEach { child ->
            val childModifier = if (child.weight != null) Modifier.weight(child.weight) else Modifier
            A2uiRenderNode(
              child,
              surfaceId,
              processor,
              onAction,
              onValueChange,
              childModifier.padding(4.dp)
            )
          }
        }
      }
    }
    "Card" -> {
      val child = node.properties["child"] as? ComponentNode
      val children = node.properties["children"].asNodeList()
      Card(
        modifier = modifier
          .fillMaxWidth()
          .padding(6.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
      ) {
        Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
          if (child != null) {
            A2uiRenderNode(child, surfaceId, processor, onAction, onValueChange, Modifier.fillMaxWidth())
          } else {
            children.forEach { item ->
              A2uiRenderNode(item, surfaceId, processor, onAction, onValueChange, Modifier.fillMaxWidth())
            }
          }
        }
      }
    }
    "Button" -> {
      val child = node.properties["child"] as? ComponentNode
      Button(
        onClick = {
          val action = buildUserAction(node, surfaceId, processor)
          if (action.isNotEmpty()) {
            onAction(action)
          }
        },
        modifier = modifier.padding(4.dp)
      ) {
        if (child != null) {
          A2uiRenderNode(child, surfaceId, processor, onAction, onValueChange, Modifier)
        } else {
          Text("Action")
        }
      }
    }
    "TextField" -> {
      val label = A2uiValueResolver.resolveString(node.properties["label"], node, processor, surfaceId) ?: ""
      val valueSpec = node.properties["text"] ?: node.properties["value"]
      val textValue = A2uiValueResolver.resolveString(valueSpec, node, processor, surfaceId) ?: ""
      val type = node.properties["textFieldType"] as? String
        ?: node.properties["type"] as? String
        ?: node.properties["variant"] as? String
      val path = (valueSpec as? Map<*, *>)?.get("path") as? String

      Column(modifier = modifier.fillMaxWidth().padding(4.dp)) {
        OutlinedTextField(
          value = textValue,
          onValueChange = { newValue ->
            if (path != null) {
              onValueChange(node, path, newValue, surfaceId)
            }
          },
          label = { Text(label) },
          keyboardOptions = keyboardOptionsForType(type),
          modifier = Modifier.fillMaxWidth(),
          shape = MaterialTheme.shapes.medium
        )
        // Premium hint to match browser demo "PremiumTextField"
        Row(
           modifier = Modifier.padding(top = 4.dp),
           horizontalArrangement = Arrangement.spacedBy(4.dp),
           verticalAlignment = Alignment.CenterVertically
        ) {
           androidx.compose.material3.Surface(
             color = Color(0xFF6200EE),
             contentColor = Color.White,
             shape = MaterialTheme.shapes.extraSmall
           ) {
             Text(
               text = "CUSTOM",
               style = MaterialTheme.typography.labelSmall,
               fontWeight = FontWeight.Bold,
               modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
             )
           }
           Text(
             text = "This is a premium override of the standard TextField.",
             style = MaterialTheme.typography.bodySmall,
             color = Color.Gray
           )
        }
      }
    }
    "DateTimeInput" -> {
      val label = A2uiValueResolver.resolveString(node.properties["label"], node, processor, surfaceId)
        ?: "Date & Time"
      val value = A2uiValueResolver.resolveString(node.properties["value"], node, processor, surfaceId) ?: ""
      val path = (node.properties["value"] as? Map<*, *>)?.get("path") as? String

      OutlinedTextField(
        value = value,
        onValueChange = { newValue ->
          if (path != null) {
            onValueChange(node, path, newValue, surfaceId)
          }
        },
        label = { Text(label) },
        modifier = modifier.fillMaxWidth()
      )
    }
    "CheckBox", "Checkbox" -> {
      val label = A2uiValueResolver.resolveString(node.properties["label"], node, processor, surfaceId) ?: ""
      val checked = A2uiValueResolver.resolveBoolean(node.properties["value"], node, processor, surfaceId) ?: false
      val path = (node.properties["value"] as? Map<*, *>)?.get("path") as? String

      Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier.padding(4.dp)
      ) {
        Checkbox(
          checked = checked,
          onCheckedChange = { newValue ->
            if (path != null) {
              onValueChange(node, path, newValue, surfaceId)
            }
          }
        )
        Text(text = label)
      }
    }
    "Divider" -> {
      val axis = node.properties["axis"] as? String
      if (axis == "vertical") {
        Divider(modifier = modifier.width(1.dp).fillMaxHeight())
      } else {
        Divider(modifier = modifier.fillMaxWidth())
      }
    }
    "OrgChart" -> {
      OrgChartRenderer(node, surfaceId, processor, onAction, modifier)
    }
    "Modal" -> {
      val entry = node.properties["entryPointChild"] as? ComponentNode
      val content = node.properties["contentChild"] as? ComponentNode
      val showDialog = remember { mutableStateOf(false) }
      ModalContainer(
        showDialog = showDialog,
        entry = entry,
        content = content,
        surfaceId = surfaceId,
        processor = processor,
        onAction = onAction,
        onValueChange = onValueChange,
        modifier = modifier
      )
    }
    else -> {
      Text(
        text = "Unsupported component: ${node.type}",
        color = MaterialTheme.colorScheme.error,
        modifier = modifier.padding(4.dp)
      )
    }
  }
}

@Composable
private fun ModalContainer(
  showDialog: MutableState<Boolean>,
  entry: ComponentNode?,
  content: ComponentNode?,
  surfaceId: String,
  processor: A2uiMessageProcessor,
  onAction: (Map<String, Any?>) -> Unit,
  onValueChange: (ComponentNode, String, Any?, String) -> Unit,
  modifier: Modifier
) {
  Box(modifier = modifier) {
    if (entry != null) {
      Box(modifier = Modifier.clickable { showDialog.value = true }) {
        A2uiRenderNode(entry, surfaceId, processor, onAction, onValueChange)
      }
    }
  }

  if (showDialog.value && content != null) {
    AlertDialog(
      onDismissRequest = { showDialog.value = false },
      confirmButton = {
        TextButton(onClick = { showDialog.value = false }) {
          Text("Close")
        }
      },
      text = {
        A2uiRenderNode(content, surfaceId, processor, onAction, onValueChange)
      }
    )
  }
}

private fun Any?.asNodeList(): List<ComponentNode> {
  return (this as? List<*>)?.filterIsInstance<ComponentNode>() ?: emptyList()
}

@Composable
private fun textStyleForHint(usageHint: String?): androidx.compose.ui.text.TextStyle {
  return when (usageHint) {
    "h1" -> MaterialTheme.typography.headlineLarge
    "h2" -> MaterialTheme.typography.headlineMedium
    "h3" -> MaterialTheme.typography.headlineSmall
    "h4" -> MaterialTheme.typography.titleLarge
    "h5" -> MaterialTheme.typography.titleMedium
    "caption" -> MaterialTheme.typography.labelSmall
    else -> MaterialTheme.typography.bodyLarge
  }
}

private fun contentScaleForFit(fit: String?): ContentScale {
  return when (fit) {
    "cover" -> ContentScale.Crop
    "contain" -> ContentScale.Fit
    "fill" -> ContentScale.FillBounds
    "scale-down" -> ContentScale.Inside
    "none" -> ContentScale.None
    else -> ContentScale.Crop
  }
}

private fun imageModifierForHint(usageHint: String?): Modifier {
  return when (usageHint) {
    "icon" -> Modifier.size(24.dp)
    "avatar" -> Modifier.size(72.dp).clip(CircleShape)
    "header" -> Modifier.fillMaxWidth().height(180.dp)
    else -> Modifier.fillMaxWidth().height(160.dp)
  }
}

private fun rowArrangement(distribution: String?): Arrangement.Horizontal {
  return when (distribution) {
    "center" -> Arrangement.Center
    "end" -> Arrangement.End
    "spaceBetween" -> Arrangement.SpaceBetween
    "spaceAround" -> Arrangement.SpaceAround
    "spaceEvenly" -> Arrangement.SpaceEvenly
    else -> Arrangement.Start
  }
}

private fun columnArrangement(distribution: String?): Arrangement.Vertical {
  return when (distribution) {
    "center" -> Arrangement.Center
    "end" -> Arrangement.Bottom
    "spaceBetween" -> Arrangement.SpaceBetween
    "spaceAround" -> Arrangement.SpaceAround
    "spaceEvenly" -> Arrangement.SpaceEvenly
    else -> Arrangement.Top
  }
}

private fun alignmentForRow(alignment: String?): Alignment.Vertical {
  return when (alignment) {
    "center" -> Alignment.CenterVertically
    "end" -> Alignment.Bottom
    "stretch" -> Alignment.CenterVertically
    else -> Alignment.Top
  }
}

private fun alignmentForColumn(alignment: String?): Alignment.Horizontal {
  return when (alignment) {
    "center" -> Alignment.CenterHorizontally
    "end" -> Alignment.End
    "stretch" -> Alignment.CenterHorizontally
    else -> Alignment.Start
  }
}

private fun keyboardOptionsForType(type: String?): KeyboardOptions {
  return when (type) {
    "number" -> KeyboardOptions(keyboardType = KeyboardType.Number)
    "longText" -> KeyboardOptions(keyboardType = KeyboardType.Text)
    "date" -> KeyboardOptions(keyboardType = KeyboardType.Number)
    "shortText" -> KeyboardOptions(keyboardType = KeyboardType.Text)
    else -> KeyboardOptions(keyboardType = KeyboardType.Text)
  }
}

private fun iconForName(name: String): ImageVector {
  return when (name.lowercase(Locale.US)) {
    "calendar_today" -> Icons.Filled.CalendarToday
    "location_on" -> Icons.Filled.LocationOn
    "mail" -> Icons.Filled.Email
    "call" -> Icons.Filled.Call
    "check_circle" -> Icons.Filled.CheckCircle
    "arrow_downward" -> Icons.Filled.ArrowDropDown
    else -> Icons.Filled.Info
  }
}

@Composable
private fun OrgChartRenderer(
  node: ComponentNode,
  surfaceId: String,
  processor: A2uiMessageProcessor,
  onAction: (Map<String, Any?>) -> Unit,
  modifier: Modifier = Modifier
) {
  val chainList = node.properties["chain"] as? List<*> ?: return
  val chain = chainList.mapNotNull { item ->
    val map = item as? Map<*, *>
    if (map != null) {
      OrgChartNode(
            title = map["title"] as? String ?: "",
            name = map["name"] as? String ?: ""
      )
    } else null
  }

  Column(
    modifier = modifier.fillMaxWidth().padding(16.dp),
    horizontalAlignment = Alignment.CenterHorizontally,
    verticalArrangement = Arrangement.spacedBy(16.dp)
  ) {
    if (chain.isEmpty()) {
      Text("No hierarchy data", style = MaterialTheme.typography.bodyMedium)
    } else {
      chain.forEachIndexed { index, orgNode ->
        val isLast = index == chain.lastIndex

        // Node Card
        Card(
          modifier = Modifier
            .clickable {
              // Custom action handling for OrgChart node click
              val baseAction = node.properties["action"] as? Map<*, *>
              if (baseAction != null) {
                val actionName = baseAction["name"] as? String
                if (actionName != null) {
                  // 1. Resolve base context
                  val contextList = baseAction["context"] as? List<*>
                  val resolvedContext = mutableMapOf<String, Any?>()
                  contextList?.forEach { item ->
                    val ctx = item as? Map<*, *>
                    if (ctx != null) {
                       val key = ctx["key"] as? String
                       val valueSpec = ctx["value"]
                       if (key != null) {
                          resolvedContext[key] = A2uiValueResolver.resolveAny(valueSpec, node, processor, surfaceId)
                       }
                    }
                  }

                  // 2. Add node-specific context
                  resolvedContext["clickedNodeTitle"] = orgNode.title
                  resolvedContext["clickedNodeName"] = orgNode.name

                  // 3. Construct payload
                  val actionPayload = mutableMapOf<String, Any?>()
                  actionPayload["name"] = actionName
                  actionPayload["surfaceId"] = surfaceId
                  actionPayload["sourceComponentId"] = node.id
                  actionPayload["timestamp"] = isoTimestamp()
                  if (resolvedContext.isNotEmpty()) {
                    actionPayload["context"] = resolvedContext
                  }

                  onAction(mapOf("userAction" to actionPayload))
                }
              }
            }
            .fillMaxWidth(0.8f),
          colors = CardDefaults.cardColors(
            containerColor = if (isLast) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface
          ),
          border = if (isLast) androidx.compose.foundation.BorderStroke(2.dp, MaterialTheme.colorScheme.primary) else null,
          elevation = CardDefaults.cardElevation(defaultElevation = if (isLast) 8.dp else 2.dp)
        ) {
          Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
          ) {
            Text(
              text = orgNode.title,
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f)
            )
            Text(
              text = orgNode.name,
              style = MaterialTheme.typography.titleMedium,
              fontWeight = FontWeight.Medium
            )
          }
        }

        // Arrow
        if (!isLast) {
          Image(
             imageVector = Icons.Filled.ArrowDropDown,
             contentDescription = "Down",
             colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(Color.Gray)
          )
        }
      }
    }
  }
}

private data class OrgChartNode(
  val title: String,
  val name: String
)

private fun buildUserAction(
  node: ComponentNode,
  surfaceId: String,
  processor: A2uiMessageProcessor
): Map<String, Any?> {
  val action = node.properties["action"] as? Map<*, *> ?: return emptyMap()
  val name = action["name"] as? String ?: return emptyMap()
  val contextList = action["context"] as? List<*>
  val resolvedContext = mutableMapOf<String, Any?>()

  contextList?.forEach { item ->
    val ctx = item as? Map<*, *> ?: return@forEach
    val key = ctx["key"] as? String ?: return@forEach
    val valueSpec = ctx["value"]
    val resolved = A2uiValueResolver.resolveAny(valueSpec, node, processor, surfaceId)
    resolvedContext[key] = resolved
  }

  val actionPayload = mutableMapOf<String, Any?>()
  actionPayload["name"] = name
  actionPayload["surfaceId"] = surfaceId
  actionPayload["sourceComponentId"] = node.id
  actionPayload["timestamp"] = isoTimestamp()
  if (resolvedContext.isNotEmpty()) {
    actionPayload["context"] = resolvedContext
  }

  return mapOf("userAction" to actionPayload)
}

private fun isoTimestamp(): String {
  val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
  formatter.timeZone = TimeZone.getTimeZone("UTC")
  return formatter.format(System.currentTimeMillis())
}
