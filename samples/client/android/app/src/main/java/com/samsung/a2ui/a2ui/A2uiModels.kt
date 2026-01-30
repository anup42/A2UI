package com.samsung.a2ui.a2ui

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject


data class BeginRenderingMessage(
  val surfaceId: String,
  val root: String,
  val styles: Map<String, String> = emptyMap()
)

data class SurfaceUpdateMessage(
  val surfaceId: String,
  val components: List<ComponentInstance>
)

data class DataModelUpdateMessage(
  val surfaceId: String,
  val path: String?,
  val contents: List<JsonObject> = emptyList(),
  val value: JsonElement? = null
)

data class DeleteSurfaceMessage(
  val surfaceId: String
)

data class ServerToClientMessage(
  val beginRendering: BeginRenderingMessage? = null,
  val surfaceUpdate: SurfaceUpdateMessage? = null,
  val dataModelUpdate: DataModelUpdateMessage? = null,
  val deleteSurface: DeleteSurfaceMessage? = null
)

data class ComponentInstance(
  val id: String,
  val weight: Float? = null,
  val type: String? = null,
  val props: JsonObject? = null
)

data class Surface(
  var rootComponentId: String? = null,
  var componentTree: ComponentNode? = null,
  val dataModel: MutableMap<String, Any?> = mutableMapOf(),
  val components: MutableMap<String, ComponentInstance> = mutableMapOf(),
  var styles: Map<String, String> = emptyMap()
)

data class ComponentNode(
  val id: String,
  val type: String,
  val properties: Map<String, Any?>,
  val dataContextPath: String,
  val weight: Float? = null
)
