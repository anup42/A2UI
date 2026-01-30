package com.samsung.a2ui.a2ui

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject

object A2uiJsonWriter {
  private val json = Json { prettyPrint = false }

  fun toJson(messages: List<ServerToClientMessage>): String {
    val element = toJsonElement(messages)
    return json.encodeToString(JsonElement.serializer(), element)
  }

  private fun toJsonElement(messages: List<ServerToClientMessage>): JsonElement {
    return buildJsonArray {
      messages.forEach { message ->
        add(messageToJson(message))
      }
    }
  }

  private fun messageToJson(message: ServerToClientMessage): JsonObject {
    return buildJsonObject {
      message.surfaceUpdate?.let { put("surfaceUpdate", surfaceUpdateToJson(it)) }
      message.beginRendering?.let { put("beginRendering", beginRenderingToJson(it)) }
      message.dataModelUpdate?.let { put("dataModelUpdate", dataModelUpdateToJson(it)) }
      message.deleteSurface?.let { put("deleteSurface", deleteSurfaceToJson(it)) }
    }
  }

  private fun surfaceUpdateToJson(message: SurfaceUpdateMessage): JsonObject {
    return buildJsonObject {
      put("surfaceId", JsonPrimitive(message.surfaceId))
      put(
        "components",
        buildJsonArray {
          message.components.forEach { component ->
            add(componentToJson(component))
          }
        }
      )
    }
  }

  private fun componentToJson(component: ComponentInstance): JsonObject {
    return buildJsonObject {
      put("id", JsonPrimitive(component.id))
      component.weight?.let { put("weight", JsonPrimitive(it)) }
      val type = component.type
      if (type != null) {
        val props = sanitize(component.props ?: JsonObject(emptyMap()))
        put(
          "component",
          buildJsonObject {
            put(type, props)
          }
        )
      }
    }
  }

  private fun beginRenderingToJson(message: BeginRenderingMessage): JsonObject {
    return buildJsonObject {
      put("surfaceId", JsonPrimitive(message.surfaceId))
      put("root", JsonPrimitive(message.root))
      if (message.styles.isNotEmpty()) {
        put(
          "styles",
          buildJsonObject {
            message.styles.forEach { (key, value) ->
              put(key, JsonPrimitive(value))
            }
          }
        )
      }
    }
  }

  private fun dataModelUpdateToJson(message: DataModelUpdateMessage): JsonObject {
    return buildJsonObject {
      put("surfaceId", JsonPrimitive(message.surfaceId))
      message.path?.let { put("path", JsonPrimitive(it)) }
      if (message.contents.isNotEmpty()) {
        put("contents", JsonArray(message.contents.map { sanitize(it) }))
      }
      message.value?.let { put("value", sanitize(it)) }
    }
  }

  private fun deleteSurfaceToJson(message: DeleteSurfaceMessage): JsonObject {
    return buildJsonObject {
      put("surfaceId", JsonPrimitive(message.surfaceId))
    }
  }

  private fun sanitize(element: JsonElement): JsonElement {
    return when (element) {
      is JsonObject -> {
        if (element.size == 1 && element.containsKey("boundValue")) {
          val inner = element["boundValue"] ?: return element
          sanitize(inner)
        } else {
          buildJsonObject {
            element.forEach { (key, value) ->
              put(key, sanitize(value))
            }
          }
        }
      }
      is JsonArray -> buildJsonArray { element.forEach { add(sanitize(it)) } }
      else -> element
    }
  }
}
