package com.samsung.a2ui.a2ui

import android.util.Log
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

class A2uiMessageProcessor {
  companion object {
    const val DEFAULT_SURFACE_ID = "@default"
  }

  private val surfaces: MutableMap<String, Surface> = mutableMapOf()
  private val json = Json { ignoreUnknownKeys = true }
  private val logTag = "A2UI-Processor"

  fun getSurfaces(): Map<String, Surface> = surfaces

  fun clearSurfaces() {
    surfaces.clear()
  }

  fun processMessages(messages: List<ServerToClientMessage>) {
    if (messages.isEmpty()) {
      Log.w(logTag, "processMessages called with empty list.")
      return
    }
    var beginCount = 0
    var updateCount = 0
    var dataCount = 0
    var deleteCount = 0
    for (message in messages) {
      message.beginRendering?.let {
        beginCount++
        handleBeginRendering(it, it.surfaceId)
      }
      message.surfaceUpdate?.let {
        updateCount++
        handleSurfaceUpdate(it, it.surfaceId)
      }
      message.dataModelUpdate?.let {
        dataCount++
        handleDataModelUpdate(it, it.surfaceId)
      }
      message.deleteSurface?.let {
        deleteCount++
        handleDeleteSurface(it)
      }
    }
    Log.i(
      logTag,
      "Processed messages: begin=$beginCount surfaceUpdate=$updateCount dataModelUpdate=$dataCount delete=$deleteCount"
    )
  }

  fun getData(node: ComponentNode, relativePath: String, surfaceId: String = DEFAULT_SURFACE_ID): Any? {
    val surface = getOrCreateSurface(surfaceId)
    val finalPath = resolvePath(relativePath, node.dataContextPath)
    return getDataByPath(surface.dataModel, finalPath)
  }

  fun setData(node: ComponentNode, relativePath: String, value: Any?, surfaceId: String = DEFAULT_SURFACE_ID) {
    val surface = getOrCreateSurface(surfaceId)
    val finalPath = resolvePath(relativePath, node.dataContextPath)
    setDataByPath(surface.dataModel, finalPath, value)
    rebuildComponentTree(surface)
  }

  fun resolvePath(path: String, dataContextPath: String?): String {
    if (path.isEmpty()) return dataContextPath ?: "/"
    if (path == ".") return dataContextPath ?: "/"
    if (path.startsWith("/")) return path

    val context = dataContextPath ?: "/"
    if (context == "/") {
      return "/$path"
    }

    return if (context.endsWith("/")) {
      "$context$path"
    } else {
      "$context/$path"
    }
  }

  private fun handleBeginRendering(message: BeginRenderingMessage, surfaceId: String) {
    val surface = getOrCreateSurface(surfaceId)
    surface.rootComponentId = message.root
    surface.styles = message.styles
    rebuildComponentTree(surface)
  }

  private fun handleSurfaceUpdate(message: SurfaceUpdateMessage, surfaceId: String) {
    val surface = getOrCreateSurface(surfaceId)
    for (component in message.components) {
      surface.components[component.id] = component
    }
    if (surface.rootComponentId.isNullOrBlank() && message.components.isNotEmpty()) {
      val rootId = message.components.firstOrNull { it.id == "root" }?.id ?: message.components.first().id
      surface.rootComponentId = rootId
      Log.w(logTag, "Missing beginRendering; defaulting root to $rootId")
    }
    rebuildComponentTree(surface)
  }

  private fun handleDataModelUpdate(message: DataModelUpdateMessage, surfaceId: String) {
    val surface = getOrCreateSurface(surfaceId)
    val path = message.path ?: "/"
    val payload = message.value ?: message.contents
    setDataByPath(surface.dataModel, path, payload)
    rebuildComponentTree(surface)
  }

  private fun handleDeleteSurface(message: DeleteSurfaceMessage) {
    surfaces.remove(message.surfaceId)
  }

  private fun rebuildComponentTree(surface: Surface) {
    var rootId = surface.rootComponentId
    if (rootId.isNullOrEmpty()) {
      surface.componentTree = null
      return
    }
    if (!surface.components.containsKey(rootId)) {
      val fallback = surface.components["root"]?.id ?: surface.components.keys.firstOrNull()
      if (fallback != null) {
        Log.w(logTag, "Root \"$rootId\" not found; falling back to \"$fallback\"")
        rootId = fallback
        surface.rootComponentId = fallback
      } else {
        surface.componentTree = null
        return
      }
    }

    val visited = mutableSetOf<String>()
    surface.componentTree = buildNodeRecursive(rootId, surface, visited, "/", "")
  }

  private fun buildNodeRecursive(
    baseComponentId: String,
    surface: Surface,
    visited: MutableSet<String>,
    dataContextPath: String,
    idSuffix: String
  ): ComponentNode? {
    val fullId = baseComponentId + idSuffix
    val componentData = surface.components[baseComponentId] ?: return null

    if (!visited.add(fullId)) {
      throw IllegalStateException("Circular dependency for component \"$fullId\".")
    }

    val type = componentData.type ?: return null
    val propsObj = componentData.props ?: JsonObject(emptyMap())
    val resolvedProps = mutableMapOf<String, Any?>()

    for ((key, value) in propsObj) {
      resolvedProps[key] = resolvePropertyValue(value, surface, visited, dataContextPath, idSuffix)
    }

    applyImplicitBoundValues(resolvedProps, surface, dataContextPath)
    visited.remove(fullId)

    return ComponentNode(
      id = fullId,
      type = type,
      properties = resolvedProps,
      dataContextPath = dataContextPath,
      weight = componentData.weight
    )
  }

  private fun resolvePropertyValue(
    value: JsonElement,
    surface: Surface,
    visited: MutableSet<String>,
    dataContextPath: String,
    idSuffix: String
  ): Any? {
    if (value is JsonPrimitive && value.isString) {
      val id = value.content
      if (surface.components.containsKey(id)) {
        return buildNodeRecursive(id, surface, visited, dataContextPath, idSuffix)
      }
      return id
    }

    if (value is JsonObject && value.containsKey("boundValue") && value.size == 1) {
      val wrapped = value["boundValue"] ?: return null
      return resolvePropertyValue(wrapped, surface, visited, dataContextPath, idSuffix)
    }

    if (value is JsonObject && isComponentArrayReference(value)) {
      value["explicitList"]?.let { explicit ->
        val list = explicit.jsonArray.mapNotNull { item ->
          val primitive = item as? JsonPrimitive ?: return@mapNotNull null
          if (primitive.isString) primitive.content else null
        }
        return list.mapNotNull { childId ->
          buildNodeRecursive(childId, surface, visited, dataContextPath, idSuffix)
        }
      }

      value["template"]?.let { template ->
        val templateObj = template.jsonObject
        val componentId = templateObj["componentId"]?.jsonPrimitive?.content ?: return emptyList<Any?>()
        val dataBinding = templateObj["dataBinding"]?.jsonPrimitive?.content ?: return emptyList<Any?>()
        val fullDataPath = resolvePath(dataBinding, dataContextPath)
        val data = getDataByPath(surface.dataModel, fullDataPath)

        if (data is List<*>) {
          return data.mapIndexedNotNull { index, _ ->
            val newSuffix = buildIndexSuffix(dataContextPath, index)
            val childPath = "$fullDataPath/$index"
            buildNodeRecursive(componentId, surface, visited, childPath, newSuffix)
          }
        }

        if (data is Map<*, *>) {
          return data.keys.mapNotNull { key ->
            val keyString = key?.toString() ?: return@mapNotNull null
            val newSuffix = ":$keyString"
            val childPath = "$fullDataPath/$keyString"
            buildNodeRecursive(componentId, surface, visited, childPath, newSuffix)
          }
        }

        return emptyList<Any?>()
      }
    }

    if (value is JsonArray) {
      return value.map { item ->
        resolvePropertyValue(item, surface, visited, dataContextPath, idSuffix)
      }
    }

    if (value is JsonObject) {
      val map = mutableMapOf<String, Any?>()
      for ((key, item) in value) {
        if (key == "path" && dataContextPath != "/" && item is JsonPrimitive && item.isString) {
          map[key] = trimTemplatePath(item.content)
        } else {
          map[key] = resolvePropertyValue(item, surface, visited, dataContextPath, idSuffix)
        }
      }
      return map
    }

    if (value is JsonPrimitive) {
      return primitiveToNative(value)
    }

    return null
  }

  private fun trimTemplatePath(path: String): String {
    var result = path
    result = result.replace(Regex("^\\.?/item"), "")
    result = result.replace(Regex("^\\.?/text"), "")
    result = result.replace(Regex("^\\.?/label"), "")
    result = result.replace(Regex("^\\.?/"), "")
    return result
  }

  private fun buildIndexSuffix(dataContextPath: String, index: Int): String {
    val parentIndices = dataContextPath.split("/").filter { it.matches(Regex("\\d+")) }
    val indices = parentIndices + index.toString()
    return ":${indices.joinToString(":")}"
  }

  private fun isComponentArrayReference(value: JsonObject): Boolean {
    return value.containsKey("explicitList") || value.containsKey("template")
  }

  private fun primitiveToNative(value: JsonPrimitive): Any? {
    if (value.isString) return value.content
    value.booleanOrNull?.let { return it }
    value.longOrNull?.let { return it }
    value.doubleOrNull?.let { return it }
    return null
  }

  private fun applyImplicitBoundValues(
    props: Map<String, Any?>,
    surface: Surface,
    dataContextPath: String
  ) {
    for (value in props.values) {
      applyImplicitBoundValues(value, surface, dataContextPath)
    }
  }

  private fun applyImplicitBoundValues(
    value: Any?,
    surface: Surface,
    dataContextPath: String
  ) {
    when (value) {
      is Map<*, *> -> {
        val boundValue = value["boundValue"]
        if (boundValue is Map<*, *>) {
          applyImplicitBoundValues(boundValue, surface, dataContextPath)
        }
        val path = value["path"] as? String
        val literal = extractLiteralValue(value)
        if (path != null && literal != null) {
          val resolvedPath = resolvePath(path, dataContextPath)
          if (getDataByPath(surface.dataModel, resolvedPath) == null) {
            setDataByPath(surface.dataModel, resolvedPath, literal)
          }
        }
        for (entry in value.values) {
          applyImplicitBoundValues(entry, surface, dataContextPath)
        }
      }
      is List<*> -> value.forEach { entry -> applyImplicitBoundValues(entry, surface, dataContextPath) }
    }
  }

  private fun extractLiteralValue(value: Map<*, *>): Any? {
    return when {
      value.containsKey("literalString") -> value["literalString"]
      value.containsKey("literalNumber") -> value["literalNumber"]
      value.containsKey("literalBoolean") -> value["literalBoolean"]
      value.containsKey("literalArray") -> value["literalArray"]
      value.containsKey("literal") -> value["literal"]
      else -> null
    }
  }

  private fun convertKeyValueArrayToMap(arr: List<JsonObject>): MutableMap<String, Any?> {
    val map = mutableMapOf<String, Any?>()
    for (item in arr) {
      val key = item["key"]?.jsonPrimitive?.content ?: continue
      val valueKey = item.keys.firstOrNull { it.startsWith("value") && it != "key" }
      val rawValue = valueKey?.let { item[it] }
      val parsed = when {
        rawValue == null -> null
        valueKey == "valueMap" && rawValue is JsonArray -> convertKeyValueArrayToMap(rawValue.mapNotNull { it as? JsonObject })
        rawValue is JsonPrimitive -> parseIfJsonString(primitiveToNative(rawValue))
        else -> jsonElementToNative(rawValue)
      }
      setDataByPath(map, key, parsed)
    }
    return map
  }

  private fun parseIfJsonString(value: Any?): Any? {
    val str = value as? String ?: return value
    val trimmed = str.trim()
    if (!((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]")))) {
      return str
    }

    return try {
      val element = json.parseToJsonElement(trimmed)
      jsonElementToNative(element)
    } catch (_: Exception) {
      str
    }
  }

  private fun jsonElementToNative(value: JsonElement): Any? {
    return when (value) {
      is JsonPrimitive -> primitiveToNative(value)
      is JsonArray -> value.map { jsonElementToNative(it) }.toMutableList()
      is JsonObject -> value.entries.associate { it.key to jsonElementToNative(it.value) }.toMutableMap()
      else -> null
    }
  }

  private fun setDataByPath(root: MutableMap<String, Any?>, path: String, value: Any?) {
    var finalValue = value

    if (value is JsonElement) {
      finalValue = jsonElementToNative(value)
    }

    if (finalValue is List<*>) {
      val jsonItems = finalValue.filterIsInstance<JsonObject>()
      if (jsonItems.isNotEmpty()) {
        finalValue = if (jsonItems.size == 1 && jsonItems[0]["key"]?.jsonPrimitive?.content == ".") {
          val item = jsonItems[0]
          val valueKey = item.keys.firstOrNull { it.startsWith("value") && it != "key" }
          val rawValue = valueKey?.let { item[it] }
          when {
            rawValue == null -> null
            valueKey == "valueMap" && rawValue is JsonArray -> convertKeyValueArrayToMap(rawValue.mapNotNull { it as? JsonObject })
            rawValue is JsonPrimitive -> parseIfJsonString(primitiveToNative(rawValue))
            else -> jsonElementToNative(rawValue)
          }
        } else {
          convertKeyValueArrayToMap(jsonItems)
        }
      }
    }

    val normalized = normalizePath(path)
    val segments = normalized.split("/").filter { it.isNotEmpty() }

    if (segments.isEmpty()) {
      if (finalValue is Map<*, *>) {
        root.clear()
        for ((key, v) in finalValue) {
          root[key.toString()] = v
        }
      }
      return
    }

    var current: Any? = root
    for (i in 0 until segments.size - 1) {
      val segment = segments[i]
      current = when (current) {
        is MutableMap<*, *> -> {
          @Suppress("UNCHECKED_CAST")
          val map = current as MutableMap<String, Any?>
          map.getOrPut(segment) { mutableMapOf<String, Any?>() }
        }
        is MutableList<*> -> {
          @Suppress("UNCHECKED_CAST")
          val list = current as MutableList<Any?>
          val index = segment.toIntOrNull() ?: return
          ensureListSize(list, index + 1)
          if (list[index] !is MutableMap<*, *> && list[index] !is MutableList<*>) {
            list[index] = mutableMapOf<String, Any?>()
          }
          list[index]
        }
        else -> return
      }
    }

    val lastSegment = segments.last()
    when (current) {
      is MutableMap<*, *> -> {
        @Suppress("UNCHECKED_CAST")
        val map = current as MutableMap<String, Any?>
        map[lastSegment] = finalValue
      }
      is MutableList<*> -> {
        @Suppress("UNCHECKED_CAST")
        val list = current as MutableList<Any?>
        val index = lastSegment.toIntOrNull() ?: return
        ensureListSize(list, index + 1)
        list[index] = finalValue
      }
    }
  }

  private fun ensureListSize(list: MutableList<Any?>, size: Int) {
    while (list.size < size) {
      list.add(null)
    }
  }

  private fun normalizePath(path: String): String {
    val dotPath = path.replace(Regex("\\[(\\d+)\\]"), ".$1")
    val segments = dotPath.split(".").filter { it.isNotBlank() }
    return "/" + segments.joinToString("/")
  }

  private fun getDataByPath(root: MutableMap<String, Any?>, path: String): Any? {
    val normalized = normalizePath(path)
    val segments = normalized.split("/").filter { it.isNotEmpty() }

    var current: Any? = root
    for (segment in segments) {
      current = when (current) {
        is Map<*, *> -> current[segment]
        is List<*> -> {
          val index = segment.toIntOrNull() ?: return null
          if (index in current.indices) current[index] else return null
        }
        else -> return null
      }
    }

    return current
  }

  private fun getOrCreateSurface(surfaceId: String): Surface {
    return surfaces.getOrPut(surfaceId) { Surface() }
  }
}
