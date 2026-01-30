package com.samsung.a2ui.a2ui

object A2uiValueResolver {
  fun resolveString(value: Any?, node: ComponentNode, processor: A2uiMessageProcessor, surfaceId: String): String? {
    return when (value) {
      is String -> value
      is Number -> value.toString()
      is Boolean -> value.toString()
      is Map<*, *> -> {
        val resolved = resolveBoundValue(value, node, processor, surfaceId)
        when (resolved) {
          null -> null
          is String -> resolved
          else -> resolved.toString()
        }
      }
      else -> null
    }
  }

  fun resolveNumber(value: Any?, node: ComponentNode, processor: A2uiMessageProcessor, surfaceId: String): Double? {
    return when (value) {
      is Number -> value.toDouble()
      is String -> value.toDoubleOrNull()
      is Map<*, *> -> {
        val resolved = resolveBoundValue(value, node, processor, surfaceId)
        when (resolved) {
          is Number -> resolved.toDouble()
          is String -> resolved.toDoubleOrNull()
          else -> null
        }
      }
      else -> null
    }
  }

  fun resolveBoolean(value: Any?, node: ComponentNode, processor: A2uiMessageProcessor, surfaceId: String): Boolean? {
    return when (value) {
      is Boolean -> value
      is Map<*, *> -> {
        val resolved = resolveBoundValue(value, node, processor, surfaceId)
        when (resolved) {
          is Boolean -> resolved
          is String -> resolved.toBooleanStrictOrNull()
          else -> null
        }
      }
      else -> null
    }
  }

  fun resolveAny(value: Any?, node: ComponentNode, processor: A2uiMessageProcessor, surfaceId: String): Any? {
    return when (value) {
      is Map<*, *> -> {
        resolveBoundValue(value, node, processor, surfaceId)
      }
      else -> value
    }
  }

  private fun resolveBoundValue(
    value: Map<*, *>,
    node: ComponentNode,
    processor: A2uiMessageProcessor,
    surfaceId: String
  ): Any? {
    val wrapped = value["boundValue"] as? Map<*, *>
    if (wrapped != null) {
      return resolveBoundValue(wrapped, node, processor, surfaceId)
    }
    val path = value["path"] as? String
    val literal = extractLiteral(value)
    if (path != null) {
      val resolved = processor.getData(node, path, surfaceId)
      if (resolved != null) {
        return resolved
      }
    }
    return literal
  }

  private fun extractLiteral(value: Map<*, *>): Any? {
    return when {
      value.containsKey("literalString") -> value["literalString"]
      value.containsKey("literalNumber") -> value["literalNumber"]
      value.containsKey("literalBoolean") -> value["literalBoolean"]
      value.containsKey("literalArray") -> value["literalArray"]
      value.containsKey("literal") -> value["literal"]
      else -> null
    }
  }
}
