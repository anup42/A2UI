export function effect(callback) {
  if (typeof callback !== "function") {
    return () => {};
  }

  queueMicrotask(() => {
    try {
      callback();
    } catch (_err) {
      // Prevent renderer crashes from user callbacks.
    }
  });

  return () => {};
}
