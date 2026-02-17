export const SignalArray = Array;

export function signalArray(value = []) {
  return Array.isArray(value) ? [...value] : [];
}
