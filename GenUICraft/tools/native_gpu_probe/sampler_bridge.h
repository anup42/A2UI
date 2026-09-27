#pragma once

// Diagnostic only. Not a supported LiteRT API and never linked into the app.
// run_controls.py requires the exact Android C API library SHA-256:
// e9cbdddb0f1c693c549e1cde40bf90ad8aaa124d15944d0dd18faaf016dd6938.
// Pinned v0.16.1 source: c/engine_internal.h wraps both objects in a sole
// unique_ptr; executor_settings_base.h defines CPU=3 and GPU=4.
// The inline executor sampler setter has no exported symbol. Its field offset
// is verified against the actual GetSamplerBackend machine instructions below.
// Every guard must pass before writing any setting; do not generalize this ABI.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

namespace pinned_sampler_bridge {
static_assert(sizeof(void*) == 8, "Diagnostic bridge requires Android arm64");

inline void* symbol(const char* name) {
  void* result = dlsym(RTLD_DEFAULT, name);
  if (!result) throw std::runtime_error("Pinned sampler symbol is absent");
  return result;
}

inline uint32_t read32(const void* base, size_t offset) {
  uint32_t result;
  std::memcpy(&result, static_cast<const unsigned char*>(base) + offset, sizeof(result));
  return result;
}

inline void* unwrap(const void* wrapper) {
  if (!wrapper) throw std::runtime_error("Null native settings wrapper");
  void* result;
  std::memcpy(&result, wrapper, sizeof(result));
  if (!result) throw std::runtime_error("Null native settings object");
  return result;
}

inline void guard_instruction(void* function, size_t offset, uint32_t expected) {
  if (read32(function, offset) != expected)
    throw std::runtime_error("Pinned sampler ABI instruction guard failed");
}
}  // namespace pinned_sampler_bridge

inline void setCpuSamplerEngine(LiteRtLmEngineSettings* settings) {
  using namespace pinned_sampler_bridge;
  void* getter = symbol("_ZN6litert2lm17GetSamplerBackendERKNS0_19LlmExecutorSettingsE");
  // ldr w20,[x0,#0x8] (execution backend), ldr w9,[x0,#0x168] (sampler).
  guard_instruction(getter, 0x24, 0xb9400814);
  guard_instruction(getter, 0x28, 0xb9416809);
  auto executor = reinterpret_cast<void* (*)(void*)>(symbol(
      "_ZN6litert2lm14EngineSettings30GetMutableMainExecutorSettingsEv"));
  void* config = executor(unwrap(settings));
  if (!config) throw std::runtime_error("Null native executor settings");
  const uint32_t before = read32(config, 0x168);
  if (read32(config, 0x8) != 4 || (before != 0 && before != 4))
    throw std::runtime_error("Unexpected GPU executor or sampler backend");
  const uint32_t cpu = 3;
  std::memcpy(static_cast<unsigned char*>(config) + 0x168, &cpu, sizeof(cpu));
  if (read32(config, 0x168) != 3 || read32(config, 0x8) != 4)
    throw std::runtime_error("CPU engine sampler readback failed");
  std::fprintf(stderr, "PINNED_SAMPLER engine_backend=GPU sampler_before=%u sampler_after=CPU(3) ABI_guards=passed\n", before);
}

inline void setCpuSamplerSession(LiteRtLmSessionConfig* settings) {
  using namespace pinned_sampler_bridge;
  void* setter = symbol("_ZN6litert2lm13SessionConfig17SetSamplerBackendENS0_7BackendE");
  void* getter = symbol("_ZNK6litert2lm13SessionConfig17GetSamplerBackendEv");
  guard_instruction(setter, 0, 0xb900bc01);  // str w1,[x0,#0xbc]
  guard_instruction(getter, 0, 0xb940bc00);  // ldr w0,[x0,#0xbc]
  void* config = unwrap(settings);
  reinterpret_cast<void (*)(void*, int)>(setter)(config, 3);
  if (reinterpret_cast<int (*)(void*)>(getter)(config) != 3)
    throw std::runtime_error("CPU session sampler readback failed");
  std::fprintf(stderr, "PINNED_SAMPLER session_sampler=CPU(3) ABI_guards=passed\n");
}
