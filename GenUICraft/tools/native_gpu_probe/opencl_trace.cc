// Diagnostic-only OpenCL source capture for our standalone native probe.
// Never linked into the Android app. Run with LD_PRELOAD and CL_TRACE_DIR.
// Default mode returns every OpenCL call unchanged.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <string>

namespace {
using CreateSource = void* (*)(void*, unsigned, const char**, const size_t*, int*);
using BuildProgram = int (*)(void*, unsigned, const void**, const char*, void (*)(void*, void*), void*);
CreateSource original_create = nullptr;
BuildProgram original_build = nullptr;
std::atomic<unsigned> counter{0};

uint64_t fingerprint(const std::string& source) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char c : source) { hash ^= c; hash *= UINT64_C(1099511628211); }
  return hash;
}

bool read_file(const std::string& path, std::string& contents) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) return false;
  char block[8192];
  while (const size_t count = std::fread(block, 1, sizeof(block), file)) contents.append(block, count);
  const bool ok = !std::ferror(file);
  std::fclose(file);
  return ok;
}

void save(const char* suffix, const std::string& data, unsigned id) {
  const char* directory = std::getenv("CL_TRACE_DIR");
  if (!directory || !*directory || id > 10000) return;
  char path[1024];
  std::snprintf(path, sizeof(path), "%s/program_%05u.%s", directory, id, suffix);
  FILE* file = std::fopen(path, "wb");
  if (!file) { std::fprintf(stderr, "CL_TRACE cannot write %s\n", path); return; }
  std::fwrite(data.data(), 1, data.size(), file);
  std::fclose(file);
}

void* capture_create(void* context, unsigned count, const char** sources,
                     const size_t* lengths, int* error) {
  const unsigned id = ++counter;
  std::string source;
  for (unsigned i = 0; i < count; ++i)
    source.append(sources[i], lengths && lengths[i] ? lengths[i] : std::strlen(sources[i]));
  save("cl", source, id);
  std::string patched;
  const char* patches = std::getenv("CL_PATCH_DIR");
  if (patches && *patches) {
    char key[32];
    std::snprintf(key, sizeof(key), "%016llx", static_cast<unsigned long long>(fingerprint(source)));
    const std::string prefix = std::string(patches) + "/" + key;
    std::string expected;
    if (read_file(prefix + ".before.cl", expected)) {
      // Exact-byte verification protects against hash collisions and stale runtime sources.
      if (expected != source || !read_file(prefix + ".after.cl", patched) || patched.empty()) {
        std::fprintf(stderr, "CL_PATCH failed source guard key=%s\n", key);
        std::abort();
      }
      save("patched.cl", patched, id);
      std::fprintf(stderr, "CL_PATCH applied key=%s source=%u\n", key, id);
    }
  }
  const char* replacement = patched.c_str();
  const size_t replacement_size = patched.size();
  void* result = patched.empty()
      ? original_create(context, count, sources, lengths, error)
      : original_create(context, 1, &replacement, &replacement_size, error);
  std::fprintf(stderr, "CL_TRACE source=%u bytes=%zu program=%p status=%d\n",
               id, source.size(), result, error ? *error : 0);
  return result;
}

int capture_build(void* program, unsigned count, const void** devices,
                  const char* options, void (*callback)(void*, void*), void* user) {
  std::fprintf(stderr, "CL_TRACE build program=%p options=%s\n", program, options ? options : "");
  return original_build(program, count, devices, options, callback, user);
}
}  // namespace

extern "C" void* dlsym(void* handle, const char* name) {
  // Android libdl exports dlsym at LIBC; dlvsym is a separate untouched entry.
  using Dlsym = void* (*)(void*, const char*);
  static Dlsym real_dlsym = reinterpret_cast<Dlsym>(dlvsym(RTLD_NEXT, "dlsym", "LIBC"));
  if (!real_dlsym) std::abort();
  void* result = real_dlsym(handle, name);
  if (!result) return result;
  if (std::strcmp(name, "clCreateProgramWithSource") == 0) {
    original_create = reinterpret_cast<CreateSource>(result);
    return reinterpret_cast<void*>(&capture_create);
  }
  if (std::strcmp(name, "clBuildProgram") == 0) {
    original_build = reinterpret_cast<BuildProgram>(result);
    return reinterpret_cast<void*>(&capture_build);
  }
  return result;
}

// Public LiteRT C option; diagnostic interposition only. The host must verify
// the log and generated kernels because internal calls may bypass ELF preemption.
extern "C" int LrtSetGpuAcceleratorCompilationOptionsPrecision(void* options, int precision) {
  using Setter = int (*)(void*, int);
  static Setter original = reinterpret_cast<Setter>(
      dlsym(RTLD_NEXT, "LrtSetGpuAcceleratorCompilationOptionsPrecision"));
  if (!original) std::abort();
  const char* enable = std::getenv("CL_MIXED_ACCUM");
  const int resolved = enable && std::strcmp(enable, "1") == 0 && precision == 1 ? 3 : precision;
  const int status = original(options, resolved);
  std::fprintf(stderr, "CL_PRECISION requested=%d resolved=%d status=%d\n", precision, resolved, status);
  return status;
}
