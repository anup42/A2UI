// Selective FP16 compatibility correction for the pinned OpenCL accelerator.
// Only pinned SDK runtime dlsym imports are redirected. No vendor code or weights change.
// Inputs/outputs remain half; Q/DQ intermediates use float before the discrete round().
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <mutex>
#include <regex>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {
using Dlsym = void* (*)(void*, const char*);
using CreateSource = void* (*)(void*, unsigned, const char**, const size_t*, int*);
std::atomic<bool> enabled{false};
std::atomic<unsigned> programs{0}, blocks{0}, rejected{0}, sources{0};
std::atomic<CreateSource> create_source{nullptr};
Dlsym real_dlsym = nullptr;
std::mutex install_mutex;
constexpr const char* TAG = "GenUICraftFp16";
// LiteRT-LM's JNI binary also embeds a GPU compiler. Pin both static and plugin routes.
struct Module { const char* name; unsigned char build_id[16]; bool installed; };
Module modules[] = {
  {"libLiteRtOpenClAccelerator.so", {0x65,0x7a,0x86,0xd2,0x63,0x99,0xac,0xbf,0xe1,0x13,0xa1,0x5d,0x35,0x3c,0x61,0x01}, false},
  {"liblitertlm_jni.so", {0xf6,0x71,0x24,0xc0,0x09,0x7e,0x4b,0x50,0x42,0x02,0xf1,0xc6,0x78,0x3a,0xa1,0x11}, false},
  {"libLiteRt.so", {0x9f,0xf2,0x94,0x3b,0x6c,0x7e,0x19,0x8f,0x6c,0xbb,0x1c,0x01,0xba,0xb9,0x8f,0xa7}, false},
  {"libLiteRtTopKOpenClSampler.so", {0x34,0x0a,0xb1,0xf1,0xe1,0x76,0x9e,0xe0,0xae,0x93,0x8b,0x0f,0x16,0x92,0x50,0x6c}, false},
};

size_t count(const std::string& text, const std::string& needle) {
  size_t n = 0, p = 0;
  while ((p = text.find(needle, p)) != std::string::npos) { ++n; p += needle.size(); }
  return n;
}

bool correct_qdq(const std::string& source, std::string& output, unsigned& changed) {
  const size_t candidates = count(source, "half4 quantized_value = round(");
  if (!candidates) return true;
  // Full recognized block, not a broad replacement of half/float types.
  static const std::regex pattern(
      R"(half4 clamped_value = min\(\(half4\)\((shared_half4_[0-9]+\.[xyzw])\), max\(\(half4\)\((shared_half4_[0-9]+\.[xyzw])\), ([A-Za-z_][A-Za-z_0-9]*)\)\);\s*half4 quantized_value = round\(\(clamped_value - \(half4\)\((shared_half4_[0-9]+\.[xyzw])\)\) \* \(half4\)\((shared_half4_[0-9]+\.[xyzw])\)\);\s*half4 dequantized_value = quantized_value \* \(half4\)\((shared_half4_[0-9]+\.[xyzw])\) \+ \(half4\)\((shared_half4_[0-9]+\.[xyzw])\);\s*([A-Za-z_][A-Za-z_0-9]*) = dequantized_value;)"
  );
  output.clear();
  size_t cursor = 0;
  changed = 0;
  for (std::sregex_iterator it(source.begin(), source.end(), pattern), end; it != end; ++it) {
    const auto& m = *it;
    if (m[2] != m[4] || m[4] != m[7]) return false;
    const std::regex destination("\\bhalf4[ \\t]+" + m[8].str() + "\\b");
    if (!std::regex_search(source, destination)) return false;
    auto coeff = [&](int i) { return "(float4)(convert_float(" + m[i].str() + "))"; };
    output.append(source, cursor, m.position() - cursor);
    output += "float4 clamped_value = min(" + coeff(1) + ", max(" + coeff(2) +
        ", convert_float4(" + m[3].str() + ")));\n";
    output += "float4 quantized_value = round((clamped_value - " + coeff(4) + ") * " + coeff(5) + ");\n";
    output += "float4 dequantized_value = quantized_value * " + coeff(6) + " + " + coeff(7) + ";\n";
    output += m[8].str() + " = convert_half4(dequantized_value);";
    cursor = m.position() + m.length();
    ++changed;
  }
  output.append(source, cursor, std::string::npos);
  return changed == candidates && output.find("half4 quantized_value = round(") == std::string::npos;
}

void* corrected_create(void* context, unsigned n, const char** strings,
                       const size_t* lengths, int* error) {
  auto original = create_source.load();
  if (!original) { if (error) *error = -59; return nullptr; }
  if (!enabled.load()) return original(context, n, strings, lengths, error);
  ++sources;
  std::string source;
  for (unsigned i = 0; i < n; ++i) source.append(strings[i], lengths && lengths[i] ? lengths[i] : std::strlen(strings[i]));
  std::string patched;
  unsigned edits = 0;
  if (!correct_qdq(source, patched, edits)) {
    ++rejected;
    __android_log_print(ANDROID_LOG_ERROR, TAG, "Unrecognized half Q/DQ kernel; corrected FP16 initialization refused");
    if (error) *error = -59; // CL_INVALID_OPERATION
    return nullptr;
  }
  if (!edits) return original(context, n, strings, lengths, error);
  ++programs;
  blocks += edits;
  const char* p = patched.c_str();
  const size_t size = patched.size();
  return original(context, 1, &p, &size, error);
}

void* guarded_dlsym(void* handle, const char* name) {
  void* result = real_dlsym(handle, name);
  if (result && std::strcmp(name, "clCreateProgramWithSource") == 0) {
    create_source.store(reinterpret_cast<CreateSource>(result));
    return reinterpret_cast<void*>(&corrected_create);
  }
  return result;
}

bool expected_build(const dl_phdr_info* info, const Module& module) {
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const auto& ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_NOTE) continue;
    const auto* p = reinterpret_cast<const unsigned char*>(info->dlpi_addr + ph.p_vaddr);
    const auto* end = p + ph.p_memsz;
    while (p + sizeof(ElfW(Nhdr)) <= end) {
      const auto* h = reinterpret_cast<const ElfW(Nhdr)*>(p);
      const auto* name = p + sizeof(*h);
      const auto* data = name + ((h->n_namesz + 3) & ~3u);
      const auto* next = data + ((h->n_descsz + 3) & ~3u);
      if (next > end || next <= p) return false;
      if (h->n_type == NT_GNU_BUILD_ID && h->n_namesz == 4 &&
          std::memcmp(name, "GNU", 4) == 0 && h->n_descsz == sizeof(module.build_id))
        return std::memcmp(data, module.build_id, sizeof(module.build_id)) == 0;
      p = next;
    }
  }
  return false;
}

int page_protection(uintptr_t address) {
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) return -1;
  char line[1024], perms[5];
  unsigned long start, end;
  int result = -1;
  while (std::fgets(line, sizeof(line), maps)) {
    if (std::sscanf(line, "%lx-%lx %4s", &start, &end, perms) == 3 && address >= start && address < end) {
      result = (perms[0] == 'r' ? PROT_READ : 0) | (perms[1] == 'w' ? PROT_WRITE : 0) | (perms[2] == 'x' ? PROT_EXEC : 0);
      break;
    }
  }
  std::fclose(maps);
  return result;
}

int install_for_module(dl_phdr_info* info, size_t, void*) {
  const char* basename = std::strrchr(info->dlpi_name, '/');
  basename = basename ? basename + 1 : info->dlpi_name;
  Module* module = nullptr;
  for (auto& candidate : modules) if (std::strcmp(basename, candidate.name) == 0) module = &candidate;
  if (!module || module->installed) return 0;
  if (!expected_build(info, *module)) { __android_log_print(ANDROID_LOG_ERROR, TAG, "Build ID mismatch: %s", basename); return 0; }
  const ElfW(Dyn)* dyn = nullptr;
  for (int i = 0; i < info->dlpi_phnum; ++i)
    if (info->dlpi_phdr[i].p_type == PT_DYNAMIC)
      dyn = reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
  if (!dyn) return 0;
  auto ptr = [&](uintptr_t address) -> uintptr_t {
    return address >= info->dlpi_addr ? address : info->dlpi_addr + address;
  };
  const ElfW(Sym)* symbols = nullptr;
  const char* names = nullptr;
  const ElfW(Rela)* relocations = nullptr;
  size_t relocation_bytes = 0;
  for (const auto* d = dyn; d->d_tag != DT_NULL; ++d) {
    if (d->d_tag == DT_SYMTAB) symbols = reinterpret_cast<const ElfW(Sym)*>(ptr(d->d_un.d_ptr));
    if (d->d_tag == DT_STRTAB) names = reinterpret_cast<const char*>(ptr(d->d_un.d_ptr));
    if (d->d_tag == DT_JMPREL) relocations = reinterpret_cast<const ElfW(Rela)*>(ptr(d->d_un.d_ptr));
    if (d->d_tag == DT_PLTRELSZ) relocation_bytes = d->d_un.d_val;
  }
  if (!symbols || !names || !relocations) return 0;
  for (size_t i = 0; i < relocation_bytes / sizeof(*relocations); ++i) {
    const auto& r = relocations[i];
    if (ELF64_R_TYPE(r.r_info) != R_AARCH64_JUMP_SLOT ||
        std::strcmp(names + symbols[ELF64_R_SYM(r.r_info)].st_name, "dlsym") != 0) continue;
    auto* slot = reinterpret_cast<uintptr_t*>(info->dlpi_addr + r.r_offset);
    if (real_dlsym && *slot != reinterpret_cast<uintptr_t>(real_dlsym)) return 0;
    const int protection = page_protection(reinterpret_cast<uintptr_t>(slot));
    const size_t page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slot) & ~(page_size - 1));
    if (protection < 0 || mprotect(page, page_size, protection | PROT_WRITE) != 0) return 0;
    real_dlsym = reinterpret_cast<Dlsym>(*slot);
    __atomic_store_n(slot, reinterpret_cast<uintptr_t>(&guarded_dlsym), __ATOMIC_RELEASE);
    if (mprotect(page, page_size, protection) != 0) {
      __atomic_store_n(slot, reinterpret_cast<uintptr_t>(real_dlsym), __ATOMIC_RELEASE);
      return 0;
    }
    module->installed = true;
    __android_log_print(ANDROID_LOG_INFO, TAG, "Installed scoped Q/DQ adapter: %s", basename);
    return 0;
  }
  return 0;
}
} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_samsung_genuicraft_sdk_provider_GpuFp16Correction_nativeInstall(JNIEnv*, jobject) {
  std::lock_guard<std::mutex> guard(install_mutex);
  dl_iterate_phdr(install_for_module, nullptr);
  for (const auto& module : modules) if (!module.installed) return false;
  return true;
}
extern "C" JNIEXPORT void JNICALL
Java_com_samsung_genuicraft_sdk_provider_GpuFp16Correction_nativeSetEnabled(JNIEnv*, jobject, jboolean value) {
  enabled.store(value);
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_samsung_genuicraft_sdk_provider_GpuFp16Correction_nativeStatistics(JNIEnv* env, jobject) {
  const std::string stats = "source_programs=" + std::to_string(sources.load()) +
      "; qdq_programs=" + std::to_string(programs.load()) +
      "; qdq_blocks=" + std::to_string(blocks.load()) + "; rejected=" + std::to_string(rejected.load());
  return env->NewStringUTF(stats.c_str());
}
extern "C" JNIEXPORT jint JNICALL
Java_com_samsung_genuicraft_sdk_provider_GpuFp16Correction_nativePatchedBlocks(JNIEnv*, jobject) {
  return rejected.load() == 0 ? static_cast<jint>(blocks.load()) : -1;
}
