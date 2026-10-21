// nouveau machine-code cache (gl_nvc0_cache).
//
// GL program binaries (gl_program_binaries) skip Mesa's GLSL front end, but loading one still runs nouveau's
// TGSI -> Maxwell code generator (nv50_ir) for every stage, eagerly in nvc0_sp_state_create. For the translated Xenos
// shaders that is 0.2-35 s per program in Eden (more on the Switch). This file caches its result.
//
// The app links with -Wl,--wrap=nvc0_program_translate: every call from Mesa's other objects (nvc0_state.c,
// nvc0_shader_state.c) lands in __wrap_nvc0_program_translate. The translation depends only on the TGSI tokens, the
// shader type, the chipset and two inputs (vp.num_ucps, cp.smem_size); every other field of the program is zero
// before it (nvc0_sp_state_create callocs it, nvc0_program_destroy memsets it). So the fields after the call are
// stored, and a hit writes them back with fresh copies of the code / relocation / fixup arrays.
//
// struct nvc0_program layout (devkitPro Mesa 20.1, from the disassembly of nvc0_program_translate): pipe_shader_state
// {type @0, tokens @8, stream_output.num_outputs @24}, type @296, need_tls @298, num_gprs @299, code @304,
// code_size @316, hdr @324, vp.num_ucps @418, cp.smem_size @444, cp.syms @448, num_barriers @464, relocs @472,
// fixups @480, tfb @488, mem @496. RelocInfo = 16-byte header + count x 20-byte entries (no pointers); FixupInfo =
// 8-byte header + count x {FixupApply apply; uint32 val} (16 bytes). The apply function pointer is stored relative
// to nvc0_program_translate together with its first 16 code bytes, which are checked before a cached entry is used.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <xxhash.h>

#include <rex/cvar.h>
#include <rex/logging.h>

#include "gl_context.h"

REXCVAR_DEFINE_BOOL(gl_nvc0_cache, true, "GPU",
                    "GL backend: cache nouveau's generated GPU code (gl_nvc0_cache_path) so shaders load without running "
                    "the code generator again");
REXCVAR_DEFINE_STRING(gl_nvc0_cache_path, "sdmc:/oblivion/cache/gl_nvc0.bin", "GPU",
                      "GL backend: nouveau GPU code cache file (gl_nvc0_cache); shipped together with gl_binaries.bin");

extern "C" bool __real_nvc0_program_translate(void* prog, uint16_t chipset, void* debug);

namespace {
constexpr size_t kPipeType = 0, kPipeTokens = 8, kSoNumOutputs = 24;
constexpr size_t kRegionBegin = 296, kRegionEnd = 496;  // type .. tfb (mem excluded)
constexpr size_t kType = 296, kCode = 304, kCodeSize = 316, kNumUcps = 418, kSmemSize = 444, kSyms = 448;
constexpr size_t kRelocs = 472, kFixups = 480, kTfb = 488;
constexpr uint8_t kShaderCompute = 5;
constexpr char kMagic[8] = {'R', 'X', 'N', 'V', 'C', '0', '0', '1'};

// tgsi_parse.h (inline there): the first token is struct tgsi_header {HeaderSize:8; BodySize:24}.
unsigned TgsiNumTokens(const void* tokens) {
  uint32_t header;
  std::memcpy(&header, tokens, 4);
  return (header & 0xFF) + (header >> 8);
}

template <typename T>
T& At(void* prog, size_t off) {
  return *reinterpret_cast<T*>(static_cast<char*>(prog) + off);
}

struct Entry {
  std::vector<uint8_t> blob;
};

std::mutex g_mutex;
bool g_loaded = false, g_enabled = false;
std::string g_path;
std::unordered_map<uint64_t, Entry> g_entries;
std::atomic<uint32_t> g_hits{0}, g_misses{0}, g_stale{0};

template <typename T>
void Put(std::vector<uint8_t>& out, const T& v) {
  const size_t n = out.size();
  out.resize(n + sizeof(T));
  std::memcpy(out.data() + n, &v, sizeof(T));
}
void PutBytes(std::vector<uint8_t>& out, const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  out.insert(out.end(), b, b + n);
}

// Caller holds g_mutex.
void LoadLocked() {
  const uint64_t fingerprint = rex::graphics::gl::MesaBinaryFingerprint();
  // The context's own builtin programs translate before EnableProgramBinaries computes the fingerprint: pass them
  // through uncached and load on a later call.
  if (!fingerprint && REXCVAR_GET(gl_nvc0_cache)) return;
  g_loaded = true;
  if (!REXCVAR_GET(gl_nvc0_cache)) {
    REXGPU_INFO("GL nvc0 cache: off");
    return;
  }
  g_path = REXCVAR_GET(gl_nvc0_cache_path);
  std::vector<uint8_t> file;
  if (FILE* f = std::fopen(g_path.c_str(), "rb")) {
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size > 0 && size < (256l << 20)) {
      file.resize(size_t(size));
      if (std::fread(file.data(), 1, file.size(), f) != file.size()) file.clear();
    }
    std::fclose(f);
  }
  constexpr size_t kHeader = sizeof(kMagic) + 8;
  uint64_t file_fp = 0;
  if (file.size() >= kHeader) std::memcpy(&file_fp, file.data() + sizeof(kMagic), 8);
  const bool ok = file.size() >= kHeader && !std::memcmp(file.data(), kMagic, sizeof(kMagic)) && file_fp == fingerprint;
  if (!ok) {
    FILE* f = std::fopen(g_path.c_str(), "wb");
    if (!f) {
      REXGPU_WARN("GL nvc0 cache: cannot create {} (cache off)", g_path);
      return;
    }
    std::fwrite(kMagic, 1, sizeof(kMagic), f);
    std::fwrite(&fingerprint, 8, 1, f);
    std::fclose(f);
    file.clear();
  }
  size_t pos = ok ? kHeader : file.size();
  uint32_t n = 0;
  while (pos + 12 <= file.size()) {
    uint64_t key;
    uint32_t size;
    std::memcpy(&key, &file[pos], 8);
    std::memcpy(&size, &file[pos + 8], 4);
    if (!size || pos + 12 + size > file.size()) break;
    g_entries[key].blob.assign(file.begin() + long(pos + 12), file.begin() + long(pos + 12 + size));
    pos += 12 + size;
    ++n;
  }
  g_enabled = true;
  REXGPU_INFO("GL nvc0 cache: {}: {} entries ({} KiB)", g_path, n, file.size() >> 10);
}

uint64_t Key(void* prog, uint16_t chipset) {
  const void* tokens = At<const void*>(prog, kPipeTokens);
  const size_t bytes = size_t(TgsiNumTokens(tokens)) * 4;
  const uint64_t seed = uint64_t(At<uint8_t>(prog, kType)) | uint64_t(chipset) << 8 |
                        uint64_t(At<uint8_t>(prog, kNumUcps)) << 24 | uint64_t(At<uint32_t>(prog, kSmemSize)) << 32;
  const uint64_t k = XXH3_64bits_withSeed(tokens, bytes, seed ^ 0x6E7663304E564331ull);
  return k ? k : 1;
}

const uint8_t* FnBase() { return reinterpret_cast<const uint8_t*>(&__real_nvc0_program_translate); }

// Serialises the program state produced by a successful translation. False if it cannot be cached.
bool Serialize(void* prog, std::vector<uint8_t>& out) {
  if (At<void*>(prog, kTfb) || At<void*>(prog, kSyms)) return false;
  PutBytes(out, static_cast<char*>(prog) + kRegionBegin, kRegionEnd - kRegionBegin);
  const uint32_t code_size = At<uint32_t>(prog, kCodeSize);
  const void* code = At<void*>(prog, kCode);
  if (code_size && !code) return false;
  Put<uint32_t>(out, code_size);
  PutBytes(out, code, code_size);
  const auto* relocs = At<const uint8_t*>(prog, kRelocs);
  const uint32_t reloc_bytes = relocs ? 16u + *reinterpret_cast<const uint32_t*>(relocs + 12) * 20u : 0u;
  Put<uint32_t>(out, reloc_bytes);
  PutBytes(out, relocs, reloc_bytes);
  const auto* fixups = At<const uint8_t*>(prog, kFixups);
  const uint32_t fixup_count = fixups ? *reinterpret_cast<const uint32_t*>(fixups) : 0u;
  Put<uint32_t>(out, fixups ? fixup_count : 0xFFFFFFFFu);
  for (uint32_t i = 0; i < fixup_count; ++i) {
    const uint8_t* e = fixups + 8 + size_t(i) * 16;
    const uint8_t* apply = *reinterpret_cast<const uint8_t* const*>(e);
    Put<int64_t>(out, int64_t(apply - FnBase()));
    Put<uint32_t>(out, *reinterpret_cast<const uint32_t*>(e + 8));
    PutBytes(out, apply, 16);
  }
  return true;
}

// Restores a cached translation into prog (zeroed except pipe / type / inputs). False = stale entry, prog untouched.
bool Restore(void* prog, const std::vector<uint8_t>& blob) {
  size_t pos = 0;
  auto get = [&](void* dst, size_t n) {
    if (pos + n > blob.size()) return false;
    std::memcpy(dst, blob.data() + pos, n);
    pos += n;
    return true;
  };
  uint8_t region[kRegionEnd - kRegionBegin];
  uint32_t code_size = 0, reloc_bytes = 0, fixup_count = 0;
  if (!get(region, sizeof(region)) || !get(&code_size, 4) || pos + code_size > blob.size()) return false;
  const size_t code_pos = pos;
  pos += code_size;
  if (!get(&reloc_bytes, 4) || pos + reloc_bytes > blob.size()) return false;
  const size_t reloc_pos = pos;
  pos += reloc_bytes;
  if (!get(&fixup_count, 4)) return false;
  const bool has_fixups = fixup_count != 0xFFFFFFFFu;
  if (!has_fixups) fixup_count = 0;
  std::vector<std::pair<const void*, uint32_t>> fix(fixup_count);
  for (uint32_t i = 0; i < fixup_count; ++i) {
    int64_t off;
    uint32_t val;
    uint8_t check[16];
    if (!get(&off, 8) || !get(&val, 4) || !get(check, 16)) return false;
    const uint8_t* apply = FnBase() + off;
    if (std::memcmp(apply, check, 16)) return false;  // another build of the executable: function moved
    fix[i] = {apply, val};
  }
  void* code = nullptr;
  if (code_size) {
    code = std::malloc(code_size);
    if (!code) return false;
    std::memcpy(code, blob.data() + code_pos, code_size);
  }
  void* relocs = nullptr;
  if (reloc_bytes) {
    relocs = std::malloc(reloc_bytes);
    std::memcpy(relocs, blob.data() + reloc_pos, reloc_bytes);
  }
  void* fixups = nullptr;
  if (has_fixups) {
    fixups = std::calloc(1, 8 + size_t(fixup_count) * 16);
    *static_cast<uint32_t*>(fixups) = fixup_count;
    for (uint32_t i = 0; i < fixup_count; ++i) {
      uint8_t* e = static_cast<uint8_t*>(fixups) + 8 + size_t(i) * 16;
      *reinterpret_cast<const void**>(e) = fix[i].first;
      *reinterpret_cast<uint32_t*>(e + 8) = fix[i].second;
    }
  }
  std::memcpy(static_cast<char*>(prog) + kRegionBegin, region, sizeof(region));
  At<void*>(prog, kCode) = code;
  At<void*>(prog, kRelocs) = relocs;
  At<void*>(prog, kFixups) = fixups;
  At<void*>(prog, kTfb) = nullptr;
  At<void*>(prog, kSyms) = nullptr;
  return true;
}
}  // namespace

extern "C" bool __wrap_nvc0_program_translate(void* prog, uint16_t chipset, void* debug) {
  const bool cacheable = At<uint32_t>(prog, kPipeType) == 0 /* TGSI */ && At<const void*>(prog, kPipeTokens) &&
                         At<uint32_t>(prog, kSoNumOutputs) == 0 && At<uint8_t>(prog, kType) != kShaderCompute;
  uint64_t key = 0;
  if (cacheable) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) LoadLocked();
    if (g_enabled) {
      key = Key(prog, chipset);
      auto it = g_entries.find(key);
      if (it != g_entries.end()) {
        if (Restore(prog, it->second.blob)) {
          const uint32_t hits = ++g_hits;
          if (hits <= 4 || hits % 50 == 0) REXGPU_INFO("GL nvc0 cache: {} hits, {} misses", hits, g_misses.load());
          return true;
        }
        if (++g_stale <= 4) REXGPU_WARN("GL nvc0 cache: stale entry {:016X} (executable changed), translating", key);
        g_entries.erase(it);
      }
    }
  }
  const bool ok = __real_nvc0_program_translate(prog, chipset, debug);
  if (!ok || !key) return ok;
  std::vector<uint8_t> blob;
  if (!Serialize(prog, blob)) return ok;
  ++g_misses;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (FILE* f = std::fopen(g_path.c_str(), "ab")) {
    const uint32_t size = uint32_t(blob.size());
    std::fwrite(&key, 8, 1, f);
    std::fwrite(&size, 4, 1, f);
    std::fwrite(blob.data(), 1, blob.size(), f);
    std::fclose(f);
  }
  g_entries[key].blob = std::move(blob);
  return ok;
}
