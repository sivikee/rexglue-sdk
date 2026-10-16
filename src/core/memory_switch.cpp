// Nintendo Switch (libnx / Horizon) host memory primitives.
//
// Why this is NOT the POSIX design: the POSIX port maps one backing file into 9 views of a 4.5 GiB
// arena (true RW aliases). On Horizon, aliases need svcCreateSharedMemory (privileged; returns
// NotImplemented 0x4201 in Eden, see ports/switch/README.md), svcMapProcessMemory (needs a real
// own-process handle, only available when launched through hbloader) or svcMapPhysicalMemory
// (needs a system-resource size from the NPDM, NROs have none). The one primitive that is always
// available and places memory at a chosen address is svcMapMemory(dst, src, size): it MOVES
// pages of the process heap into the 2 GiB "stack region" (dst must be inside it).
//
// Design ("linear windows"):
//  * guest address g lives at base + g (identical to the POSIX arena, so generated code and the
//    runtime are unchanged), with base = StackRegionAddress + 2 MiB - 0x40000000.
//  * Only these guest ranges exist (reserved in libnx virtmem so libnx thread stacks keep the rest
//    of the stack region):
//        0x40000000..0x83FFFFFF  virtual 64 KiB heap (v40000000, incl. title thread stacks at
//                                0x70000000+), the 0x7F... MMIO window and the XEX heap (v80000000, 64 MiB)
//        0xA0000000..0xBFFFFFFF  physical memory; the "raw physical" view (TranslatePhysical) is the
//                                SAME range (physical_membase = base + 0xA0000000), so there is no aliasing
//    xmemory.cpp redirects 4 KiB / C0 / E0 / v00000000 allocations onto these heaps on Switch.
//  * Commit = take a contiguous run from a pool inside the process heap (reserved by the app, see
//    SwitchSetGuestPool) and svcMapMemory it onto the guest address. Unit: 64 KiB.
//    Decommit unmaps whole units back to the pool; partial units are zeroed.
//  * Per-page protection does not exist here: Protect() is a recorded no-op (so no write-watch / no
//    fault-driven MMIO; the generated code uses explicit MMIO macros).
//  * A fault inside the windows demand-commits the unit and retries (HandleGuestMemoryFault).
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include <switch.h>

#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>

extern "C" void oblivion_bootlog(const char* fmt, ...) __attribute__((weak, format(printf, 1, 2)));
#define BOOTLOG(...) do { if (oblivion_bootlog) oblivion_bootlog(__VA_ARGS__); } while (0)

namespace rex::memory {

namespace {

constexpr size_t kUnit = 0x10000;  // commit granularity (64 KiB)
// The first ~1 MiB of the stack region is already taken (libnx maps the main thread stack there),
// so the arena is shifted up by 2 MiB and the top window is 2 MiB shorter.
constexpr uint64_t kRegionSkip = 0x200000;

struct Window {
  uint32_t guest_lo;
  uint32_t size;
  size_t first_unit;  // index of the window's first unit in the global unit table
};

Window g_windows[] = {
    {0x40000000u, 0x44000000u, 0},  // v40000000 heap (incl. title thread stacks 0x70000000+) + 7F MMIO + XEX
    {0xA0000000u, 0x1FE00000u, 0},  // physical (2 MiB shorter: see kRegionSkip)
};
constexpr int kWindowCount = 2;

struct State {
  std::mutex mutex;
  bool arena_ready = false;
  uint8_t* base = nullptr;
  uint8_t* pool = nullptr;
  size_t pool_units = 0;
  std::vector<bool> pool_used;   // per pool unit
  std::vector<int32_t> pool_of;  // per guest unit: pool unit index or -1
  size_t total_units = 0;
  int protect_logged = 0;
};

State& S() {
  static State s;
  return s;
}

uint8_t* g_pool_base = nullptr;
size_t g_pool_size = 0;
std::map<uintptr_t, size_t>* g_anon = nullptr;

// Returns window index or -1, and the unit index (global) for a host address.
int FindWindow(State& s, const void* addr, size_t* unit) {
  uintptr_t off = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(s.base);
  if (reinterpret_cast<uintptr_t>(addr) < reinterpret_cast<uintptr_t>(s.base) || off > 0xFFFFFFFFull) {
    return -1;
  }
  for (int i = 0; i < kWindowCount; ++i) {
    const Window& w = g_windows[i];
    if (off >= w.guest_lo && off < uint64_t(w.guest_lo) + w.size) {
      if (unit) {
        *unit = w.first_unit + (off - w.guest_lo) / kUnit;
      }
      return i;
    }
  }
  return -1;
}

uint8_t* UnitAddr(State& s, int win, size_t unit) {
  const Window& w = g_windows[win];
  return s.base + w.guest_lo + (unit - w.first_unit) * kUnit;
}

// First-fit search for `count` contiguous free pool units.
int64_t PoolAlloc(State& s, size_t count) {
  size_t run = 0;
  for (size_t i = 0; i < s.pool_units; ++i) {
    if (s.pool_used[i]) {
      run = 0;
      continue;
    }
    if (++run == count) {
      size_t start = i + 1 - count;
      for (size_t k = start; k <= i; ++k) {
        s.pool_used[k] = true;
      }
      return static_cast<int64_t>(start);
    }
  }
  return -1;
}

// Diagnostic: log every mapped block overlapping the guest windows.
void LogWindowBlocks(State& s, const char* label) {
  u32 pi = 0;
  for (int i = 0; i < kWindowCount; ++i) {
    u64 a = reinterpret_cast<u64>(s.base) + g_windows[i].guest_lo, end = a + g_windows[i].size;
    while (a < end) {
      MemoryInfo m{};
      if (R_FAILED(svcQueryMemory(&m, &pi, a))) break;
      if (m.type != 0) {
        REXLOG_WARN("{}: window {} block {:X}+{:X} type={} perm={} attr={}", label, i, m.addr, m.size, m.type,
                    m.perm, m.attr);
      }
      a = m.addr + m.size;
    }
  }
}

// Commit units [first, first+count) of window `win` (all currently uncommitted and the
// destination is contiguous in guest space; the pool run is contiguous too).
bool CommitRun(State& s, int win, size_t first, size_t count) {
  int64_t p = PoolAlloc(s, count);
  if (p < 0) {
    REXLOG_ERROR("Switch guest pool exhausted (need {} KiB, pool {} MiB)", count * kUnit / 1024,
                 s.pool_units * kUnit >> 20);
    return false;
  }
  uint8_t* src = s.pool + static_cast<size_t>(p) * kUnit;
  uint8_t* dst = UnitAddr(s, win, first);
  std::memset(src, 0, count * kUnit);
  REXLOG_DEBUG("svcMapMemory dst={} src={} len=0x{:X}", static_cast<void*>(dst), static_cast<void*>(src), count * kUnit);
  Result rc = svcMapMemory(dst, src, count * kUnit);
  if (R_FAILED(rc)) {
    REXLOG_ERROR("svcMapMemory(dst={}, src={}, 0x{:X}) failed: 0x{:X}", static_cast<void*>(dst),
                 static_cast<void*>(src), count * kUnit, rc);
    MemoryInfo mi_dst{}, mi_src{};
    u32 pi = 0;
    svcQueryMemory(&mi_dst, &pi, reinterpret_cast<u64>(dst));
    svcQueryMemory(&mi_src, &pi, reinterpret_cast<u64>(src));
    REXLOG_ERROR("  dst: base={:X} size={:X} type={} perm={} attr={}; src: base={:X} size={:X} type={} perm={} attr={}",
                 mi_dst.addr, mi_dst.size, mi_dst.type, mi_dst.perm, mi_dst.attr, mi_src.addr,
                 mi_src.size, mi_src.type, mi_src.perm, mi_src.attr);
    {
      u64 a = reinterpret_cast<u64>(dst), end = a + count * kUnit;
      for (int n = 0; n < 8 && a < end; ++n) {
        MemoryInfo m{};
        svcQueryMemory(&m, &pi, a);
        REXLOG_ERROR("  dst walk: block {:X}+{:X} type={} perm={} attr={}", m.addr, m.size, m.type, m.perm, m.attr);
        a = m.addr + m.size;
      }
    }
    LogWindowBlocks(s, "commit failure");
    for (size_t k = 0; k < count; ++k) {
      s.pool_used[static_cast<size_t>(p) + k] = false;
    }
    return false;
  }
  for (size_t k = 0; k < count; ++k) {
    s.pool_of[first + k] = static_cast<int32_t>(static_cast<size_t>(p) + k);
  }
  return true;
}

// Commit every uncommitted unit overlapping [addr, addr+length).
bool CommitLocked(State& s, const void* addr, size_t length) {
  size_t first_unit = 0;
  int win = FindWindow(s, addr, &first_unit);
  if (win < 0 || !s.arena_ready) {
    return false;
  }
  const Window& w = g_windows[win];
  uintptr_t start = reinterpret_cast<uintptr_t>(addr);
  uintptr_t win_end = reinterpret_cast<uintptr_t>(s.base) + w.guest_lo + w.size;
  uintptr_t end = start + length;
  if (end > win_end) {
    end = win_end;
  }
  size_t last_unit = first_unit;
  FindWindow(s, reinterpret_cast<void*>(end - 1), &last_unit);
  size_t u = first_unit;
  while (u <= last_unit) {
    if (s.pool_of[u] >= 0) {
      ++u;
      continue;
    }
    size_t run = 1;
    while (u + run <= last_unit && s.pool_of[u + run] < 0) {
      ++run;
    }
    if (!CommitRun(s, win, u, run)) {
      return false;
    }
    u += run;
  }
  return true;
}

// Release committed units that are FULLY inside [addr, addr+length); zero partially covered ones.
void DecommitLocked(State& s, void* addr, size_t length) {
  size_t first_unit = 0;
  int win = FindWindow(s, addr, &first_unit);
  if (win < 0) {
    return;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(addr);
  uintptr_t end = start + length;
  uintptr_t unit0 = reinterpret_cast<uintptr_t>(UnitAddr(s, win, first_unit));
  uintptr_t win_end = reinterpret_cast<uintptr_t>(s.base) + g_windows[win].guest_lo + g_windows[win].size;
  if (end > win_end) {
    end = win_end;
  }
  size_t u = first_unit;
  for (uintptr_t ua = unit0; ua < end; ua += kUnit, ++u) {
    uintptr_t ue = ua + kUnit;
    uintptr_t zs = ua > start ? ua : start;
    uintptr_t ze = ue < end ? ue : end;
    if (s.pool_of[u] < 0) {
      continue;
    }
    if (zs == ua && ze == ue) {
      uint8_t* src = s.pool + static_cast<size_t>(s.pool_of[u]) * kUnit;
      svcUnmapMemory(reinterpret_cast<void*>(ua), src, kUnit);
      s.pool_used[static_cast<size_t>(s.pool_of[u])] = false;
      s.pool_of[u] = -1;
    } else {
      std::memset(reinterpret_cast<void*>(zs), 0, ze - zs);
    }
  }
}

}  // namespace

size_t page_size() {
  return 0x1000;
}
size_t allocation_granularity() {
  return 0x1000;
}

bool IsWritableExecutableMemorySupported() {
  return false;
}

void SwitchSetGuestPool(void* pool, size_t size) {
  g_pool_base = static_cast<uint8_t*>(pool);
  g_pool_size = size;
}

void* SwitchInitGuestArena() {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.arena_ready) {
    return s.base;
  }
  if (!g_pool_base || g_pool_size < kUnit) {
    REXLOG_ERROR("Switch guest pool not set (app must call SwitchSetGuestPool from __libnx_initheap)");
    return nullptr;
  }
  u64 stack_addr = 0, stack_size = 0;
  BOOTLOG("arena: begin, pool=%p size=0x%llx", (void*)g_pool_base, (unsigned long long)g_pool_size);
  svcGetInfo(&stack_addr, InfoType_StackRegionAddress, CUR_PROCESS_HANDLE, 0);
  svcGetInfo(&stack_size, InfoType_StackRegionSize, CUR_PROCESS_HANDLE, 0);
  BOOTLOG("arena: stack region 0x%llx size 0x%llx", (unsigned long long)stack_addr, (unsigned long long)stack_size);
  if (stack_size < 0x80000000ull) {
    REXLOG_ERROR("Stack region too small for the guest windows (0x{:X})", stack_size);
    return nullptr;
  }
  s.base = reinterpret_cast<uint8_t*>(stack_addr + kRegionSkip - 0x40000000ull);
  s.pool = g_pool_base;
  s.pool_units = g_pool_size / kUnit;
  s.pool_used.assign(s.pool_units, false);
  size_t units = 0;
  virtmemLock();
  for (int i = 0; i < kWindowCount; ++i) {
    g_windows[i].first_unit = units;
    units += g_windows[i].size / kUnit;
    {
      u64 qa = reinterpret_cast<u64>(s.base) + g_windows[i].guest_lo, qend = qa + g_windows[i].size;
      u32 qpi = 0;
      int nblocks = 0;
      while (qa < qend && nblocks < 16) {
        MemoryInfo qm{};
        if (R_FAILED(svcQueryMemory(&qm, &qpi, qa))) break;
        BOOTLOG("arena: win%d block addr=0x%llx size=0x%llx type=%u perm=%u", i, (unsigned long long)qm.addr,
                (unsigned long long)qm.size, qm.type, qm.perm);
        ++nblocks;
        if (!qm.size) break;
        qa = qm.addr + qm.size;
      }
    }
    BOOTLOG("arena: reserving win%d %p + 0x%x", i, (void*)(s.base + g_windows[i].guest_lo), g_windows[i].size);
    // Keep libnx from handing these ranges out as thread-stack mirrors.
    void* resv = virtmemAddReservation(s.base + g_windows[i].guest_lo, g_windows[i].size);
    BOOTLOG("arena: reservation win%d -> %p", i, resv);
  }
  virtmemUnlock();
  s.total_units = units;
  s.pool_of.assign(units, -1);
  BOOTLOG("arena: reservations done");
  LogWindowBlocks(s, "guest arena init");
  s.arena_ready = true;
  BOOTLOG("arena: ready, base=%p", (void*)s.base);
  REXLOG_INFO("Switch guest arena: base {} (stack region {:X}+{:X}), pool {} MiB at {}",
              static_cast<void*>(s.base), stack_addr, stack_size, g_pool_size >> 20,
              static_cast<void*>(s.pool));
  return s.base;
}

bool HandleGuestMemoryFault(uintptr_t address) {
  State& s = S();
  if (!s.arena_ready || !s.mutex.try_lock()) {
    return false;
  }
  bool ok = false;
  if (FindWindow(s, reinterpret_cast<void*>(address), nullptr) >= 0) {
    ok = CommitLocked(s, reinterpret_cast<void*>(address), 1);
  }
  s.mutex.unlock();
  return ok;
}

void* AllocFixed(void* base_address, size_t length, AllocationType allocation_type,
                 PageAccess access) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!base_address) {
    void* p = std::aligned_alloc(0x1000, (length + 0xFFF) & ~size_t(0xFFF));
    if (p) {
      std::memset(p, 0, length);
      if (!g_anon) {
        g_anon = new std::map<uintptr_t, size_t>();
      }
      (*g_anon)[reinterpret_cast<uintptr_t>(p)] = length;
    }
    return p;
  }
  if (!s.arena_ready || FindWindow(s, base_address, nullptr) < 0) {
    REXLOG_ERROR("memory::AllocFixed: {} (0x{:X}) is outside the Switch guest windows", base_address,
                 length);
    return nullptr;
  }
  if (allocation_type == AllocationType::kReserve) {
    return base_address;
  }
  (void)access;
  return CommitLocked(s, base_address, length) ? base_address : nullptr;
}

bool DeallocFixed(void* base_address, size_t length, DeallocationType deallocation_type) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (g_anon) {
    auto an = g_anon->find(reinterpret_cast<uintptr_t>(base_address));
    if (an != g_anon->end()) {
      std::free(base_address);
      g_anon->erase(an);
      return true;
    }
  }
  if (!s.arena_ready || FindWindow(s, base_address, nullptr) < 0) {
    return false;
  }
  if (length == 0 || deallocation_type == DeallocationType::kRelease) {
    return true;  // regions are released through kDecommit; the windows live until shutdown
  }
  DecommitLocked(s, base_address, length);
  return true;
}

bool Protect(void* base_address, size_t length, PageAccess access, PageAccess* out_old_access) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (out_old_access) {
    *out_old_access = PageAccess::kReadWrite;
  }
  if (access != PageAccess::kReadWrite && s.protect_logged < 8) {
    ++s.protect_logged;
    REXLOG_WARN("memory::Protect({}, 0x{:X}, {}) ignored on Switch (no per-page protection)",
                base_address, length, static_cast<int>(access));
  }
  return true;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  size_t unit = 0;
  int win = s.arena_ready ? FindWindow(s, base_address, &unit) : -1;
  if (win < 0) {
    access_out = PageAccess::kNoAccess;
    length = 0;
    return false;
  }
  bool committed = s.pool_of[unit] >= 0;
  uintptr_t a = reinterpret_cast<uintptr_t>(base_address);
  uintptr_t unit_end = reinterpret_cast<uintptr_t>(UnitAddr(s, win, unit)) + kUnit;
  access_out = committed ? PageAccess::kReadWrite : PageAccess::kNoAccess;
  length = unit_end - a;
  return true;
}

// The POSIX file-mapping API has no meaning here (single linear arena, see header comment).
FileMappingHandle CreateFileMappingHandle(const std::filesystem::path&, size_t, PageAccess, bool) {
  return 1;
}

void CloseFileMappingHandle(FileMappingHandle, const std::filesystem::path&) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.arena_ready) {
    return;
  }
  for (int i = 0; i < kWindowCount; ++i) {
    const Window& w = g_windows[i];
    DecommitLocked(s, s.base + w.guest_lo, w.size);
  }
}

void* MapFileView(FileMappingHandle, void*, size_t, PageAccess, size_t) {
  return nullptr;
}

bool UnmapFileView(FileMappingHandle, void*, size_t) {
  return false;
}

}  // namespace rex::memory
