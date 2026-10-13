// Nintendo Switch (libnx / Horizon) host memory primitives.
//
// Design (see ports/switch/README.md, "Memory design"):
//  * The 4.5 GiB guest arena is a VA reservation taken from the ASLR region (virtmem).
//  * Horizon has no demand-paged anonymous memory and no sparse files, so the single
//    "backing file" of the POSIX port is emulated by a table of shared-memory SEGMENTS
//    (svcCreateSharedMemory), keyed by "file offset". A segment is committed lazily (by
//    AllocFixed(commit) or, as a safety net, by a fault handler) and is mapped with
//    svcMapSharedMemory into EVERY view that covers its file range. That reproduces the
//    aliased RW views (0x7F/0xA0/0xC0/0xE0 + physical, 0x80/0x90) of the POSIX design.
//    svcMapSharedMemory maps a whole object only, hence a segment must fit entirely inside
//    each view that maps it: segments are 16 MiB inside the physical block (smallest
//    aliasing view is the 16 MiB 0x7F view) and 1 MiB elsewhere.
//  * allocation_granularity() is 64 KiB (like Windows), so the 0xE0000000 view's 4 KiB
//    host offset is applied by REX_PHYS_HOST_OFFSET / PhysicalHostOffset on Switch too.
//  * Per-page protection is NOT available on shared memory in general (needs hardware
//    verification, see memprobe), so Protect() only records success. Write-watch /
//    MMIO-by-fault therefore do not work yet (GPU only; MMIO uses explicit macros).
//  * Decommit frees a segment only when the range covers it entirely, otherwise the range
//    is zeroed (fresh commits must read as zero).
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include <switch.h>

#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>

namespace rex::memory {

namespace {

constexpr uint64_t kPhysStart = 0x100000000ull;
constexpr uint64_t kPhysEnd = 0x120000000ull;
constexpr size_t kPhysSegment = 16ull << 20;
constexpr size_t kVirtSegment = 1ull << 20;

struct Segment {
  Handle handle = INVALID_HANDLE;
  size_t length = 0;
};

struct View {
  uint8_t* base = nullptr;
  size_t length = 0;
  uint64_t file_offset = 0;
};

struct Backing {
  size_t length = 0;
  std::map<uint64_t, Segment> segments;  // by file offset
  std::vector<View> views;
};

struct State {
  std::mutex mutex;
  std::map<uintptr_t, size_t> anon;  // plain heap allocations made through AllocFixed(nullptr)
  Backing* backing = nullptr;        // one guest arena per process
  int protect_logged = 0;
};

State& S() {
  static State s;
  return s;
}

inline size_t SegmentSize(uint64_t file_offset) {
  return (file_offset >= kPhysStart && file_offset < kPhysEnd) ? kPhysSegment : kVirtSegment;
}

const View* FindView(Backing* b, const void* addr) {
  auto p = reinterpret_cast<const uint8_t*>(addr);
  for (auto& v : b->views) {
    if (p >= v.base && p < v.base + v.length) {
      return &v;
    }
  }
  return nullptr;
}

bool MapSegmentInto(const View& v, uint64_t seg_off, const Segment& seg) {
  if (seg_off < v.file_offset || seg_off + seg.length > v.file_offset + v.length) {
    return false;  // segment does not fit entirely in this view (e.g. 0x7F view)
  }
  uint8_t* at = v.base + (seg_off - v.file_offset);
  Result rc = svcMapSharedMemory(seg.handle, at, seg.length, Perm_Rw);
  if (R_FAILED(rc)) {
    REXLOG_ERROR("svcMapSharedMemory({}, 0x{:X}) failed: 0x{:X}", static_cast<void*>(at),
                 seg.length, rc);
    return false;
  }
  return true;
}

void UnmapSegmentFrom(const View& v, uint64_t seg_off, const Segment& seg) {
  if (seg_off < v.file_offset || seg_off + seg.length > v.file_offset + v.length) {
    return;
  }
  svcUnmapSharedMemory(seg.handle, v.base + (seg_off - v.file_offset), seg.length);
}

bool CreateSegment(Backing* b, uint64_t seg_off) {
  size_t len = SegmentSize(seg_off);
  Segment seg;
  seg.length = len;
  SharedMemory sh;
  Result rc = shmemCreate(&sh, len, Perm_Rw, Perm_Rw);
  if (R_FAILED(rc)) {
    REXLOG_ERROR("shmemCreate(0x{:X}) for file offset 0x{:X} failed: 0x{:X}", len, seg_off, rc);
    return false;
  }
  seg.handle = sh.handle;
  b->segments.emplace(seg_off, seg);
  for (auto& v : b->views) {
    MapSegmentInto(v, seg_off, seg);
  }
  return true;
}

void DestroySegment(Backing* b, std::map<uint64_t, Segment>::iterator it) {
  for (auto& v : b->views) {
    UnmapSegmentFrom(v, it->first, it->second);
  }
  svcCloseHandle(it->second.handle);
  b->segments.erase(it);
}

// Commit every segment overlapping [addr, addr+length) of the view containing addr.
bool CommitLocked(Backing* b, const void* addr, size_t length) {
  const View* v = FindView(b, addr);
  if (!v) {
    return false;
  }
  uint64_t off = v->file_offset + (static_cast<const uint8_t*>(addr) - v->base);
  uint64_t end = off + length;
  uint64_t view_end = v->file_offset + v->length;
  if (end > view_end) {
    end = view_end;
  }
  while (off < end) {
    size_t seg_size = SegmentSize(off);
    uint64_t seg_off = off & ~(uint64_t(seg_size) - 1);
    if (!b->segments.count(seg_off)) {
      if (!CreateSegment(b, seg_off)) {
        return false;
      }
    }
    off = seg_off + seg_size;
  }
  return true;
}

}  // namespace

size_t page_size() {
  return 0x1000;
}
size_t allocation_granularity() {
  return 0x10000;
}

bool IsWritableExecutableMemorySupported() {
  return false;
}

void* ReserveGuestAddressSpace(size_t size) {
  virtmemLock();
  void* p = virtmemFindAslr(size, 0x10000);
  if (p) {
    virtmemAddReservation(p, size);
  }
  virtmemUnlock();
  if (!p) {
    REXLOG_ERROR("virtmemFindAslr(0x{:X}) failed", size);
  }
  return p;
}

bool HandleGuestMemoryFault(uintptr_t address) {
  State& s = S();
  if (!s.backing || !s.mutex.try_lock()) {
    return false;
  }
  bool ok = false;
  if (FindView(s.backing, reinterpret_cast<void*>(address))) {
    ok = CommitLocked(s.backing, reinterpret_cast<void*>(address), 1);
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
      s.anon[reinterpret_cast<uintptr_t>(p)] = length;
    }
    return p;
  }
  if (!s.backing || !FindView(s.backing, base_address)) {
    REXLOG_ERROR("memory::AllocFixed: {} is outside the guest arena", base_address);
    return nullptr;
  }
  if (allocation_type == AllocationType::kReserve) {
    return base_address;
  }
  (void)access;
  return CommitLocked(s.backing, base_address, length) ? base_address : nullptr;
}

bool DeallocFixed(void* base_address, size_t length, DeallocationType deallocation_type) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto an = s.anon.find(reinterpret_cast<uintptr_t>(base_address));
  if (an != s.anon.end()) {
    std::free(base_address);
    s.anon.erase(an);
    return true;
  }
  Backing* b = s.backing;
  if (!b || !FindView(b, base_address)) {
    return false;
  }
  if (length == 0 || deallocation_type == DeallocationType::kRelease) {
    // The guest heaps release regions they reserved earlier; committed pages are returned
    // through kDecommit, the arena itself lives until CloseFileMappingHandle.
    return true;
  }
  const View* v = FindView(b, base_address);
  uint64_t off = v->file_offset + (static_cast<uint8_t*>(base_address) - v->base);
  uint64_t end = off + length;
  for (auto it = b->segments.begin(); it != b->segments.end();) {
    uint64_t so = it->first;
    uint64_t se = so + it->second.length;
    if (se <= off || so >= end) {
      ++it;
      continue;
    }
    if (off <= so && end >= se) {
      auto victim = it++;
      DestroySegment(b, victim);
    } else {
      uint64_t zs = off > so ? off : so;
      uint64_t ze = end < se ? end : se;
      std::memset(v->base + (zs - v->file_offset), 0, ze - zs);
      ++it;
    }
  }
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
  Backing* b = s.backing;
  const View* v = b ? FindView(b, base_address) : nullptr;
  if (!v) {
    access_out = PageAccess::kNoAccess;
    length = 0;
    return false;
  }
  uint64_t off = v->file_offset + (static_cast<uint8_t*>(base_address) - v->base);
  uint64_t view_end = v->file_offset + v->length;
  auto it = b->segments.upper_bound(off);
  if (it != b->segments.begin()) {
    auto prev = std::prev(it);
    if (off < prev->first + prev->second.length) {
      access_out = PageAccess::kReadWrite;
      length = prev->first + prev->second.length - off;
      return true;
    }
  }
  access_out = PageAccess::kNoAccess;
  length = (it != b->segments.end() && it->first < view_end ? it->first : view_end) - off;
  return true;
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path&, size_t length, PageAccess,
                                          bool) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.backing) {
    REXLOG_ERROR("memory::CreateFileMappingHandle: only one guest arena is supported on Switch");
    return kFileMappingHandleInvalid;
  }
  s.backing = new Backing();
  s.backing->length = length;
  return reinterpret_cast<FileMappingHandle>(s.backing);
}

void CloseFileMappingHandle(FileMappingHandle handle, const std::filesystem::path&) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* b = reinterpret_cast<Backing*>(handle);
  if (!b || b != s.backing) {
    return;
  }
  while (!b->segments.empty()) {
    DestroySegment(b, b->segments.begin());
  }
  delete b;
  s.backing = nullptr;
}

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length, PageAccess,
                  size_t file_offset) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* b = reinterpret_cast<Backing*>(handle);
  if (!b || b != s.backing || !base_address) {
    // The guest arena must be placed by the caller inside ReserveGuestAddressSpace().
    return nullptr;
  }
  View v;
  v.base = static_cast<uint8_t*>(base_address);
  v.length = length;
  v.file_offset = file_offset;
  b->views.push_back(v);
  for (auto& seg : b->segments) {
    MapSegmentInto(v, seg.first, seg.second);
  }
  return base_address;
}

bool UnmapFileView(FileMappingHandle handle, void* base_address, size_t) {
  State& s = S();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* b = reinterpret_cast<Backing*>(handle);
  if (!b || b != s.backing) {
    return false;
  }
  for (auto it = b->views.begin(); it != b->views.end(); ++it) {
    if (it->base == base_address) {
      for (auto& seg : b->segments) {
        UnmapSegmentFrom(*it, seg.first, seg.second);
      }
      b->views.erase(it);
      return true;
    }
  }
  return false;
}

}  // namespace rex::memory
