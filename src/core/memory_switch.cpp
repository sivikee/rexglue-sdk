// STUB (Nintendo Switch port, phase 1): host virtual memory primitives.
// This is THE central OS layer: the runtime reserves a 4 GiB guest address space at a
// fixed base and commits/decommits/protects pages inside it, plus file-backed views for
// the physical heaps (memory_posix.cpp does this with mmap/mprotect/shm_open).
// Must eventually be built on libnx virtmem (virtmemReserve) + svcMapPhysicalMemory /
// svcSetMemoryPermission, or a software-translated memory model. Every allocating entry
// point currently fails (returns nullptr/false) after logging.
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>

namespace rex::memory {

size_t page_size() {
  return 0x1000;
}
size_t allocation_granularity() {
  return 0x1000;
}

bool IsWritableExecutableMemorySupported() {
  return false;
}

void* AllocFixed(void*, size_t, AllocationType, PageAccess) {
  REXLOG_ERROR("memory::AllocFixed: not implemented on Switch (stub)");
  return nullptr;
}

bool DeallocFixed(void*, size_t, DeallocationType) {
  REXLOG_ERROR("memory::DeallocFixed: not implemented on Switch (stub)");
  return false;
}

bool Protect(void*, size_t, PageAccess, PageAccess*) {
  REXLOG_ERROR("memory::Protect: not implemented on Switch (stub)");
  return false;
}

bool QueryProtect(void*, size_t&, PageAccess&) {
  REXLOG_ERROR("memory::QueryProtect: not implemented on Switch (stub)");
  return false;
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path&, size_t, PageAccess, bool) {
  REXLOG_ERROR("memory::CreateFileMappingHandle: not implemented on Switch (stub)");
  return kFileMappingHandleInvalid;
}

void CloseFileMappingHandle(FileMappingHandle, const std::filesystem::path&) {}

void* MapFileView(FileMappingHandle, void*, size_t, PageAccess, size_t) {
  REXLOG_ERROR("memory::MapFileView: not implemented on Switch (stub)");
  return nullptr;
}

bool UnmapFileView(FileMappingHandle, void*, size_t) {
  return false;
}

}  // namespace rex::memory
