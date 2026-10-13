// STUB (Nintendo Switch port, phase 1): memory-mapped files.
// Must eventually: implement MappedMemory::Open by reading the file into a heap/virtmem
// buffer (or paging via fread), since there is no mmap on libnx. Open() currently fails.
#include <rex/memory/mapped_memory.h>
#include <rex/platform.h>

namespace rex::memory {

std::unique_ptr<MappedMemory> MappedMemory::Open(const std::filesystem::path&, Mode, size_t,
                                                 size_t) {
  return nullptr;
}

std::unique_ptr<ChunkedMappedMemoryWriter> ChunkedMappedMemoryWriter::Open(
    const std::filesystem::path&, size_t, bool) {
  return nullptr;
}

}  // namespace rex::memory
