// Nintendo Switch memory-mapped files: libnx has no mmap, so a read-only "mapping" is the
// file (or the requested slice of it) read into a heap buffer. Intended for small files
// (XEX modules, STFS headers). Large archives (BSA, multi-GB) are not read through here;
// files above kMaxBufferedSize are refused so a stray call fails instead of exhausting RAM.
#include <cstdio>
#include <cstdlib>

#include <rex/logging.h>
#include <rex/memory/mapped_memory.h>
#include <rex/platform.h>

namespace rex::memory {

namespace {

constexpr size_t kMaxBufferedSize = 512ull << 20;

class BufferedMappedMemory : public MappedMemory {
 public:
  BufferedMappedMemory(void* data, size_t size) : MappedMemory(data, size) {}
  ~BufferedMappedMemory() override { std::free(data_); }
};

}  // namespace

std::unique_ptr<MappedMemory> MappedMemory::Open(const std::filesystem::path& path, Mode mode,
                                                 size_t offset, size_t length) {
  FILE* f = std::fopen(path.string().c_str(), mode == Mode::kRead ? "rb" : "r+b");
  if (!f) {
    return nullptr;
  }
  std::fseek(f, 0, SEEK_END);
  long file_size = std::ftell(f);
  if (file_size < 0 || static_cast<size_t>(file_size) < offset) {
    std::fclose(f);
    return nullptr;
  }
  size_t avail = static_cast<size_t>(file_size) - offset;
  size_t size = length ? (length < avail ? length : avail) : avail;
  if (size > kMaxBufferedSize) {
    REXLOG_ERROR("MappedMemory::Open: {} is too large to buffer on Switch ({} bytes)",
                 path.string(), size);
    std::fclose(f);
    return nullptr;
  }
  void* data = std::malloc(size ? size : 1);
  if (!data) {
    std::fclose(f);
    return nullptr;
  }
  std::fseek(f, static_cast<long>(offset), SEEK_SET);
  size_t got = std::fread(data, 1, size, f);
  std::fclose(f);
  if (got != size) {
    std::free(data);
    return nullptr;
  }
  return std::make_unique<BufferedMappedMemory>(data, size);
}

// Writers (profiling / code cache dumps) are not needed on Switch.
std::unique_ptr<ChunkedMappedMemoryWriter> ChunkedMappedMemoryWriter::Open(
    const std::filesystem::path&, size_t, bool) {
  return nullptr;
}

}  // namespace rex::memory
