/**
 * @file        graphics/gl/shared_memory.h
 * @brief       Guest physical memory mirror for shaders: 4 x 128 MiB SSBO windows whose backing buffers
 *              grow on demand up to the highest touched address (GL has no sparse buffers, and the Switch
 *              application memory budget cannot afford 512 MiB).
 */
#pragma once

#include <cstdint>
#include <utility>

#include <glad/glad.h>

#include <rex/graphics/shared_memory.h>

namespace rex::graphics::gl {

class GlSharedMemory final : public SharedMemory {
 public:
  static constexpr uint32_t kChunkSizeLog2 = 27;  // GL_MAX_SHADER_STORAGE_BLOCK_SIZE on Mesa/nouveau
  static constexpr uint32_t kChunkSize = 1u << kChunkSizeLog2;
  static constexpr uint32_t kChunkCount = kBufferSize / kChunkSize;
  static constexpr uint32_t kGranuleLog2 = 23;  // 8 MiB growth step

  explicit GlSharedMemory(memory::Memory& memory) : SharedMemory(memory) {}
  ~GlSharedMemory() override { Shutdown(true); }

  bool Initialize();
  void Shutdown(bool from_destructor = false);

  // Binds the 4 chunks to SSBO binding points first_binding + i (a tiny dummy for unallocated chunks).
  void BindStorageBuffers(uint32_t first_binding);
  GLuint chunk(uint32_t index) const { return chunks_[index]; }
  uint32_t total_bytes() const;

 protected:
  bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                        uint32_t length_allocations) override;
  bool UploadRanges(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) override;

 private:
  bool EnsureChunkSize(uint32_t chunk_index, uint32_t size);

  GLuint chunks_[kChunkCount] = {};
  uint32_t chunk_sizes_[kChunkCount] = {};
  GLuint dummy_ = 0;
};

}  // namespace rex::graphics::gl
