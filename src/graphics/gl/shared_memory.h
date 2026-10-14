/**
 * @file        graphics/gl/shared_memory.h
 * @brief       Guest physical memory mirror for shaders: 4 x 128 MiB chunks, each backed by a GL buffer
 *              that only covers a window [lo, lo + size) of the chunk (grown on demand, as GL has no sparse
 *              buffers and the Switch application memory budget cannot afford 512 MiB). Shaders read the
 *              buffers through R32UI texture buffers (see spirv_to_glsl.cpp), with the per-chunk window base
 *              passed in the xe_sm_base uniform.
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
  static constexpr uint32_t kGranuleLog2 = 22;  // 4 MiB growth step

  explicit GlSharedMemory(memory::Memory& memory) : SharedMemory(memory) {}
  ~GlSharedMemory() override { Shutdown(true); }

  bool Initialize();
  void Shutdown(bool from_destructor = false);

  // Binds the chunk texture buffers (shader-visible units) for the next draw.
  void BindStorageBuffers(uint32_t first_binding);
  // Sets the xe_sm_base[4] uniform (window base of every chunk in dwords) of the current program.
  void SetBaseUniform(GLint location) const;
  GLuint chunk(uint32_t index) const { return chunks_[index]; }
  // Byte offset (inside the chunk) of the first byte covered by the chunk buffer.
  uint32_t chunk_window_lo(uint32_t index) const { return chunk_lo_[index]; }
  uint32_t total_bytes() const;

 protected:
  bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                        uint32_t length_allocations) override;
  bool UploadRanges(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) override;

 private:
  bool EnsureWindow(uint32_t chunk_index, uint32_t begin, uint32_t end);

  GLuint chunks_[kChunkCount] = {};
  GLuint tbos_[kChunkCount] = {};  // R32UI texture buffer views (shaders read shared memory through these)
  uint32_t chunk_lo_[kChunkCount] = {};
  uint32_t chunk_sizes_[kChunkCount] = {};
  GLuint dummy_ = 0;
};

}  // namespace rex::graphics::gl
