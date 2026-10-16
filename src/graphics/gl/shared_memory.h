/**
 * @file        graphics/gl/shared_memory.h
 * @brief       Guest physical memory mirror for shaders. GL has no sparse buffers and the nouveau/Eden
 *              driver cannot allocate large buffers once memory is fragmented (a 52 MiB buffer fails), so
 *              the mirror is a COMPACT store: guest memory is made resident in 1 MiB granules that are
 *              packed into one R32UI texture buffer ("store") on demand; a second small R32UI texture
 *              buffer (the "page table", one dword per guest MiB) gives the store offset of every
 *              granule. Shaders read shared memory as store[pt[a >> 18] + (a & 0x3FFFF)] (see
 *              spirv_to_glsl.cpp).
 */
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/shared_memory.h>

#include "stream_ring.h"

namespace rex::graphics::gl {

class GlSharedMemory final : public SharedMemory {
 public:
  static constexpr uint32_t kGranuleLog2 = 20;  // residency granule: 1 MiB
  static constexpr uint32_t kGranuleSize = 1u << kGranuleLog2;
  static constexpr uint32_t kGranuleCount = kBufferSize >> kGranuleLog2;
  static constexpr uint32_t kGrowStep = 8u << 20;  // store capacity growth step

  explicit GlSharedMemory(memory::Memory& memory) : SharedMemory(memory) {}
  ~GlSharedMemory() override { Shutdown(true); }

  bool Initialize();
  void Shutdown(bool from_destructor = false);

  // Binds the store and page table texture buffers (shader-visible units) for the next draw.
  void BindStorageBuffers(uint32_t first_binding);
  // No per-draw uniform any more (kept for the call sites).
  void SetBaseUniform(GLint location) const { (void)location; }
  // Host buffer + byte offset holding guest physical range [address, address + length) contiguously.
  // False if the range is not resident or not contiguous in the store.
  bool GetHostRange(uint32_t address, uint32_t length, GLuint& buffer, GLintptr& offset) const;
  uint32_t total_bytes() const { return next_slot_ << kGranuleLog2; }
  // GPU-side access (compute resolves, front buffer blit, lazy downloads).
  GLuint store_buffer() const { return store_; }
  GLuint page_table_buffer() const { return pt_; }
  // R32UI texture buffers over the store / page table (draws read them with texelFetch; the GPU resolve compute
  // shader writes the store through an image buffer).
  GLuint store_texture_buffer() const { return store_tbo_; }
  GLuint page_table_texture_buffer() const { return pt_tbo_; }
  uint32_t store_capacity() const { return capacity_; }
  // Store byte offset of a guest physical address and the number of bytes from it to the end of its granule.
  bool StoreOffset(uint32_t address, GLintptr& offset, uint32_t& bytes_to_granule_end) const;
  // Called before the store contents are discarded (allocation failure fallback): GPU-only data must be written back.
  void SetPreStoreResetHook(std::function<void()> hook) { pre_store_reset_hook_ = std::move(hook); }
  // Called with a guest physical range before it is read from guest memory (flushes pending async resolves).
  void SetPreReadHook(std::function<void(uint32_t, uint32_t)> hook) { pre_read_hook_ = std::move(hook); }

 protected:
  bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                        uint32_t length_allocations) override;
  bool UploadRanges(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) override;

 private:
  bool EnsureCapacity(uint32_t slots_needed_total);

  StreamRing staging_;
  std::function<void(uint32_t, uint32_t)> pre_read_hook_;
  std::function<void()> pre_store_reset_hook_;
  GLuint store_ = 0;
  GLuint store_tbo_ = 0;
  GLuint pt_ = 0;
  GLuint pt_tbo_ = 0;
  uint32_t capacity_ = 0;   // store size in bytes
  uint32_t next_slot_ = 0;  // granule slots handed out so far
  uint32_t page_table_[kGranuleCount] = {};  // dword offset into the store, or 0xFFFFFFFF
};

}  // namespace rex::graphics::gl
