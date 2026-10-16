#include "shared_memory.h"

#include <algorithm>
#include <cstring>

#include <rex/logging.h>

#include "gl_ext.h"
#include "perf.h"
#include "spirv_to_glsl.h"

namespace rex::graphics::gl {

bool GlSharedMemory::Initialize() {
  InitializeCommon();
  InitializeSparseHostGpuMemory(kGranuleLog2);
  std::fill(std::begin(page_table_), std::end(page_table_), 0xFFFFFFFFu);
  glGenBuffers(1, &pt_);
  glBindBuffer(GL_COPY_WRITE_BUFFER, pt_);
  glBufferData(GL_COPY_WRITE_BUFFER, sizeof(page_table_), page_table_, GL_DYNAMIC_DRAW);
  glGenTextures(1, &pt_tbo_);
  glBindTexture(GL_TEXTURE_BUFFER, pt_tbo_);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, pt_);
  // The store starts as a tiny dummy so the texture buffer is always complete.
  glGenBuffers(1, &store_);
  glBindBuffer(GL_COPY_WRITE_BUFFER, store_);
  uint8_t zeros[256] = {};
  glBufferData(GL_COPY_WRITE_BUFFER, sizeof(zeros), zeros, GL_DYNAMIC_DRAW);
  glGenTextures(1, &store_tbo_);
  glBindTexture(GL_TEXTURE_BUFFER, store_tbo_);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, store_);
  capacity_ = 0;
  next_slot_ = 0;
  return true;
}

void GlSharedMemory::Shutdown(bool from_destructor) {
  if (store_) glDeleteBuffers(1, &store_);
  if (pt_) glDeleteBuffers(1, &pt_);
  if (store_tbo_) glDeleteTextures(1, &store_tbo_);
  if (pt_tbo_) glDeleteTextures(1, &pt_tbo_);
  store_ = pt_ = store_tbo_ = pt_tbo_ = 0;
  capacity_ = next_slot_ = 0;
  if (!from_destructor) {
    ShutdownCommon();
  }
}

bool GlSharedMemory::EnsureCapacity(uint32_t slots_needed_total) {
  uint64_t need = uint64_t(slots_needed_total) << kGranuleLog2;
  if (need <= capacity_) {
    return true;
  }
  uint32_t new_cap = uint32_t((need + kGrowStep - 1) / kGrowStep * kGrowStep);
  auto alloc = [&](GLuint& out) -> bool {
    glGenBuffers(1, &out);
    glBindBuffer(GL_COPY_WRITE_BUFFER, out);
    glBufferData(GL_COPY_WRITE_BUFFER, new_cap, nullptr, GL_DYNAMIC_DRAW);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
      REXGPU_WARN("GL shared memory: allocating the {} MiB store failed (resident {} MiB): error {:x}", new_cap >> 20,
                  total_bytes() >> 20, err);
      glDeleteBuffers(1, &out);
      out = 0;
      return false;
    }
    return true;
  };
  GLuint nb = 0;
  bool copied = true;
  if (!alloc(nb)) {
    // The old store and the grown one cannot coexist: drop the old one, every page is re-uploaded on demand.
    if (!capacity_) {
      return false;
    }
    glBindTexture(GL_TEXTURE_BUFFER, store_tbo_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, pt_);  // detach
    glDeleteBuffers(1, &store_);
    store_ = 0;
    {
      perf::Scope perf_finish_scope(perf::kFinishTicks);
      glFinish();
    }
    InvalidateAllPages();
    copied = false;
    if (!alloc(nb)) {
      capacity_ = 0;
      return false;
    }
  }
  if (store_ && copied && capacity_) {
    glBindBuffer(GL_COPY_READ_BUFFER, store_);
    glBindBuffer(GL_COPY_WRITE_BUFFER, nb);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, capacity_);
    glDeleteBuffers(1, &store_);
  }
  store_ = nb;
  capacity_ = new_cap;
  glBindTexture(GL_TEXTURE_BUFFER, store_tbo_);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, store_);
  REXGPU_INFO("GL shared memory: store {} MiB (resident granules {} MiB)", capacity_ >> 20, total_bytes() >> 20);
  return true;
}

bool GlSharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                      uint32_t length_allocations) {
  // Granules of this run that still need a slot get consecutive slots, so that ranges requested
  // together (index buffers used directly by GL) stay contiguous in the store.
  uint32_t missing = 0;
  for (uint32_t g = offset_allocations; g < offset_allocations + length_allocations && g < kGranuleCount; ++g) {
    if (page_table_[g] == 0xFFFFFFFFu) ++missing;
  }
  if (!missing) {
    return true;
  }
  if (!EnsureCapacity(next_slot_ + missing)) {
    return false;
  }
  uint32_t first = ~0u, last = 0;
  for (uint32_t g = offset_allocations; g < offset_allocations + length_allocations && g < kGranuleCount; ++g) {
    if (page_table_[g] != 0xFFFFFFFFu) continue;
    page_table_[g] = (next_slot_++ << kGranuleLog2) >> 2;  // dwords
    first = std::min(first, g);
    last = std::max(last, g);
  }
  glBindBuffer(GL_COPY_WRITE_BUFFER, pt_);
  glBufferSubData(GL_COPY_WRITE_BUFFER, first * 4, (last - first + 1) * 4, &page_table_[first]);
  return true;
}

bool GlSharedMemory::UploadRanges(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) {
  perf::Scope perf_sm_scope(perf::kSmUploadTicks);
  for (const auto& range : upload_page_ranges) {
    uint32_t start = range.first << page_size_log2();
    uint32_t length = range.second << page_size_log2();
    MakeRangeValid(start, length, false);
    while (length) {
      uint32_t g = start >> kGranuleLog2;
      uint32_t in = start & (kGranuleSize - 1);
      uint32_t part = std::min(length, kGranuleSize - in);
      if (g >= kGranuleCount || page_table_[g] == 0xFFFFFFFFu) {
        REXGPU_ERROR("GL shared memory: upload outside resident granule (address {:08X}+{:X})", start, part);
        return false;
      }
      glBindBuffer(GL_COPY_WRITE_BUFFER, store_);
      perf::Add(perf::kSmUploadBytes, part);
      glBufferSubData(GL_COPY_WRITE_BUFFER, GLintptr(page_table_[g]) * 4 + in, part,
                      memory().TranslatePhysical(start));
      start += part;
      length -= part;
    }
  }
  return true;
}

bool GlSharedMemory::GetHostRange(uint32_t address, uint32_t length, GLuint& buffer, GLintptr& offset) const {
  uint32_t g0 = address >> kGranuleLog2;
  uint32_t g1 = (address + std::max(length, 1u) - 1) >> kGranuleLog2;
  if (g1 >= kGranuleCount || page_table_[g0] == 0xFFFFFFFFu) {
    return false;
  }
  for (uint32_t g = g0 + 1; g <= g1; ++g) {
    if (page_table_[g] != page_table_[g - 1] + (kGranuleSize >> 2)) {
      return false;
    }
  }
  buffer = store_;
  offset = GLintptr(page_table_[g0]) * 4 + (address & (kGranuleSize - 1));
  return true;
}

void GlSharedMemory::BindStorageBuffers(uint32_t first_binding) {
  (void)first_binding;
  glActiveTexture(GL_TEXTURE0 + kSharedMemoryTboUnitBase);
  glBindTexture(GL_TEXTURE_BUFFER, store_tbo_);
  glBindSampler(kSharedMemoryTboUnitBase, 0);
  glActiveTexture(GL_TEXTURE0 + kSharedMemoryTboUnitBase + 1);
  glBindTexture(GL_TEXTURE_BUFFER, pt_tbo_);
  glBindSampler(kSharedMemoryTboUnitBase + 1, 0);
  glActiveTexture(GL_TEXTURE0);
}

}  // namespace rex::graphics::gl
