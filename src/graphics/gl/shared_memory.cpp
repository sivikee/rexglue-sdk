#include "shared_memory.h"

#include <algorithm>
#include <cstring>

#include <rex/logging.h>

#include "gl_ext.h"
#include "spirv_to_glsl.h"

namespace rex::graphics::gl {

bool GlSharedMemory::Initialize() {
  InitializeCommon();
  InitializeSparseHostGpuMemory(kGranuleLog2);
  glGenBuffers(1, &dummy_);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, dummy_);
  uint8_t zeros[256] = {};
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zeros), zeros, GL_STATIC_DRAW);
  glGenTextures(kChunkCount, tbos_);
  for (uint32_t i = 0; i < kChunkCount; ++i) {
    glBindTexture(GL_TEXTURE_BUFFER, tbos_[i]);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, dummy_);
  }
  return true;
}

void GlSharedMemory::Shutdown(bool from_destructor) {
  for (uint32_t i = 0; i < kChunkCount; ++i) {
    if (chunks_[i]) {
      glDeleteBuffers(1, &chunks_[i]);
      chunks_[i] = 0;
      chunk_sizes_[i] = 0;
    }
  }
  if (tbos_[0]) {
    glDeleteTextures(kChunkCount, tbos_);
    for (auto& t : tbos_) t = 0;
  }
  if (dummy_) {
    glDeleteBuffers(1, &dummy_);
    dummy_ = 0;
  }
  if (!from_destructor) {
    ShutdownCommon();
  }
}

uint32_t GlSharedMemory::total_bytes() const {
  uint32_t total = 0;
  for (uint32_t s : chunk_sizes_) total += s;
  return total;
}

bool GlSharedMemory::EnsureChunkSize(uint32_t chunk_index, uint32_t size) {
  if (chunk_index >= kChunkCount) {
    return false;
  }
  size = std::min(size, kChunkSize);
  if (chunk_sizes_[chunk_index] >= size) {
    return true;
  }
  GLuint buffer;
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_COPY_WRITE_BUFFER, buffer);
  glBufferData(GL_COPY_WRITE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    REXGPU_ERROR("GL shared memory: failed to allocate chunk {} of {} MiB (total resident {} MiB): error {:x}",
                 chunk_index, size >> 20, total_bytes() >> 20, err);
    glDeleteBuffers(1, &buffer);
    return false;
  }
  if (chunks_[chunk_index]) {
    // Preserve what was uploaded (pages are considered valid by the base class).
    glBindBuffer(GL_COPY_READ_BUFFER, chunks_[chunk_index]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, chunk_sizes_[chunk_index]);
    glDeleteBuffers(1, &chunks_[chunk_index]);
  }
  chunks_[chunk_index] = buffer;
  chunk_sizes_[chunk_index] = size;
  glBindTexture(GL_TEXTURE_BUFFER, tbos_[chunk_index]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, buffer);
  REXGPU_INFO("GL shared memory: chunk {} resident {} MiB (total {} MiB)", chunk_index, size >> 20,
              total_bytes() >> 20);
  return true;
}

bool GlSharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                      uint32_t length_allocations) {
  for (uint32_t g = offset_allocations; g < offset_allocations + length_allocations; ++g) {
    uint32_t chunk_index = (g << kGranuleLog2) >> kChunkSizeLog2;
    uint32_t end_in_chunk = (((g << kGranuleLog2) & (kChunkSize - 1)) + (1u << kGranuleLog2));
    if (!EnsureChunkSize(chunk_index, end_in_chunk)) {
      return false;
    }
  }
  return true;
}

bool GlSharedMemory::UploadRanges(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) {
  for (const auto& range : upload_page_ranges) {
    uint32_t start = range.first << page_size_log2();
    uint32_t length = range.second << page_size_log2();
    MakeRangeValid(start, length, false);
    while (length) {
      uint32_t chunk_index = start >> kChunkSizeLog2;
      uint32_t chunk_offset = start & (kChunkSize - 1);
      uint32_t part = std::min(length, kChunkSize - chunk_offset);
      if (chunk_index >= kChunkCount || chunk_sizes_[chunk_index] < chunk_offset + part) {
        REXGPU_ERROR("GL shared memory: upload outside resident chunk {} (offset {:X}+{:X}, size {:X})",
                     chunk_index, chunk_offset, part, chunk_index < kChunkCount ? chunk_sizes_[chunk_index] : 0);
        return false;
      }
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunks_[chunk_index]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, chunk_offset, part, memory().TranslatePhysical(start));
      start += part;
      length -= part;
    }
  }
  return true;
}

void GlSharedMemory::BindStorageBuffers(uint32_t first_binding) {
  for (uint32_t i = 0; i < kChunkCount; ++i) {
    glActiveTexture(GL_TEXTURE0 + kSharedMemoryTboUnitBase + i);
    glBindTexture(GL_TEXTURE_BUFFER, tbos_[i]);
    glBindSampler(kSharedMemoryTboUnitBase + i, 0);
  }
  glActiveTexture(GL_TEXTURE0);
  for (uint32_t i = 0; i < kChunkCount; ++i) {
    if (chunks_[i]) {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, first_binding + i, chunks_[i], 0, chunk_sizes_[i]);
    } else {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, first_binding + i, dummy_, 0, 256);
    }
  }
}

}  // namespace rex::graphics::gl
