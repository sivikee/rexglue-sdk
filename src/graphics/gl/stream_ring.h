/**
 * @file        graphics/gl/stream_ring.h
 * @brief       Streaming upload ring for the GL backend. Avoids the implicit CPU<->GPU synchronisation
 *              (nouveau stalls glBufferSubData / glTexSubImage into buffers the GPU still uses) by writing
 *              only into ring regions the GPU has finished with. Persistent-mapped (ARB_buffer_storage)
 *              when available, otherwise glMapBufferRange with MAP_UNSYNCHRONIZED. The ring is split in
 *              segments, each guarded by a fence that is armed (Commit) after the commands reading it
 *              were issued.
 */
#pragma once

#include <cstdint>

#include <glad/glad.h>

namespace rex::graphics::gl {

class StreamRing {
 public:
  static constexpr uint32_t kMaxSegments = 32;
  ~StreamRing() { Shutdown(); }

  bool Initialize(uint32_t size, uint32_t segment_count, GLenum target, bool allow_persistent);
  void Shutdown();
  bool valid() const { return buffer_ != 0; }
  GLuint buffer() const { return buffer_; }
  uint32_t segment_size() const { return segment_size_; }
  bool persistent() const { return persistent_; }

  // Reserves size bytes (size <= segment_size). Returns a CPU pointer to write to (valid until the next
  // Commit()) and the offset in the buffer. Blocks only if the GPU is still reading the segment.
  void* Alloc(uint32_t size, uint32_t alignment, uint32_t& offset_out);
  // Call after the GPU commands that read the allocations made so far have been issued. Unmaps in the
  // non-persistent mode and arms fences of the segments that were left.
  void Commit();

 private:
  void Enter(int segment);
  void Leave(int segment);

  GLuint buffer_ = 0;
  GLenum target_ = GL_ARRAY_BUFFER;
  uint32_t size_ = 0, segment_size_ = 0, segment_count_ = 0;
  uint32_t head_ = 0;
  int cur_seg_ = 0;
  bool persistent_ = false;
  uint8_t* mapped_ = nullptr;
  uint8_t* fallback_map_ = nullptr;
  uint32_t fallback_map_begin_ = 0;
  uint32_t fallback_map_end_ = 0;
  GLsync fences_[kMaxSegments] = {};
  bool unfenced_[kMaxSegments] = {};
};

}  // namespace rex::graphics::gl
