#include "stream_ring.h"

#include <algorithm>

#include <rex/logging.h>

#include "gl_ext.h"
#include "perf.h"

namespace rex::graphics::gl {

bool StreamRing::Initialize(uint32_t size, uint32_t segment_count, GLenum target, bool allow_persistent) {
  Shutdown();
  segment_count_ = std::min(segment_count, kMaxSegments);
  segment_size_ = size / segment_count_;
  size_ = segment_size_ * segment_count_;
  target_ = target;
  glGenBuffers(1, &buffer_);
  glBindBuffer(target_, buffer_);
  persistent_ = false;
  if (allow_persistent && glBufferStorage_) {
    const GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    while (glGetError() != GL_NO_ERROR) {
    }
    glBufferStorage_(target_, size_, nullptr, flags);
    if (glGetError() == GL_NO_ERROR) {
      void* p = glMapBufferRange(target_, 0, size_, flags);
      if (p && glGetError() == GL_NO_ERROR) {
        mapped_ = static_cast<uint8_t*>(p);
        persistent_ = true;
      }
    }
    if (!persistent_) {
      // Immutable storage cannot be re-specified: recreate the buffer for the fallback.
      while (glGetError() != GL_NO_ERROR) {
      }
      glBindBuffer(target_, 0);
      glDeleteBuffers(1, &buffer_);
      glGenBuffers(1, &buffer_);
      glBindBuffer(target_, buffer_);
    }
  }
  if (!persistent_) {
    glBufferData(target_, size_, nullptr, GL_STREAM_DRAW);
    if (glGetError() != GL_NO_ERROR) {
      glBindBuffer(target_, 0);
      glDeleteBuffers(1, &buffer_);
      buffer_ = 0;
      return false;
    }
  }
  glBindBuffer(target_, 0);
  head_ = 0;
  cur_seg_ = 0;
  REXGPU_INFO("GL: stream ring {} KiB x {} ({})", segment_size_ >> 10, segment_count_,
              persistent_ ? "persistent" : "unsynchronized map");
  return true;
}

void StreamRing::Shutdown() {
  for (auto& f : fences_) {
    if (f) glDeleteSync(f);
    f = nullptr;
  }
  for (auto& u : unfenced_) u = false;
  if (buffer_) {
    glBindBuffer(target_, buffer_);
    if (persistent_ || fallback_map_) glUnmapBuffer(target_);
    glBindBuffer(target_, 0);
    glDeleteBuffers(1, &buffer_);
  }
  buffer_ = 0;
  mapped_ = fallback_map_ = nullptr;
  persistent_ = false;
}

void StreamRing::Leave(int segment) {
  if (segment >= 0) unfenced_[segment] = true;
}

void StreamRing::Enter(int segment) {
  if (unfenced_[segment]) {
    Commit();  // wrapped onto a segment that was left but never fenced
  }
  if (fences_[segment]) {
    perf::FenceWaitLogged(fences_[segment], "stream ring");
    glDeleteSync(fences_[segment]);
    fences_[segment] = nullptr;
  }
}

void* StreamRing::Alloc(uint32_t size, uint32_t alignment, uint32_t& offset_out) {
  if (!buffer_ || size == 0 || size > segment_size_) return nullptr;
  alignment = std::max(alignment, 1u);
  uint32_t start = (head_ + alignment - 1) / alignment * alignment;
  if (start / segment_size_ != (start + size - 1) / segment_size_) {
    start = (start / segment_size_ + 1) * segment_size_;  // never straddle segments
  }
  if (start + size > size_) {
    for (int s = cur_seg_; s < int(segment_count_); ++s) Leave(s);
    start = 0;
    cur_seg_ = -1;
  }
  int last = int((start + size - 1) / segment_size_);
  for (int s = cur_seg_ + 1; s <= last; ++s) {
    Leave(s - 1);
    Enter(s);
    cur_seg_ = s;
  }
  head_ = start + size;
  offset_out = start;
  if (persistent_) return mapped_ + start;
  // Fallback: map a window covering this allocation without synchronisation.
  glBindBuffer(target_, buffer_);
  if (fallback_map_ && start >= fallback_map_begin_ && start + size <= fallback_map_end_) {
    return fallback_map_ + (start - fallback_map_begin_);
  }
  if (fallback_map_) {
    glFlushMappedBufferRange(target_, 0, fallback_map_end_ - fallback_map_begin_);
    glUnmapBuffer(target_);
    fallback_map_ = nullptr;
  }
  uint32_t map_begin = start;
  uint32_t map_end = std::max((start / segment_size_ + 1) * segment_size_, start + size);
  fallback_map_ = static_cast<uint8_t*>(
      glMapBufferRange(target_, map_begin, map_end - map_begin,
                       GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT | GL_MAP_FLUSH_EXPLICIT_BIT));
  fallback_map_begin_ = map_begin;
  fallback_map_end_ = map_end;
  return fallback_map_ ? fallback_map_ : nullptr;
}

void StreamRing::Commit() {
  if (fallback_map_) {
    glBindBuffer(target_, buffer_);
    glFlushMappedBufferRange(target_, 0, fallback_map_end_ - fallback_map_begin_);
    glUnmapBuffer(target_);
    fallback_map_ = nullptr;
  }
  for (uint32_t s = 0; s < segment_count_; ++s) {
    if (!unfenced_[s]) continue;
    if (fences_[s]) glDeleteSync(fences_[s]);
    fences_[s] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    unfenced_[s] = false;
  }
}

}  // namespace rex::graphics::gl
