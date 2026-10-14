#include "primitive_processor.h"

#include <algorithm>
#include <cstring>

#include <rex/logging.h>

namespace rex::graphics::gl {

GlPrimitiveProcessor::~GlPrimitiveProcessor() {
  Shutdown(true);
}

bool GlPrimitiveProcessor::Initialize() {
  // GL 4.3 has geometry shaders, but the Xenos SPIR-V geometry emulation of the Vulkan backend is not
  // ported: use the vertex-shader expansion fallbacks (point sprites, rectangle lists) and convert quad
  // lists and triangle fans to triangle lists on the CPU.
  if (!InitializeCommon(/*full_32bit_vertex_indices_supported=*/true,
                        /*triangle_fans_supported=*/false, /*line_loops_supported=*/false,
                        /*quad_lists_supported=*/false,
                        /*point_sprites_supported_without_vs_expansion=*/false,
                        /*rectangle_lists_supported_without_vs_expansion=*/false)) {
    Shutdown();
    return false;
  }
  glGenBuffers(1, &stream_index_buffer_);
  return true;
}

void GlPrimitiveProcessor::Shutdown(bool from_destructor) {
  frame_converted_.clear();
  frame_chunks_.clear();
  if (stream_index_buffer_) {
    glDeleteBuffers(1, &stream_index_buffer_);
    stream_index_buffer_ = 0;
  }
  if (builtin_index_buffer_) {
    glDeleteBuffers(1, &builtin_index_buffer_);
    builtin_index_buffer_ = 0;
  }
  if (!from_destructor) {
    ShutdownCommon();
  }
}

bool GlPrimitiveProcessor::InitializeBuiltinIndexBuffer(size_t size_bytes,
                                                        std::function<void(void*)> fill_callback) {
  std::vector<uint8_t> data(size_bytes);
  fill_callback(data.data());
  glGenBuffers(1, &builtin_index_buffer_);
  glBindBuffer(GL_COPY_WRITE_BUFFER, builtin_index_buffer_);
  glBufferData(GL_COPY_WRITE_BUFFER, GLsizeiptr(size_bytes), data.data(), GL_STATIC_DRAW);
  return true;
}

void* GlPrimitiveProcessor::RequestHostConvertedIndexBufferForCurrentFrame(
    xenos::IndexFormat format, uint32_t index_count, bool coalign_for_simd,
    uint32_t coalignment_original_address, size_t& backend_handle_out) {
  size_t index_size = format == xenos::IndexFormat::kInt16 ? sizeof(uint16_t) : sizeof(uint32_t);
  size_t needed = index_size * index_count + (coalign_for_simd ? XE_GPU_PRIMITIVE_PROCESSOR_SIMD_SIZE : 0);
  needed = (needed + 15) & ~size_t(15);
  // Find space in the current chunk, or open a new one.
  if (frame_chunks_.empty() || frame_chunk_current_ >= frame_chunks_.size() ||
      frame_chunks_[frame_chunk_current_].size() - frame_chunk_used_ < needed) {
    if (!frame_chunks_.empty() && frame_chunk_current_ + 1 < frame_chunks_.size() &&
        frame_chunks_[frame_chunk_current_ + 1].size() >= needed) {
      ++frame_chunk_current_;
    } else {
      size_t chunk_size = std::max(needed, size_t(4) << 20);
      frame_chunks_.emplace_back(chunk_size + 64);
      frame_chunk_current_ = frame_chunks_.size() - 1;
    }
    frame_chunk_used_ = 0;
  }
  uint8_t* base = frame_chunks_[frame_chunk_current_].data();
  // 16-byte alignment of the chunk start is not guaranteed by vector, align manually.
  uintptr_t aligned = (reinterpret_cast<uintptr_t>(base) + 15) & ~uintptr_t(15);
  uint8_t* mapping = reinterpret_cast<uint8_t*>(aligned) + frame_chunk_used_;
  frame_chunk_used_ += needed;
  uint8_t* data = mapping;
  if (coalign_for_simd) {
    data += GetSimdCoalignmentOffset(mapping, coalignment_original_address);
  }
  backend_handle_out = frame_converted_.size();
  frame_converted_.push_back({data, index_size * index_count});
  return data;
}

GLuint GlPrimitiveProcessor::UseConvertedIndexBuffer(size_t handle, GLintptr& offset_out) {
  const Converted& c = frame_converted_[handle];
  glBindBuffer(GL_COPY_WRITE_BUFFER, stream_index_buffer_);
  if (stream_index_buffer_size_ < c.size) {
    stream_index_buffer_size_ = std::max(c.size, size_t(1) << 20);
    glBufferData(GL_COPY_WRITE_BUFFER, GLsizeiptr(stream_index_buffer_size_), nullptr, GL_STREAM_DRAW);
  }
  // Orphan to avoid implicit sync with the previous draw using this buffer.
  glBufferData(GL_COPY_WRITE_BUFFER, GLsizeiptr(stream_index_buffer_size_), nullptr, GL_STREAM_DRAW);
  glBufferSubData(GL_COPY_WRITE_BUFFER, 0, GLsizeiptr(c.size), c.data);
  offset_out = 0;
  return stream_index_buffer_;
}

}  // namespace rex::graphics::gl
