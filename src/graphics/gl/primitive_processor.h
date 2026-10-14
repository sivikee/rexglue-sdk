/**
 * @file        graphics/gl/primitive_processor.h
 * @brief       GL primitive processor: guest primitive type / index conversion, GL index buffers.
 */
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/primitive_processor.h>

namespace rex::graphics::gl {

class GlPrimitiveProcessor final : public PrimitiveProcessor {
 public:
  GlPrimitiveProcessor(const RegisterFile& register_file, memory::Memory& memory,
                       SharedMemory& shared_memory)
      : PrimitiveProcessor(register_file, memory, shared_memory) {}
  ~GlPrimitiveProcessor() override;

  bool Initialize();
  void Shutdown(bool from_destructor = false);
  void EndFrame() {
    ClearPerFrameCache();
    frame_converted_.clear();
    frame_chunk_used_ = 0;
    frame_chunk_current_ = 0;
  }

  // Uploads/returns a GL element buffer for a host-converted handle (valid until the next call).
  GLuint UseConvertedIndexBuffer(size_t handle, GLintptr& offset_out);
  GLuint builtin_index_buffer() const { return builtin_index_buffer_; }
  GLintptr GetBuiltinOffset(size_t handle) const {
    return GLintptr(GetBuiltinIndexBufferOffsetBytes(handle));
  }

 protected:
  bool InitializeBuiltinIndexBuffer(size_t size_bytes,
                                    std::function<void(void*)> fill_callback) override;
  void* RequestHostConvertedIndexBufferForCurrentFrame(xenos::IndexFormat format,
                                                       uint32_t index_count, bool coalign_for_simd,
                                                       uint32_t coalignment_original_address,
                                                       size_t& backend_handle_out) override;

 private:
  GLuint builtin_index_buffer_ = 0;
  GLuint stream_index_buffer_ = 0;
  size_t stream_index_buffer_size_ = 0;

  // CPU arena for converted indices of the current frame (stable pointers: chunk list).
  struct Converted {
    uint8_t* data;
    size_t size;
  };
  std::deque<std::vector<uint8_t>> frame_chunks_;
  size_t frame_chunk_current_ = 0;
  size_t frame_chunk_used_ = 0;
  std::vector<Converted> frame_converted_;
};

}  // namespace rex::graphics::gl
