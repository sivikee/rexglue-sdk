/**
 * @file        graphics/gl/command_processor.h
 * @brief       OpenGL Xenos command processor (Switch port).
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rex/graphics/command_processor.h>

#include "gl_context.h"

namespace rex::graphics::gl {

class GlCommandProcessor : public CommandProcessor {
 public:
  GlCommandProcessor(GraphicsSystem* graphics_system, system::KernelState* kernel_state);
  ~GlCommandProcessor() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                 uint32_t frontbuffer_height) override;

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;

  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address, uint32_t dword_count) override;
  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info, bool major_mode_explicit) override;
  bool IssueCopy() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  GlContext context_;
  uint32_t swap_count_ = 0;
};

}  // namespace rex::graphics::gl
