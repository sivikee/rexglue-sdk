/**
 * @file        graphics/gl/command_processor.h
 * @brief       OpenGL Xenos command processor (Switch port).
 */
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/command_processor.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/primitive_processor.h>
#include <rex/graphics/util/draw.h>

#include "gl_context.h"

namespace rex::graphics::gl {

class GlSharedMemory;
class GlPrimitiveProcessor;
class GlPipelineCache;
class GlRenderTargets;
class GlTextureCache;

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
  void DrawFpsOverlay(int sw, int sh);
  void ShaderPrecompileScreen();
  void ValidateBufferByHash(uint32_t address, uint32_t size);

 private:
  struct Impl;
  void UpdateSystemConstants(bool primitive_polygonal, const PrimitiveProcessor::ProcessingResult& ppr,
                             const draw_util::ViewportInfo& viewport_info, uint32_t used_texture_mask,
                             reg::RB_DEPTHCONTROL normalized_depth_control);
  void ApplyRasterState(bool primitive_polygonal, reg::RB_DEPTHCONTROL depth_control,
                        uint32_t color_mask, const draw_util::ViewportInfo& viewport_info,
                        bool depth_stencil_bound);
  void UploadAndBindConstants(const Shader* vertex_shader, const Shader* pixel_shader);
  bool PrepareVertexBuffers(const Shader& vertex_shader);

  std::unique_ptr<Impl> impl_;
  GlContext context_;
  uint32_t swap_count_ = 0;
  uint32_t draw_count_ = 0;
  uint32_t draw_fail_count_ = 0;
  uint32_t draws_skipped_ = 0;
  uint32_t frame_draws_ = 0;
  bool dbg_log_vertex_ = false;
  SpirvShaderTranslator::SystemConstants system_constants_;
};

}  // namespace rex::graphics::gl
