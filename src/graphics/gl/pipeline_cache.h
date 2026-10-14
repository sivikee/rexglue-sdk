/**
 * @file        graphics/gl/pipeline_cache.h
 * @brief       Shader translation (Xenos ucode -> SPIR-V -> GLSL) and linked GL program cache.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/string/buffer.h>

#include "shader.h"

namespace rex::graphics::gl {

struct GlProgram {
  GLuint program = 0;
  std::atomic<bool> failed{false};
  std::atomic<bool> ready{false};
  GlShader::GlTranslation* vertex = nullptr;
  GlShader::GlTranslation* pixel = nullptr;
};

class GlPipelineCache {
 public:
  explicit GlPipelineCache(const RegisterFile& register_file) : register_file_(register_file) {}
  ~GlPipelineCache();

  bool Initialize(class GlContext* gl_context);
  void Shutdown();

  GlShader* LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                       uint32_t dword_count);
  void AnalyzeShaderUcode(Shader& shader) { shader.AnalyzeUcode(ucode_disasm_buffer_); }

  SpirvShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
      uint32_t interpolator_mask, bool ps_param_gen_used) const;
  SpirvShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
      reg::RB_DEPTHCONTROL normalized_depth_control) const;

  // Translates (if needed) the shader(s).
  bool EnsureShadersTranslated(GlShader::GlTranslation* vertex, GlShader::GlTranslation* pixel);
  // Returns the linked program (or nullptr on failure).
  const GlProgram* GetProgram(GlShader::GlTranslation* vertex, GlShader::GlTranslation* pixel);

  uint32_t translated_shader_count() const { return translated_count_.load(); }
  uint32_t pending_programs() const { return pending_.load(); }

 private:
  const RegisterFile& register_file_;
  std::unique_ptr<SpirvShaderTranslator> translator_;
  string::StringBuffer ucode_disasm_buffer_;
  std::unordered_map<uint64_t, std::unique_ptr<GlShader>> shaders_;

  struct ProgramKey {
    const void* vertex;
    const void* pixel;
    bool operator==(const ProgramKey& o) const { return vertex == o.vertex && pixel == o.pixel; }
  };
  struct ProgramKeyHash {
    size_t operator()(const ProgramKey& k) const {
      return std::hash<const void*>()(k.vertex) * 31 + std::hash<const void*>()(k.pixel);
    }
  };
  std::unordered_map<ProgramKey, GlProgram, ProgramKeyHash> programs_;
  void BuildProgram(GlProgram& program);
  void WorkerMain();
  static void WorkerThunk(void* self);
  class GlContext* gl_context_ = nullptr;
  void* worker_thread_ = nullptr;
  void* worker_context_ = nullptr;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::pair<GlShader::GlTranslation*, GlShader::GlTranslation*>> queue_;
  std::atomic<bool> running_{false};
  bool async_ = true;
  std::atomic<uint32_t> pending_{0};
  std::atomic<uint32_t> translated_count_{0};
  uint32_t dumped_ = 0;
};

}  // namespace rex::graphics::gl
