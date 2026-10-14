#include "pipeline_cache.h"

#include <chrono>
#include <cstdio>
#include <string>

#include <xxhash.h>

#include <rex/cvar.h>
#include <rex/graphics/util/draw.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(gl_dump_shaders, "", "GPU",
                      "GL backend: directory to dump generated GLSL of failing/first shaders (empty = off)");

namespace rex::graphics::gl {

GlPipelineCache::~GlPipelineCache() {
  Shutdown();
}

bool GlPipelineCache::Initialize() {
  SpirvShaderTranslator::Features features(false);
  features.max_storage_buffer_range = 128 * 1024 * 1024;
  features.full_draw_index_uint32 = true;
  features.vertex_pipeline_stores_and_atomics = false;  // memexport not supported yet
  features.fragment_stores_and_atomics = false;
  features.clip_distance = true;
  features.cull_distance = true;
  features.image_view_format_swizzle = false;  // host swizzle done in the shader via system constants
  features.signed_zero_inf_nan_preserve_float32 = false;
  features.denorm_flush_to_zero_float32 = false;
  features.rounding_mode_rte_float32 = false;
  features.fragment_shader_sample_interlock = false;
  features.demote_to_helper_invocation = false;
  features.sample_rate_shading = false;
  translator_ = std::make_unique<SpirvShaderTranslator>(features, false, false, false, 1, 1);
  return true;
}

void GlPipelineCache::Shutdown() {
  for (auto& p : programs_) {
    if (p.second.program) {
      glDeleteProgram(p.second.program);
    }
  }
  programs_.clear();
  shaders_.clear();
}

GlShader* GlPipelineCache::LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                                      uint32_t dword_count) {
  uint64_t hash = XXH3_64bits(host_address, dword_count * sizeof(uint32_t));
  auto it = shaders_.find(hash);
  if (it != shaders_.end()) {
    return it->second.get();
  }
  auto shader = std::make_unique<GlShader>(shader_type, hash, host_address, dword_count);
  GlShader* ptr = shader.get();
  shaders_.emplace(hash, std::move(shader));
  return ptr;
}

SpirvShaderTranslator::Modification GlPipelineCache::GetCurrentVertexShaderModification(
    const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
    uint32_t interpolator_mask, bool ps_param_gen_used) const {
  const auto& regs = register_file_;
  SpirvShaderTranslator::Modification modification(translator_->GetDefaultVertexShaderModification(
      shader.GetDynamicAddressableRegisterCount(regs.Get<reg::SQ_PROGRAM_CNTL>().vs_num_reg),
      host_vertex_shader_type));
  modification.vertex.interpolator_mask = interpolator_mask;
  if (Shader::IsHostVertexShaderTypeDomain(host_vertex_shader_type)) {
    modification.vertex.tessellation_mode = uint32_t(regs.Get<reg::VGT_HOS_CNTL>().tess_mode);
  } else {
    modification.vertex.tessellation_mode = 0;
  }
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  uint32_t user_clip_planes = pa_cl_clip_cntl.clip_disable ? 0 : pa_cl_clip_cntl.ucp_ena;
  modification.vertex.user_clip_plane_cull =
      (user_clip_planes && pa_cl_clip_cntl.ucp_cull_only_ena) ? 1 : 0;
  modification.vertex.user_clip_plane_count = rex::bit_count(user_clip_planes);
  modification.vertex.point_ps_ucp_mode = pa_cl_clip_cntl.ps_ucp_mode;
  if (host_vertex_shader_type == Shader::HostVertexShaderType::kPointListAsTriangleStrip) {
    modification.vertex.output_point_parameters = uint32_t(ps_param_gen_used);
  } else {
    modification.vertex.output_point_parameters =
        uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b001) &&
                 regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList);
  }
  modification.vertex.vertex_kill_and = uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b100) &&
                                                 !pa_cl_clip_cntl.vtx_kill_or);
  return modification;
}

SpirvShaderTranslator::Modification GlPipelineCache::GetCurrentPixelShaderModification(
    const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
    reg::RB_DEPTHCONTROL normalized_depth_control) const {
  const auto& regs = register_file_;
  SpirvShaderTranslator::Modification modification(translator_->GetDefaultPixelShaderModification(
      shader.GetDynamicAddressableRegisterCount(regs.Get<reg::SQ_PROGRAM_CNTL>().ps_num_reg)));
  modification.pixel.interpolator_mask = interpolator_mask;
  modification.pixel.interpolators_centroid =
      interpolator_mask & ~xenos::GetInterpolatorSamplingPattern(
                              regs.Get<reg::RB_SURFACE_INFO>().msaa_samples,
                              regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
                              regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);
  if (param_gen_pos < xenos::kMaxInterpolators) {
    modification.pixel.param_gen_enable = 1;
    modification.pixel.param_gen_interpolator = param_gen_pos;
    modification.pixel.param_gen_point =
        uint32_t(regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList);
  } else {
    modification.pixel.param_gen_enable = 0;
    modification.pixel.param_gen_interpolator = 0;
    modification.pixel.param_gen_point = 0;
  }
  using DepthStencilMode = SpirvShaderTranslator::Modification::DepthStencilMode;
  if (shader.implicit_early_z_write_allowed() &&
      (!shader.writes_color_target(0) ||
       !draw_util::DoesCoverageDependOnAlpha(regs.Get<reg::RB_COLORCONTROL>()))) {
    modification.pixel.depth_stencil_mode = DepthStencilMode::kEarlyHint;
  } else {
    modification.pixel.depth_stencil_mode = DepthStencilMode::kNoModifiers;
  }
  return modification;
}

bool GlPipelineCache::EnsureShadersTranslated(GlShader::GlTranslation* vertex,
                                              GlShader::GlTranslation* pixel) {
  auto translate = [&](GlShader::GlTranslation* t) -> bool {
    if (!t->is_translated()) {
      t->shader().AnalyzeUcode(ucode_disasm_buffer_);
      auto t0 = std::chrono::steady_clock::now();
      if (!translator_->TranslateAnalyzedShader(*t)) {
        REXGPU_ERROR("GL: shader {:016X} translation failed", t->shader().ucode_data_hash());
        return false;
      }
      ++translated_count_;
      double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (translated_count_ <= 8 || translated_count_ % 25 == 0) {
        REXGPU_INFO("GL: translated shader #{} {:016X} mod {:016X} ({}) in {:.1f} ms, SPIR-V {} bytes",
                    translated_count_, t->shader().ucode_data_hash(), t->modification(),
                    t->shader().type() == xenos::ShaderType::kVertex ? "VS" : "PS", ms,
                    t->translated_binary().size());
      }
    }
    return t->is_valid();
  };
  if (!translate(vertex)) {
    return false;
  }
  if (pixel && !translate(pixel)) {
    return false;
  }
  return true;
}

namespace {
GLuint CompileStage(GLenum type, const std::string& source, std::string& log_out) {
  GLuint s = glCreateShader(type);
  const char* src = source.c_str();
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    GLint len = 0;
    glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
    log_out.assign(size_t(len > 1 ? len : 1), 0);
    glGetShaderInfoLog(s, len, nullptr, log_out.data());
    glDeleteShader(s);
    return 0;
  }
  return s;
}
}  // namespace

const GlProgram* GlPipelineCache::GetProgram(GlShader::GlTranslation* vertex,
                                             GlShader::GlTranslation* pixel) {
  ProgramKey key{vertex, pixel};
  auto it = programs_.find(key);
  if (it != programs_.end()) {
    return it->second.failed ? nullptr : &it->second;
  }
  GlProgram& program = programs_[key];
  program.vertex = vertex;
  program.pixel = pixel;
  program.failed = true;
  auto dump = [&](const char* tag, const std::string& source, uint64_t hash) {
    std::string dir = REXCVAR_GET(gl_dump_shaders);
    if (dir.empty()) {
      return;
    }
    char name[256];
    std::snprintf(name, sizeof(name), "%s/%s_%016llX.glsl", dir.c_str(), tag,
                  (unsigned long long)hash);
    if (FILE* f = std::fopen(name, "wb")) {
      std::fwrite(source.data(), 1, source.size(), f);
      std::fclose(f);
    }
  };
  const GlslShader* vs_glsl = vertex->GetOrCreateGlsl();
  const GlslShader* ps_glsl = pixel ? pixel->GetOrCreateGlsl() : nullptr;
  if (!vs_glsl || (pixel && !ps_glsl)) {
    return nullptr;
  }
  auto t0 = std::chrono::steady_clock::now();
  std::string log;
  GLuint vs = CompileStage(GL_VERTEX_SHADER, vs_glsl->source, log);
  if (!vs) {
    REXGPU_ERROR("GL: vertex shader {:016X} mod {:016X} failed to compile: {}",
                 vertex->shader().ucode_data_hash(), vertex->modification(), log);
    dump("vs_fail", vs_glsl->source, vertex->shader().ucode_data_hash());
    return nullptr;
  }
  GLuint fs = 0;
  if (pixel) {
    fs = CompileStage(GL_FRAGMENT_SHADER, ps_glsl->source, log);
    if (!fs) {
      REXGPU_ERROR("GL: pixel shader {:016X} mod {:016X} failed to compile: {}",
                   pixel->shader().ucode_data_hash(), pixel->modification(), log);
      dump("ps_fail", ps_glsl->source, pixel->shader().ucode_data_hash());
      glDeleteShader(vs);
      return nullptr;
    }
  }
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  if (fs) {
    glAttachShader(prog, fs);
  }
  glLinkProgram(prog);
  GLint ok = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  glDeleteShader(vs);
  if (fs) {
    glDeleteShader(fs);
  }
  if (!ok) {
    GLint len = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    log.assign(size_t(len > 1 ? len : 1), 0);
    glGetProgramInfoLog(prog, len, nullptr, log.data());
    REXGPU_ERROR("GL: program link failed (vs {:016X}, ps {:016X}): {}",
                 vertex->shader().ucode_data_hash(),
                 pixel ? pixel->shader().ucode_data_hash() : uint64_t(0), log);
    dump("vs_link", vs_glsl->source, vertex->shader().ucode_data_hash());
    if (ps_glsl) {
      dump("ps_link", ps_glsl->source, pixel->shader().ucode_data_hash());
    }
    glDeleteProgram(prog);
    return nullptr;
  }
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (programs_.size() <= 8 || programs_.size() % 25 == 0) {
    REXGPU_INFO("GL: linked program #{} in {:.1f} ms", programs_.size(), ms);
  }
  if (dumped_ < 4) {
    ++dumped_;
    dump("vs_ok", vs_glsl->source, vertex->shader().ucode_data_hash());
    if (ps_glsl) {
      dump("ps_ok", ps_glsl->source, pixel->shader().ucode_data_hash());
    }
  }
  program.program = prog;
  program.failed = false;
  return &program;
}

}  // namespace rex::graphics::gl
