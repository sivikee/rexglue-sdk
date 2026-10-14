#include "pipeline_cache.h"

#include <chrono>
#include <cstdio>
#include <string>

#include <switch.h>
#include <xxhash.h>
#include "gl_context.h"
#include "gl_ext.h"

#include <rex/cvar.h>
#include <rex/graphics/util/draw.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(gl_parallel_compile, false, "GPU", "GL backend: enable GL_KHR_parallel_shader_compile driver threads");
REXCVAR_DEFINE_BOOL(gl_async_compile, true, "GPU", "GL backend: translate and compile shader programs on a worker thread (draws are skipped until ready)");
REXCVAR_DEFINE_STRING(gl_dump_shaders, "", "GPU",
                      "GL backend: directory to dump generated GLSL of failing/first shaders (empty = off)");

namespace rex::graphics::gl {

GlPipelineCache::~GlPipelineCache() {
  Shutdown();
}

bool GlPipelineCache::Initialize(GlContext* gl_context) {
  gl_context_ = gl_context;
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
  if (glMaxShaderCompilerThreadsKHR_ && REXCVAR_GET(gl_parallel_compile)) {
    glMaxShaderCompilerThreadsKHR_(0xFFFFFFFFu);
  }
  async_ = REXCVAR_GET(gl_async_compile);
  if (async_) {
    worker_context_ = reinterpret_cast<void*>(1);
    if (!worker_context_) {
      async_ = false;
    } else {
      running_ = true;
      Thread* t = new Thread();
      if (R_FAILED(threadCreate(t, &GlPipelineCache::WorkerThunk, this, nullptr, 0x400000, 0x2C, -2)) || R_FAILED(threadStart(t))) {
        REXGPU_ERROR("GL: failed to start the shader compile thread");
        running_ = false;
        async_ = false;
        delete t;
      } else {
        worker_thread_ = t;
      }
    }
  }
  return true;
}

void GlPipelineCache::Shutdown() {
  if (worker_thread_) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    cv_.notify_all();
    threadWaitForExit(static_cast<Thread*>(worker_thread_));
    threadClose(static_cast<Thread*>(worker_thread_));
    delete static_cast<Thread*>(worker_thread_);
    worker_thread_ = nullptr;
  }
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
                    translated_count_.load(), t->shader().ucode_data_hash(), t->modification(),
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
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = programs_.find(key);
  if (it == programs_.end()) {
    GlProgram& program = programs_[key];
    program.vertex = vertex;
    program.pixel = pixel;
    if (async_) {
      queue_.emplace_back(vertex, pixel);
      ++pending_;
      cv_.notify_one();
      return nullptr;
    }
    TranslateProgram(program);
    if (program.state.load() == GlProgram::kGlslReady) {
      StartLink(program);
      FinishLink(program, true);
    }
    return program.state.load() == GlProgram::kReady ? &program : nullptr;
  }
  GlProgram& p = it->second;
  switch (p.state.load()) {
    case GlProgram::kGlslReady:
      StartLink(p);
      return nullptr;
    case GlProgram::kLinking:
      FinishLink(p, false);
      return p.state.load() == GlProgram::kReady ? &p : nullptr;
    case GlProgram::kReady:
      return &p;
    default:
      return nullptr;
  }
}

void GlPipelineCache::WorkerThunk(void* self) {
  static_cast<GlPipelineCache*>(self)->WorkerMain();
}

// CPU-only worker: Xenos ucode -> SPIR-V -> GLSL. All GL calls stay on the command processor thread.
void GlPipelineCache::WorkerMain() {
  while (running_) {
    std::pair<GlShader::GlTranslation*, GlShader::GlTranslation*> job;
    GlProgram* program;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !running_ || !queue_.empty(); });
      if (!running_) {
        break;
      }
      job = queue_.front();
      queue_.pop_front();
      program = &programs_.find(ProgramKey{job.first, job.second})->second;
    }
    TranslateProgram(*program);
    --pending_;
  }
}

void GlPipelineCache::TranslateProgram(GlProgram& program) {
  GlShader::GlTranslation* vertex = program.vertex;
  GlShader::GlTranslation* pixel = program.pixel;
  if (!EnsureShadersTranslated(vertex, pixel)) {
    program.state = GlProgram::kFailed;
    return;
  }
  const GlslShader* vs_glsl = vertex->GetOrCreateGlsl();
  const GlslShader* ps_glsl = pixel ? pixel->GetOrCreateGlsl() : nullptr;
  if (!vs_glsl || (pixel && !ps_glsl)) {
    program.state = GlProgram::kFailed;
    return;
  }
  program.state = GlProgram::kGlslReady;
}

void GlPipelineCache::StartLink(GlProgram& program) {
  const GlslShader* vs_glsl = program.vertex->glsl();
  const GlslShader* ps_glsl = program.pixel ? program.pixel->glsl() : nullptr;
  program.link_start = std::chrono::steady_clock::now();
  auto make = [&](GLenum type, const std::string& source) {
    GLuint s = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);  // asynchronous with GL_KHR_parallel_shader_compile
    return s;
  };
  program.vs_id = make(GL_VERTEX_SHADER, vs_glsl->source);
  program.fs_id = ps_glsl ? make(GL_FRAGMENT_SHADER, ps_glsl->source) : 0;
  program.program = glCreateProgram();
  glAttachShader(program.program, program.vs_id);
  if (program.fs_id) {
    glAttachShader(program.program, program.fs_id);
  }
  glLinkProgram(program.program);
  program.state = GlProgram::kLinking;
}

void GlPipelineCache::FinishLink(GlProgram& program, bool block) {
  if (!block) {
    GLint done = 1;
    if (glMaxShaderCompilerThreadsKHR_ && REXCVAR_GET(gl_parallel_compile)) {
      glGetProgramiv(program.program, 0x91B1 /* GL_COMPLETION_STATUS_KHR */, &done);
    }
    if (!done) {
      return;
    }
  }
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
  GlShader::GlTranslation* vertex = program.vertex;
  GlShader::GlTranslation* pixel = program.pixel;
  GLint ok = 0;
  glGetProgramiv(program.program, GL_LINK_STATUS, &ok);
  std::string log;
  if (!ok) {
    auto shader_log = [&](GLuint s, const char* what) {
      GLint cs = 0;
      glGetShaderiv(s, GL_COMPILE_STATUS, &cs);
      if (!cs) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        log.assign(size_t(len > 1 ? len : 1), 0);
        glGetShaderInfoLog(s, len, nullptr, log.data());
        REXGPU_ERROR("GL: {} shader failed to compile: {}", what, log);
      }
    };
    shader_log(program.vs_id, "vertex");
    if (program.fs_id) {
      shader_log(program.fs_id, "pixel");
    }
    GLint len = 0;
    glGetProgramiv(program.program, GL_INFO_LOG_LENGTH, &len);
    log.assign(size_t(len > 1 ? len : 1), 0);
    glGetProgramInfoLog(program.program, len, nullptr, log.data());
    REXGPU_ERROR("GL: program link failed (vs {:016X}, ps {:016X}): {}",
                 vertex->shader().ucode_data_hash(),
                 pixel ? pixel->shader().ucode_data_hash() : uint64_t(0), log);
    dump("vs_fail", vertex->glsl()->source, vertex->shader().ucode_data_hash());
    if (pixel) {
      dump("ps_fail", pixel->glsl()->source, pixel->shader().ucode_data_hash());
    }
    glDeleteProgram(program.program);
    program.program = 0;
    program.state = GlProgram::kFailed;
  } else {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          program.link_start)
                    .count();
    static uint32_t linked = 0;
    uint32_t n = ++linked;
    if (n <= 8 || n % 25 == 0) {
      REXGPU_INFO("GL: program #{} ready {:.0f} ms after the link was issued (block={})", n, ms,
                  int(block));
    }
    if (dumped_ < 4) {
      ++dumped_;
      dump("vs_ok", vertex->glsl()->source, vertex->shader().ucode_data_hash());
      if (pixel) {
        dump("ps_ok", pixel->glsl()->source, pixel->shader().ucode_data_hash());
      }
    }
    program.state = GlProgram::kReady;
  }
  glDeleteShader(program.vs_id);
  if (program.fs_id) {
    glDeleteShader(program.fs_id);
  }
  program.vs_id = program.fs_id = 0;
}

}  // namespace rex::graphics::gl
